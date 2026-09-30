"""Retries must refetch after a truncated HTTP read, not replay GDAL's cache.

On a flaky network a range response can be cut short mid-body. GDAL keeps
the short block in its process-wide ``/vsicurl/`` cache, so a plain reopen
of the same URL is served that block again without a new request, and a
retry fails exactly like the first attempt (the real symptom was libtiff
reporting ``got 47864 bytes, expected 399882`` on every attempt while the
same file downloaded fine with curl). These tests run a local server that
truncates the first few data-range responses and then behaves.
"""

import http.server
import re
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Tuple

import numpy as np
import pytest
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

from s2mosaic.gdal_env import (
    apply_gdal_network_defaults,
    fresh_remote_reads,
    propagate_remote_read_env,
    restore_gdal_network_env,
)
from s2mosaic.geometry import _OCM_BANDS
from s2mosaic.helpers import SceneFetchError, with_scene_retry
from s2mosaic.pipelines.bounds import _fetch_one_ocm
from s2mosaic.readers import GridTileReader, _HandleCache, _retry_open_raster

SIZE = 1024


class TruncatingServer:
    """Serve one directory with Range support; truncate the first N data ranges.

    Headers promise the full range, then the connection closes after an
    eighth of the body, the way a dropped connection looks to the client.
    The first bytes of the file (the COG header) are never truncated, so the
    open succeeds and the failure lands on a tile read.
    """

    def __init__(self, root: Path, truncate_first: int) -> None:
        self.root = root
        self.truncate_first = truncate_first
        self.truncated = 0
        self.requests = 0
        self._lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_HEAD(self) -> None:
                size = (
                    (server.root / self.path.lstrip("/").split("?")[0]).stat().st_size
                )
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self) -> None:
                path = server.root / self.path.lstrip("/").split("?")[0]
                data = path.read_bytes()
                with server._lock:
                    server.requests += 1
                m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if m is None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                a = int(m.group(1))
                b = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
                body = data[a : b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with server._lock:
                    cut = a > 0 and server.truncated < server.truncate_first
                    if cut:
                        server.truncated += 1
                if cut:
                    self.wfile.write(body[: max(1, len(body) // 8)])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                self.wfile.write(body)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self._httpd = Server(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def url(self, name: str) -> str:
        # Each test writes its COG under a unique name (see the ``cog``
        # fixture), so GDAL's process-wide cache never carries state between
        # tests.
        return f"http://127.0.0.1:{self.port}/{name}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def cog(tmp_path: Path) -> Tuple[Path, np.ndarray]:
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 10000, (SIZE, SIZE), dtype=np.uint16)
    path = tmp_path / f"cog_{tmp_path.name}.tif"
    with rio.open(
        path,
        "w",
        driver="COG",
        width=SIZE,
        height=SIZE,
        count=1,
        dtype="uint16",
        crs="EPSG:32650",
        transform=from_origin(500000, 7000000, 10, 10),
        blocksize=256,
        compress="DEFLATE",
        overviews="NONE",
    ) as dst:
        dst.write(truth, 1)
    return path, truth


@pytest.fixture
def gdal_defaults() -> Iterator[None]:
    snapshot = apply_gdal_network_defaults()
    try:
        yield
    finally:
        restore_gdal_network_env(snapshot)


def _serve(cog: Tuple[Path, np.ndarray], truncate_first: int) -> TruncatingServer:
    return TruncatingServer(cog[0].parent, truncate_first)


@pytest.mark.usefixtures("gdal_defaults")
class TestRetryRefetches:
    def test_plain_reopen_replays_the_truncated_block(self, cog):
        """The GDAL behaviour the fix works around.

        One truncated response, then a healthy server: a plain reopen still
        fails and sends no new request. If a future GDAL stops caching short
        blocks this skips rather than fails, since the fix is then redundant
        but harmless.
        """
        server = _serve(cog, truncate_first=1)
        try:
            url = server.url(cog[0].name)
            with pytest.raises(RasterioIOError):
                with rio.open(url) as src:
                    src.read(1)
            requests_before = server.requests
            try:
                with rio.open(url) as src:
                    src.read(1)
            except RasterioIOError:
                assert server.requests == requests_before, (
                    "the failing reopen should have been served from cache"
                )
            else:
                pytest.skip("this GDAL does not replay truncated blocks")
        finally:
            server.close()

    def test_fresh_remote_reads_refetches_and_recovers(self, cog):
        server = _serve(cog, truncate_first=1)
        try:
            url = server.url(cog[0].name)
            with pytest.raises(RasterioIOError):
                with rio.open(url) as src:
                    src.read(1)
            requests_before = server.requests
            with fresh_remote_reads():
                with rio.open(url) as src:
                    got = src.read(1)
            assert server.requests > requests_before
            np.testing.assert_array_equal(got, cog[1])
        finally:
            server.close()

    def test_handle_opened_fresh_stays_fresh_after_the_context(self, cog):
        """Phase 2 reopens a handle under the context and reads it later."""
        server = _serve(cog, truncate_first=1)
        try:
            url = server.url(cog[0].name)
            with pytest.raises(RasterioIOError):
                with rio.open(url) as src:
                    src.read(1)
            with fresh_remote_reads():
                src = rio.open(url)
            try:
                got = src.read(1)
            finally:
                src.close()
            np.testing.assert_array_equal(got, cog[1])
        finally:
            server.close()

    def test_later_plain_opens_see_clean_data(self, cog):
        server = _serve(cog, truncate_first=1)
        try:
            url = server.url(cog[0].name)
            with pytest.raises(RasterioIOError):
                with rio.open(url) as src:
                    src.read(1)
            with fresh_remote_reads():
                with rio.open(url) as src:
                    src.read(1)
            with rio.open(url) as src:
                np.testing.assert_array_equal(src.read(1), cog[1])
        finally:
            server.close()


@pytest.mark.usefixtures("gdal_defaults")
class TestSceneRetryRecovers:
    def _fetcher(self, url: str):
        @with_scene_retry(attempts=3, base_delay=0.01)
        def fetch() -> np.ndarray:
            with rio.open(url) as src:
                return src.read(1)

        return fetch

    def test_recovers_after_two_truncated_reads(self, cog):
        server = _serve(cog, truncate_first=2)
        try:
            got = self._fetcher(server.url(cog[0].name))()
            np.testing.assert_array_equal(got, cog[1])
            assert server.truncated == 2
        finally:
            server.close()

    def test_gives_up_when_every_attempt_is_truncated(self, cog):
        server = _serve(cog, truncate_first=10)
        try:
            with pytest.raises(SceneFetchError):
                self._fetcher(server.url(cog[0].name))()
            # every attempt reached the server rather than the cache
            assert server.truncated == 3
        finally:
            server.close()


@pytest.mark.usefixtures("gdal_defaults")
class TestSceneRetryFromWorkerThread:
    """Scene fetches run on streaming workers and read bands on a sub-pool.

    ``rasterio.Env`` entered off the main thread applies to that thread only,
    so a retry there must carry the uncached setting into its sub-pool
    workers, or their opens replay the truncated block. The tests above all
    retry on the main thread, where the setting happens to be process-wide.
    """

    def test_sub_pool_reads_refetch(self, cog):
        server = _serve(cog, truncate_first=1)
        url = server.url(cog[0].name)

        def read(u: str) -> np.ndarray:
            with rio.open(u) as src:
                return src.read(1)

        @with_scene_retry(attempts=3, base_delay=0.01)
        def fetch() -> np.ndarray:
            with ThreadPoolExecutor(max_workers=3) as pool:
                return list(pool.map(propagate_remote_read_env(read), [url]))[0]

        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                got = worker.submit(fetch).result()
            np.testing.assert_array_equal(got, cog[1])
        finally:
            server.close()

    def test_bounds_ocm_fetch_recovers(self, cog, monkeypatch):
        monkeypatch.setattr("s2mosaic.helpers.backoff_delay", lambda *a, **k: 0.0)
        server = _serve(cog, truncate_first=1)
        url = server.url(cog[0].name)
        item = SimpleNamespace(
            id="scene",
            assets={b: SimpleNamespace(href=url) for b in _OCM_BANDS},
        )
        source = SimpleNamespace(asset_name=lambda b: b, sign=lambda h: h)
        bounds = (500000.0, 7000000.0 - SIZE * 10, 500000.0 + SIZE * 10, 7000000.0)
        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                fetched = worker.submit(
                    _fetch_one_ocm, item, source, bounds, 32650, 10, (0, 0, SIZE, SIZE)
                ).result()
            for band in fetched.arr[:, fetched.crop[0], fetched.crop[1]]:
                np.testing.assert_array_equal(band, cog[1])
        finally:
            server.close()


@pytest.mark.usefixtures("gdal_defaults")
class TestTileReaderRecovers:
    def test_refresh_open_is_uncached(self, cog):
        server = _serve(cog, truncate_first=1)
        try:
            url = server.url(cog[0].name)
            with pytest.raises(RasterioIOError):
                with rio.open(url) as src:
                    src.read(1)
            src = _retry_open_raster(lambda refresh: rio.open(url), refresh=True)
            try:
                np.testing.assert_array_equal(src.read(1), cog[1])
            finally:
                src.close()
        finally:
            server.close()

    def test_grid_tile_reader_recovers_through_reopen(self, cog, monkeypatch):
        monkeypatch.setattr("s2mosaic.readers.backoff_delay", lambda attempt: 0.0)
        server = _serve(cog, truncate_first=2)
        try:
            url = server.url(cog[0].name)
            cache = _HandleCache([[lambda refresh=False: url]])
            reader = GridTileReader(
                cache,
                href_band_indices=[1],
                s2_scene_size=SIZE,
                rio_resampling=Resampling.nearest,
            )
            try:
                got = reader(0, 0, (0, 0, SIZE, SIZE))
            finally:
                reader.close()
            np.testing.assert_array_equal(got, cog[1])
            assert server.truncated == 2
        finally:
            server.close()
