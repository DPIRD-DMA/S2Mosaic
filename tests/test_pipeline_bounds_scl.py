"""SCL mask-fetch helpers for bounds/AOI mosaics.

These drive the real rasterio/WarpedVRT path against a tiny GeoTIFF written to
``tmp_path`` rather than mocking it, so the window arithmetic and the
block-count estimate are exercised as they run in production. The only thing
stubbed out is the network: the source signs hrefs with identity and the STAC
item is a stand-in exposing just the ``assets`` mapping these helpers read.

Until now this module's coverage came entirely from the ``slow`` end-to-end
tests, which are excluded from the default run, so a regression here reached a
release without any local or CI failure.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import Affine

from s2mosaic.pipelines.bounds_scl import (
    _fetch_one_scl,
    _fetch_one_scl_tiled,
    _should_use_tiled_scl_fetch,
    _source_block_count_for_scl_tiles,
)
from s2mosaic.sources import Source

TARGET_CRS = 32750
RESOLUTION = 20
BLOCK = 64
SIZE = 256  # 4 x 4 source blocks at BLOCK=64
MINX, MAXY = 100_000.0, 200_000.0
BOUNDS = (MINX, MAXY - SIZE * RESOLUTION, MINX + SIZE * RESOLUTION, MAXY)
FULL_SPEC = [(0, 0, SIZE, SIZE)]

TEST_SOURCE = Source(
    name="TEST",
    stac_url="",
    collection_id="",
    sign=lambda href: href,
)


@pytest.fixture
def scl_item(tmp_path):
    """A stand-in STAC item whose SCL asset is a local tiled GeoTIFF.

    Each pixel carries ``row // BLOCK * 4 + col // BLOCK``, i.e. its source
    block index, so a fetch's values identify which part of the source it
    actually read.
    """
    path = tmp_path / "scl.tif"
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    data = ((rows // BLOCK) * 4 + (cols // BLOCK)).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": SIZE,
        "height": SIZE,
        "crs": f"EPSG:{TARGET_CRS}",
        "transform": Affine(RESOLUTION, 0, MINX, 0, -RESOLUTION, MAXY),
        "tiled": True,
        "blockxsize": BLOCK,
        "blockysize": BLOCK,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return SimpleNamespace(
        id="test-item",
        assets={"SCL": SimpleNamespace(href=str(path))},
        bbox=None,
        geometry=None,
    )


def _block_count(item, tile_specs):
    return _source_block_count_for_scl_tiles(
        item, TEST_SOURCE, BOUNDS, TARGET_CRS, RESOLUTION, SIZE, SIZE, tile_specs
    )


class TestFetchOneScl:
    def test_reads_only_the_scene_window(self, scl_item):
        # A scene covering the second block row/column only. The fetch is
        # sized to the window, not to the whole bounds, which is what keeps
        # per-scene SCL cost bounded by scene size on a wide AOI.
        scene_window = (BLOCK, BLOCK, BLOCK, BLOCK)

        fetch = _fetch_one_scl(
            scl_item, TEST_SOURCE, BOUNDS, TARGET_CRS, RESOLUTION, scene_window
        )

        assert fetch.arr.shape == (BLOCK, BLOCK)
        assert fetch.arr.dtype == np.uint8
        assert fetch.target_window == scene_window
        assert fetch.crop == (slice(0, BLOCK), slice(0, BLOCK))
        # Block (1, 1) of the source is encoded as 1 * 4 + 1.
        np.testing.assert_array_equal(fetch.arr, 5)


class TestFetchOneSclTiled:
    def test_full_extent_spec_delegates_to_the_plain_fetch(self, scl_item):
        scene_window = (0, 0, SIZE, SIZE)

        tiled = _fetch_one_scl_tiled(
            scl_item,
            TEST_SOURCE,
            BOUNDS,
            TARGET_CRS,
            RESOLUTION,
            SIZE,
            SIZE,
            FULL_SPEC,
            scene_window,
        )
        plain = _fetch_one_scl(
            scl_item, TEST_SOURCE, BOUNDS, TARGET_CRS, RESOLUTION, scene_window
        )

        assert tiled.target_window == plain.target_window
        assert tiled.crop == plain.crop
        np.testing.assert_array_equal(tiled.arr, plain.arr)

    def test_grid_that_disagrees_with_the_bounds_is_rejected(self, scl_item):
        # The tile specs index a grid the caller computed separately. If it
        # does not match the one these bounds imply, the windows would read
        # the wrong pixels, so fail loudly rather than silently misalign.
        with pytest.raises(ValueError, match="does not match requested bounds grid"):
            _fetch_one_scl_tiled(
                scl_item,
                TEST_SOURCE,
                BOUNDS,
                TARGET_CRS,
                RESOLUTION,
                SIZE + 1,
                SIZE,
                [(0, 0, BLOCK, BLOCK)],
                (0, 0, SIZE, SIZE),
            )

    def test_sparse_tiles_return_their_union_with_gaps_zeroed(self, scl_item):
        # Two tiles on a diagonal. The result is the bbox spanning both, so
        # the off-diagonal corners are inside the returned array but were
        # never read and must stay zero.
        tile_specs = [(0, 0, BLOCK, BLOCK), (BLOCK, BLOCK, BLOCK, BLOCK)]

        fetch = _fetch_one_scl_tiled(
            scl_item,
            TEST_SOURCE,
            BOUNDS,
            TARGET_CRS,
            RESOLUTION,
            SIZE,
            SIZE,
            tile_specs,
            (0, 0, SIZE, SIZE),
        )

        assert fetch.target_window == (0, 0, 2 * BLOCK, 2 * BLOCK)
        assert fetch.crop == (slice(0, 2 * BLOCK), slice(0, 2 * BLOCK))
        np.testing.assert_array_equal(fetch.arr[:BLOCK, :BLOCK], 0)  # block (0, 0)
        np.testing.assert_array_equal(fetch.arr[BLOCK:, BLOCK:], 5)  # block (1, 1)
        np.testing.assert_array_equal(fetch.arr[:BLOCK, BLOCK:], 0)  # never read
        np.testing.assert_array_equal(fetch.arr[BLOCK:, :BLOCK], 0)  # never read

    def test_tiles_are_clipped_to_the_scene_window(self, scl_item):
        # A tile spanning the whole width against a scene occupying only the
        # right half: the read must cover the intersection, not the tile.
        fetch = _fetch_one_scl_tiled(
            scl_item,
            TEST_SOURCE,
            BOUNDS,
            TARGET_CRS,
            RESOLUTION,
            SIZE,
            SIZE,
            [(0, 0, BLOCK, SIZE)],
            (SIZE // 2, 0, SIZE // 2, BLOCK),
        )

        assert fetch.target_window == (SIZE // 2, 0, SIZE // 2, BLOCK)

    def test_scene_disjoint_from_every_tile_returns_zeros(self, scl_item):
        # No intersection at all: return an empty block sized to the scene
        # rather than reading anything.
        scene_window = (0, 0, BLOCK, BLOCK)

        fetch = _fetch_one_scl_tiled(
            scl_item,
            TEST_SOURCE,
            BOUNDS,
            TARGET_CRS,
            RESOLUTION,
            SIZE,
            SIZE,
            [(2 * BLOCK, 2 * BLOCK, BLOCK, BLOCK)],
            scene_window,
        )

        assert fetch.target_window == scene_window
        np.testing.assert_array_equal(fetch.arr, 0)


class TestSourceBlockCountForSclTiles:
    def test_grid_that_disagrees_with_the_bounds_is_rejected(self, scl_item):
        with pytest.raises(ValueError, match="does not match requested bounds grid"):
            _source_block_count_for_scl_tiles(
                scl_item,
                TEST_SOURCE,
                BOUNDS,
                TARGET_CRS,
                RESOLUTION,
                SIZE,
                SIZE + 1,
                FULL_SPEC,
            )

    def test_full_extent_touches_every_source_block(self, scl_item):
        assert _block_count(scl_item, FULL_SPEC) == 16  # 4 x 4 blocks

    def test_one_tile_touches_only_its_own_block(self, scl_item):
        assert _block_count(scl_item, [(0, 0, BLOCK, BLOCK)]) == 1

    def test_blocks_shared_between_tiles_are_counted_once(self, scl_item):
        # Two tiles that each land inside source block (0, 0). The estimate
        # is of bytes fetched, so the shared block must not count twice.
        halved = BLOCK // 2
        overlapping = [(0, 0, halved, halved), (halved, halved, halved, halved)]
        assert _block_count(scl_item, overlapping) == 1


class TestShouldUseTiledSclFetch:
    def _should(self, items, tile_specs):
        return _should_use_tiled_scl_fetch(
            items, TEST_SOURCE, BOUNDS, TARGET_CRS, RESOLUTION, SIZE, SIZE, tile_specs
        )

    def test_full_extent_spec_is_never_tiled(self, scl_item):
        assert self._should(scl_item, FULL_SPEC) is False

    def test_no_items_is_never_tiled(self):
        assert self._should([], [(0, 0, BLOCK, BLOCK)]) is False

    def test_sparse_tiles_that_save_blocks_are_tiled(self, scl_item):
        # One block out of sixteen, far below the saving threshold.
        assert self._should(scl_item, [(0, 0, BLOCK, BLOCK)]) is True

    def test_tiles_covering_everything_are_not_tiled(self, scl_item):
        # Split in half rather than one full-extent spec, so the early return
        # above does not apply: the specs still touch every source block, so
        # sparse reads would fetch the same bytes with more requests.
        covering = [(0, 0, SIZE // 2, SIZE), (SIZE // 2, 0, SIZE // 2, SIZE)]
        assert self._should(scl_item, covering) is False

    def test_a_single_item_is_accepted_without_a_sequence(self, scl_item):
        # Callers pass either one item or a collection; both must work.
        assert self._should([scl_item], [(0, 0, BLOCK, BLOCK)]) is True

    def test_a_degenerate_block_estimate_falls_back_to_whole_reads(
        self, scl_item, monkeypatch
    ):
        # If the estimate cannot find any source blocks, the ratio below it
        # would compare 0 <= 0 and conclude that tiling saves everything.
        # Guard against that by staying on the whole-extent read.
        import s2mosaic.pipelines.bounds_scl as scl_mod

        monkeypatch.setattr(
            scl_mod, "_source_block_count_for_scl_tiles", lambda *a, **k: 0
        )

        assert self._should(scl_item, [(0, 0, BLOCK, BLOCK)]) is False

    def test_at_most_five_items_are_sampled(self, scl_item, monkeypatch):
        # The estimate opens the source COG per item, so a long scene list
        # must not turn the decision into a per-scene cost.
        import s2mosaic.pipelines.bounds_scl as scl_mod

        seen = []
        real = scl_mod._source_block_count_for_scl_tiles

        def counting(item, *args, **kwargs):
            seen.append(item)
            return real(item, *args, **kwargs)

        monkeypatch.setattr(scl_mod, "_source_block_count_for_scl_tiles", counting)

        assert self._should([scl_item] * 20, [(0, 0, BLOCK, BLOCK)]) is True
        # Five sampled items, each costed twice (full extent against tiled).
        assert len(seen) == 10
