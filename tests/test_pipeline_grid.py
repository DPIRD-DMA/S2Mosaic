import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from s2mosaic.helpers import NoClearPixelsError, SceneFetchError
from s2mosaic.pipelines.grid import stream_mosaic_pipeline
from s2mosaic.sources import MPC
from s2mosaic.stac import ITEM_COL


class TestGridOrderedMaskStreaming:
    class FakeAsset:
        href = "sample.tif"

    class FakeItem:
        def __init__(self, scene_id):
            self.id = scene_id
            self.assets = {"B04": TestGridOrderedMaskStreaming.FakeAsset()}

    def _sorted_scenes(self, n_scenes):
        return pd.DataFrame(
            {ITEM_COL: [self.FakeItem(f"scene-{i}") for i in range(n_scenes)]}
        )

    def _patch_grid_pipeline_io(self, monkeypatch):
        import s2mosaic.pipelines.grid as core_mod

        monkeypatch.setattr(
            core_mod,
            "_build_output_profile",
            lambda sample_href_signed, s2_scene_size: {
                "driver": "GTiff",
                "dtype": np.dtype(np.uint16),
                "width": s2_scene_size,
                "height": s2_scene_size,
                "count": 1,
                "crs": None,
                "transform": from_origin(0, s2_scene_size, 1, 1),
            },
        )
        monkeypatch.setattr(
            core_mod,
            "make_grid_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, spec: np.ones(
                    (spec[2], spec[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            core_mod,
            "run_tile_aggregation",
            lambda **kwargs: np.ones(
                (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                dtype=np.uint16,
            ),
        )

    def test_grid_first_stops_mask_stream_after_coverage_is_filled(self, monkeypatch):
        import s2mosaic.pipelines.grid as core_mod

        self._patch_grid_pipeline_io(monkeypatch)
        calls = []

        def fake_compute_one_scene_mask(**kwargs):
            calls.append(kwargs["item"].id)
            return np.ones((4, 4), dtype=bool)

        monkeypatch.setattr(
            core_mod, "_compute_one_scene_mask", fake_compute_one_scene_mask
        )

        out, profile, _dropped = stream_mosaic_pipeline(
            sorted_scenes=self._sorted_scenes(4),
            bands=["B04"],
            coverage_mask=np.ones((4, 4), dtype=bool),
            mosaic_method="first",
            cloud_mask="SCL",
            source=MPC,
            s2_scene_size=4,
            tile_size=4,
            tile_workers=1,
        )

        assert calls == ["scene-0"]
        assert out is not None
        assert profile["width"] == 4

    def test_grid_reports_dropped_scenes_after_mask_fetch_failure(
        self, monkeypatch, capsys
    ):
        import s2mosaic.pipelines.grid as core_mod

        self._patch_grid_pipeline_io(monkeypatch)

        def fake_compute_one_scene_mask(**kwargs):
            if kwargs["item"].id == "scene-1":
                raise SceneFetchError("simulated transient network failure")
            return np.ones((4, 4), dtype=bool)

        monkeypatch.setattr(
            core_mod, "_compute_one_scene_mask", fake_compute_one_scene_mask
        )

        out, _profile, dropped = stream_mosaic_pipeline(
            sorted_scenes=self._sorted_scenes(3),
            bands=["B04"],
            coverage_mask=np.ones((4, 4), dtype=bool),
            mosaic_method="mean",
            cloud_mask="SCL",
            source=MPC,
            s2_scene_size=4,
            tile_size=4,
            tile_workers=1,
        )

        assert out is not None
        assert [d["id"] for d in dropped] == ["scene-1"]
        assert "simulated transient network failure" in dropped[0]["reason"]
        # The user-facing summary line must go to stderr regardless of logging
        # config, which is the whole point of report_dropped_scenes.
        captured = capsys.readouterr()
        assert "1/3 scenes dropped" in captured.err
        assert "scene-1" in captured.err


class TestGridNoClearPixels:
    """Issue #14: fully cloudy is not the same as failed to fetch.

    A scene that downloads fine but has no clear pixel is skipped without a
    stored mask, the same as a scene whose fetch failed. With nothing kept,
    the pipeline used to report every such scene as a fetch failure.
    """

    _io = TestGridOrderedMaskStreaming()

    def _run(self, monkeypatch, scene_masks, mosaic_method="mean"):
        import s2mosaic.pipelines.grid as core_mod

        self._io._patch_grid_pipeline_io(monkeypatch)

        def fake_compute_one_scene_mask(**kwargs):
            mask = scene_masks[kwargs["item"].id]
            if isinstance(mask, Exception):
                raise mask
            return mask

        monkeypatch.setattr(
            core_mod, "_compute_one_scene_mask", fake_compute_one_scene_mask
        )
        return stream_mosaic_pipeline(
            sorted_scenes=self._io._sorted_scenes(len(scene_masks)),
            bands=["B04"],
            coverage_mask=np.ones((4, 4), dtype=bool),
            mosaic_method=mosaic_method,
            cloud_mask="SCL",
            source=MPC,
            s2_scene_size=4,
            tile_size=4,
            tile_workers=1,
        )

    @pytest.mark.parametrize("mosaic_method", ["mean", "first", "percentile"])
    def test_all_scenes_fully_masked_raises_no_clear_pixels(
        self, monkeypatch, mosaic_method
    ):
        cloudy = np.zeros((4, 4), dtype=bool)

        with pytest.raises(NoClearPixelsError) as exc_info:
            self._run(
                monkeypatch,
                {"scene-0": cloudy, "scene-1": cloudy},
                mosaic_method=mosaic_method,
            )

        message = str(exc_info.value)
        assert "failed to fetch" not in message
        assert "2 scenes were fully masked" in message
        assert exc_info.value.n_scenes == 2
        assert exc_info.value.n_failed == 0
        # Existing ``except RuntimeError`` handlers keep catching it.
        assert isinstance(exc_info.value, RuntimeError)

    def test_all_scenes_failed_to_fetch_is_not_no_clear_pixels(self, monkeypatch):
        failure = SceneFetchError("simulated transient network failure")

        with pytest.raises(RuntimeError, match="failed to fetch") as exc_info:
            self._run(monkeypatch, {"scene-0": failure, "scene-1": failure})

        assert not isinstance(exc_info.value, NoClearPixelsError)

    def test_mixed_failed_and_fully_masked_reports_both_counts(self, monkeypatch):
        with pytest.raises(NoClearPixelsError) as exc_info:
            self._run(
                monkeypatch,
                {
                    "scene-0": np.zeros((4, 4), dtype=bool),
                    "scene-1": SceneFetchError("simulated failure"),
                    "scene-2": np.zeros((4, 4), dtype=bool),
                },
            )

        message = str(exc_info.value)
        assert "2 fully masked" in message
        assert "1 failed to fetch" in message
        assert exc_info.value.n_scenes == 3
        assert exc_info.value.n_failed == 1
