import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from s2mosaic.config import MosaicRequest
from s2mosaic.output import (
    _jsonable,
    _safe_token,
    finalize_output,
    get_output_path,
    output_request_hash,
    output_sidecar_metadata,
    resolve_export_path,
    write_output_sidecar,
)


class TestExportPaths:
    def test_auto_filename_uses_v2_readable_core(self, tmp_path):
        path = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            scene_order="oldest",
            mosaic_method="mean",
            bands=["B04", "B03", "B02"],
            grid_id="50HMH",
            source_name="MPC",
            resolution=10,
            cloud_mask="OCM",
            filename_hash="abc123def0",
        )

        assert path.name == (
            "grid-50HMH_2023-06-01_to_2023-06-08_"
            "B04-B03-B02_mean_scene-oldest_10m_OCM_MPC_abc123def0.tif"
        )

    def test_auto_percentile_filename_includes_percentile(self, tmp_path):
        p25 = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 8, 1),
            scene_order="valid_data",
            mosaic_method="percentile",
            percentile=25,
            bands=["B04"],
            bounds=(115.8301, -31.9702, 115.9103, -31.9404),
        )
        p75 = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 8, 1),
            scene_order="valid_data",
            mosaic_method="percentile",
            percentile=75,
            bands=["B04"],
            bounds=(115.8301, -31.9702, 115.9103, -31.9404),
        )

        assert "_percentile-p25_" in p25.name
        assert "_percentile-p75_" in p75.name
        assert p25 != p75

    def test_request_hash_changes_for_output_affecting_fields(self):
        start = date(2023, 6, 1)
        end = date(2023, 6, 8)
        base = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
        ).normalized()
        lower_resolution = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
            resolution=20,
        ).normalized()
        stricter_query = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
            additional_query={"eo:cloud_cover": {"lt": 20}},
        ).normalized()

        base_hash = output_request_hash(
            base,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        )

        assert base_hash != output_request_hash(
            lower_resolution,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        )
        assert base_hash != output_request_hash(
            stricter_query,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        )

    def test_request_hash_ignores_non_output_fields(self, tmp_path):
        start = date(2023, 6, 1)
        end = date(2023, 6, 8)
        base = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
        ).normalized()
        runtime_only = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
            output_dir=tmp_path / "one",
            output_path=tmp_path / "custom.tif",
            overwrite=False,
            show_progress=True,
            tile_workers=8,
        ).normalized()

        assert output_request_hash(
            base,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        ) == output_request_hash(
            runtime_only,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        )

    def test_request_hash_includes_callable_closure_values(self):
        start = date(2023, 6, 1)
        end = date(2023, 6, 8)

        def sort_above(threshold):
            def sort_fn(items):
                return items[items["score"] > threshold]

            return sort_fn

        low_threshold = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
            scene_sort_fn=sort_above(10),
        ).normalized()
        high_threshold = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
            scene_sort_fn=sort_above(20),
        ).normalized()

        assert output_request_hash(
            low_threshold,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        ) != output_request_hash(
            high_threshold,
            mode="grid",
            start_date=start,
            end_date=end,
            source_name="MPC",
        )

    def test_jsonable_callable_instance_uses_state_not_repr_address(self):
        class Sorter:
            def __init__(self, threshold):
                self.threshold = threshold

            def __call__(self, items):
                return items

        first = _jsonable(Sorter(10))
        second = _jsonable(Sorter(10))
        different = _jsonable(Sorter(20))

        assert first == second
        assert first != different
        assert "0x" not in json.dumps(first)

    def test_jsonable_rejects_callable_with_unserialisable_state(self):
        class Sorter:
            def __init__(self):
                self.handle = object()

            def __call__(self, items):
                return items

        with pytest.raises(ValueError, match="Callable request parameters"):
            _jsonable(Sorter())

    def test_aoi_filename_uses_geometry_hash(self, tmp_path):
        aoi = Polygon([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)])
        path = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            scene_order="valid_data",
            mosaic_method="mean",
            bands=["B04"],
            aoi=aoi,
            filename_hash="abc123def0",
        )

        assert path.name.startswith("aoi-")
        assert "abc123def0" in path.name
        assert "40.0" not in path.name

    def test_output_sidecar_metadata_is_written(self, tmp_path):
        request = MosaicRequest(
            grid_id="50HMH",
            start_year=2023,
            duration_days=7,
            bands=["B04"],
        ).normalized()
        metadata = output_sidecar_metadata(
            request,
            mode="grid",
            filename_hash="abc123def0",
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            source_name="MPC",
        )
        export_path = tmp_path / "mosaic.tif"

        write_output_sidecar(export_path, metadata)

        sidecar = json.loads(export_path.with_suffix(".json").read_text())
        assert sidecar["filename_hash"] == "abc123def0"
        assert sidecar["mode"] == "grid"
        assert sidecar["request"]["grid_id"] == "50HMH"
        assert sidecar["request"]["bands"] == ["B04"]

    def test_output_sidecar_write_is_atomic_on_failure(self, monkeypatch, tmp_path):
        export_path = tmp_path / "mosaic.tif"
        sidecar_path = tmp_path / "mosaic.json"
        sidecar_path.write_text("existing\n", encoding="utf-8")
        temp_paths = []

        class FailingTempFile:
            name = str(tmp_path / ".mosaic.tmp.json")

            def __enter__(self):
                Path(self.name).write_text("partial\n", encoding="utf-8")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, _payload):
                raise OSError("disk full")

        def fake_named_temporary_file(**_kwargs):
            temp_paths.append(Path(FailingTempFile.name))
            return FailingTempFile()

        monkeypatch.setattr(
            "s2mosaic.output.tempfile.NamedTemporaryFile",
            fake_named_temporary_file,
        )

        with pytest.raises(OSError, match="disk full"):
            write_output_sidecar(export_path, {"filename_hash": "new"})

        assert sidecar_path.read_text(encoding="utf-8") == "existing\n"
        assert temp_paths and not temp_paths[0].exists()

    def test_jsonable_rejects_recursive_metadata(self):
        recursive = {}
        recursive["self"] = recursive

        with pytest.raises(ValueError, match="recursive"):
            _jsonable(recursive)

    def test_safe_token_preserves_negative_and_decimal_distinctions(self):
        assert _safe_token(-1.25) == "neg1p2500"
        assert _safe_token("neg1p2500") == "neg1p2500"
        assert _safe_token(-1.25) != _safe_token("m1p2500")

    def test_finalize_output_accepts_lazy_array_like_coverage_mask(self):
        class LazyMask:
            def __array__(self, dtype=None):
                arr = np.array([[True, False], [False, True]], dtype=bool)
                return arr.astype(dtype, copy=False) if dtype is not None else arr

        array = np.ones((1, 2, 2), dtype=np.uint16) * 7
        result, _ = finalize_output(
            array=array,
            profile={"driver": "GTiff"},
            bands=["B04"],
            coverage_mask=LazyMask(),
            export_path=None,
        )

        np.testing.assert_array_equal(
            result,
            np.array([[[7, 0], [0, 7]]], dtype=np.uint16),
        )

    def test_finalize_output_export_does_not_mutate_profile(
        self, monkeypatch, tmp_path
    ):
        profile = {"driver": "GTiff", "width": 2, "height": 2}
        seen_profiles = []

        class FakeWriter:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def write(self, _array):
                return None

        def fake_open(_path, _mode, **kwargs):
            seen_profiles.append(kwargs)
            return FakeWriter()

        monkeypatch.setattr("s2mosaic.output.rio.open", fake_open)

        result = finalize_output(
            array=np.ones((1, 2, 2), dtype=np.uint16),
            profile=profile,
            bands=["B04"],
            coverage_mask=None,
            export_path=tmp_path / "out.tif",
        )

        assert result == tmp_path / "out.tif"
        assert profile == {"driver": "GTiff", "width": 2, "height": 2}
        assert seen_profiles[0]["count"] == 1

    def test_finalize_output_export_is_atomic_on_failure(self, monkeypatch, tmp_path):
        export_path = tmp_path / "out.tif"
        export_path.write_bytes(b"existing")
        temp_paths = []

        class FailingWriter:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, _array):
                raise OSError("write failed")

        def fake_open(path, _mode, **_kwargs):
            path = Path(path)
            temp_paths.append(path)
            path.write_bytes(b"partial")
            return FailingWriter()

        monkeypatch.setattr("s2mosaic.output.rio.open", fake_open)

        with pytest.raises(OSError, match="write failed"):
            finalize_output(
                array=np.ones((1, 2, 2), dtype=np.uint16),
                profile={"driver": "GTiff", "width": 2, "height": 2},
                bands=["B04"],
                coverage_mask=None,
                export_path=export_path,
            )

        assert export_path.read_bytes() == b"existing"
        assert temp_paths and not temp_paths[0].exists()

    def test_output_path_is_used_directly(self, tmp_path):
        path = resolve_export_path(
            output_dir=None,
            output_path=tmp_path / "nested" / "custom.tif",
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            scene_order="oldest",
            mosaic_method="mean",
            bands=["B04"],
            grid_id="50HMH",
        )

        assert path == tmp_path / "nested" / "custom.tif"
        assert path.parent.exists()

    def test_output_dir_and_output_path_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="Only one of output_dir or output_path"):
            resolve_export_path(
                output_dir=tmp_path,
                output_path=tmp_path / "custom.tif",
                start_date=date(2023, 6, 1),
                end_date=date(2023, 6, 8),
                scene_order="oldest",
                mosaic_method="mean",
                bands=["B04"],
                grid_id="50HMH",
            )

    def test_output_path_requires_tif_filename(self, tmp_path):
        with pytest.raises(ValueError, match="must include a .tif or .tiff filename"):
            resolve_export_path(
                output_dir=None,
                output_path=tmp_path / "custom",
                start_date=date(2023, 6, 1),
                end_date=date(2023, 6, 8),
                scene_order="oldest",
                mosaic_method="mean",
                bands=["B04"],
                grid_id="50HMH",
            )
