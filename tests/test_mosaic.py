import pytest
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import sys


# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic import mosaic
from s2mosaic.helpers import validate_inputs


class TestMosaicInputValidation:
    """Test input validation for the mosaic function"""

    def test_invalid_grid_id_lowercase(self):
        """Test that lowercase grid IDs are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("50hmh", 2023)

    def test_invalid_grid_id_special_chars(self):
        """Test that grid IDs with special characters are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("50H-MH", 2023)

    def test_invalid_grid_id_numbers_only(self):
        """Test that numeric-only grid IDs are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("12345", 2023)

    def test_invalid_sort_method(self):
        """Test that invalid sort methods are rejected"""
        with pytest.raises(ValueError, match="Invalid sort method"):
            mosaic("50HMH", 2023, sort_method="invalid_method")

    def test_invalid_mosaic_method(self):
        """Test that invalid mosaic methods are rejected"""
        with pytest.raises(ValueError, match="Invalid mosaic method"):
            mosaic("50HMH", 2023, mosaic_method="invalid_method")

    def test_invalid_no_data_threshold_negative(self):
        """Test that negative no_data_threshold is rejected"""
        with pytest.raises(
            ValueError, match="No data threshold must be between 0 and 1"
        ):
            mosaic("50HMH", 2023, no_data_threshold=-0.1)

    def test_invalid_no_data_threshold_greater_than_one(self):
        """Test that no_data_threshold > 1 is rejected"""
        with pytest.raises(
            ValueError, match="No data threshold must be between 0 and 1"
        ):
            mosaic("50HMH", 2023, no_data_threshold=1.5)

    def test_invalid_band(self):
        """Test that invalid band names are rejected"""
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic("50HMH", 2023, required_bands=["B04", "INVALID_BAND"])

    def test_visual_band_with_other_bands(self):
        """Test that visual band cannot be used with other bands"""
        with pytest.raises(ValueError, match="Cannot use visual band with other bands"):
            mosaic("50HMH", 2023, required_bands=["visual", "B04"])

    def test_percentile_without_percentile_method(self):
        """Test that percentile parameter requires percentile method"""
        with pytest.raises(
            ValueError,
            match="percentile_value is only valid for percentile mosaic method",
        ):
            mosaic("50HMH", 2023, mosaic_method="mean", percentile_value=50)

    def test_percentile_method_without_percentile(self):
        """Test that percentile method requires percentile parameter"""
        with pytest.raises(
            ValueError,
            match="percentile_value must be provided for percentile mosaic method",
        ):
            mosaic("50HMH", 2023, mosaic_method="percentile")

    def test_invalid_percentile_negative(self):
        """Test that negative percentile values are rejected"""
        with pytest.raises(
            ValueError, match="percentile_value must be between 0 and 100"
        ):
            mosaic("50HMH", 2023, mosaic_method="percentile", percentile_value=-10)

    def test_invalid_percentile_greater_than_100(self):
        """Test that percentile values > 100 are rejected"""
        with pytest.raises(
            ValueError, match="percentile_value must be between 0 and 100"
        ):
            mosaic("50HMH", 2023, mosaic_method="percentile", percentile_value=150)


class TestMosaicValidInputs:
    """Verify validate_inputs() accepts known-good parameter combinations.

    Calls validate_inputs() directly to avoid the network/processing cost of a
    full mosaic() run — these tests only care about validation behaviour.
    """

    DEFAULT_KWARGS = {
        "sort_method": "valid_data",
        "mosaic_method": "mean",
        "no_data_threshold": 0.01,
        "required_bands": ["B04", "B03", "B02", "B08"],
        "grid_id": "50HMH",
        "percentile_value": None,
    }

    def _validate(self, **overrides):
        validate_inputs(**{**self.DEFAULT_KWARGS, **overrides})

    def test_valid_grid_id(self):
        self._validate()

    @pytest.mark.parametrize("method", ["valid_data", "oldest", "newest"])
    def test_valid_sort_methods(self, method):
        self._validate(sort_method=method)

    @pytest.mark.parametrize("method", ["mean", "first"])
    def test_valid_mosaic_methods(self, method):
        self._validate(mosaic_method=method)

    def test_valid_percentile_method(self):
        self._validate(mosaic_method="percentile", percentile_value=50.0)

    @pytest.mark.parametrize("threshold", [0.0, 0.01, 0.5, 1.0, None])
    def test_valid_no_data_thresholds(self, threshold):
        self._validate(no_data_threshold=threshold)

    @pytest.mark.parametrize(
        "bands",
        [
            ["B04", "B03", "B02"],
            ["B04", "B03", "B02", "B08"],
            ["visual"],
            [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            ],
            ["AOT", "SCL", "WVP"],
        ],
    )
    def test_valid_bands(self, bands):
        self._validate(required_bands=bands)


@pytest.mark.slow
class TestMosaicEndToEnd:
    """End-to-end tests using debug cache for performance"""

    @pytest.fixture(autouse=True)
    def enable_debug_cache(self, monkeypatch):
        """Cache STAC/COG fetches across this class to keep wall time low."""
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

    @pytest.fixture(autouse=True)
    def time_test(self):
        """Print duration for tests that take more than a minute."""
        start_time = time.time()
        yield
        duration = time.time() - start_time
        if duration > 60:
            print(f"\n  {self.__class__.__name__} took {duration:.1f} seconds")

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_mosaic_return_array_no_output_dir(self):
        """Test mosaic returns array and profile when no output_dir is specified"""
        print("\n🔄 Starting test_mosaic_return_array_no_output_dir")

        try:
            result = mosaic(
                "50HMH",
                2023,
                start_month=6,
                start_day=1,
                duration_days=7,
                required_bands=["B04", "B03", "B02"],
            )
            print(f"✅ Mosaic function completed, result type: {type(result)}")
        except Exception as e:
            print(f"❌ Mosaic function failed with error: {e}")
            raise

        # Check if result is a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        print(f"✅ Result is tuple: {isinstance(result, tuple)}")

        # Check tuple length
        assert len(result) == 2, f"Expected tuple of length 2, got length {len(result)}"
        print(f"✅ Tuple length is 2: {len(result) == 2}")

        array, profile = result

        # Check array type
        assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
        print(f"✅ Array is numpy.ndarray: {isinstance(array, np.ndarray)}")

        # Check profile is dict-like (has keys, values, items methods)
        assert (
            hasattr(profile, "keys")
            and hasattr(profile, "values")
            and hasattr(profile, "items")
        ), f"Profile should be dict-like, got {type(profile)}"
        print(f"✅ Profile is dict-like: {type(profile)}")

        # Check profile has expected rasterio keys
        expected_keys = {
            "driver",
            "dtype",
            "width",
            "height",
            "count",
            "crs",
            "transform",
        }
        profile_keys = set(profile.keys())
        missing_keys = expected_keys - profile_keys
        assert len(missing_keys) == 0, f"Profile missing expected keys: {missing_keys}"
        print(f"✅ Profile has expected keys: {profile_keys}")

        # Check array dimensions
        assert array.ndim == 3, (
            f"Expected 3D array, got {array.ndim}D array with shape {array.shape}"
        )
        print(f"✅ Array is 3D: {array.ndim == 3}, shape: {array.shape}")

        # Check number of bands
        assert array.shape[0] == 3, f"Expected 3 bands, got {array.shape[0]} bands"
        print(f"✅ Array has 3 bands: {array.shape[0] == 3}")

        # Check data type
        valid_dtypes = [np.uint8, np.int16, np.uint16, np.float32]
        assert array.dtype in valid_dtypes, (
            f"Expected dtype in {valid_dtypes}, got {array.dtype}"
        )
        print(f"✅ Array dtype is valid: {array.dtype} in {valid_dtypes}")

        print("🎉 All assertions passed!")

    def test_mosaic_save_to_file(self, temp_output_dir):
        """Test mosaic saves file when output_dir is specified"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            required_bands=["B04", "B03", "B02"],
        )

        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".tif"
        assert result.parent == temp_output_dir

    def test_mosaic_overwrite_false(self, temp_output_dir):
        """Test that existing files are not overwritten when overwrite=False"""
        # Create first mosaic
        result1 = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            required_bands=["B04", "B03", "B02"],
        )

        # Get file modification time
        original_mtime = result1.stat().st_mtime

        # Try to create same mosaic with overwrite=False
        result2 = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            required_bands=["B04", "B03", "B02"],
            overwrite=False,
        )

        # Should return same path without creating new file
        assert result1 == result2
        assert result2.stat().st_mtime == original_mtime

    @pytest.mark.parametrize("sort_method", ["valid_data", "oldest", "newest"])
    def test_mosaic_different_sort_methods(self, sort_method):
        """Test mosaic with different sort methods"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            sort_method=sort_method,
            required_bands=["B04"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        assert hasattr(profile, "keys")
        assert array.shape[0] == 1

    @pytest.mark.parametrize("mosaic_method", ["mean", "first"])
    def test_mosaic_different_mosaic_methods(self, mosaic_method):
        """Test mosaic with different mosaic methods"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            mosaic_method=mosaic_method,
            required_bands=["B04"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    @pytest.mark.parametrize("percentile", [10, 50, 90])
    def test_mosaic_percentile_method(self, percentile):
        """Test mosaic with percentile method"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            mosaic_method="percentile",
            percentile_value=percentile,
            required_bands=["B04"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    def test_mosaic_visual_band(self):
        """Test mosaic with visual band"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            required_bands=["visual"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        assert hasattr(profile, "keys")  # Check it's dict-like
        assert hasattr(profile, "keys")  # Check it's dict-like
        assert array.shape[0] == 3  # RGB channels
        assert array.dtype == np.uint8  # Visual should be uint8

    @pytest.mark.parametrize("duration_days", [7, 14, 20, 21])
    def test_mosaic_different_time_ranges(self, duration_days):
        """Test mosaic with different time range specifications"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=duration_days,
            required_bands=["B04"],
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    @pytest.mark.parametrize("start_month, cloud_cover_lt", [(1, 50), (6, 20), (6, 10)])
    def test_mosaic_different_cloud_cover_thresholds(self, start_month, cloud_cover_lt):
        """Test mosaic with different cloud cover thresholds"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=start_month,
            start_day=1,
            duration_months=1,
            required_bands=["B04"],
            additional_query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    def test_mosaic_no_scenes_found(self):
        """Test behaviour when no scenes are found"""
        # Use a very restrictive date range and cloud cover that likely has no data
        with pytest.raises(Exception, match="No scenes found"):
            mosaic(
                "50HMH",
                2023,
                start_month=6,
                start_day=1,
                duration_days=1,  # Single day
                required_bands=["B04"],
                additional_query={
                    "eo:cloud_cover": {"lt": 0.1}
                },  # Very low cloud cover
            )

    def test_mosaic_custom_sort_function(self):
        """Test mosaic with custom sort function"""

        def custom_sort(items):
            # Sort by datetime ascending (oldest first)
            return items.sort_values("datetime", ascending=True).reset_index(drop=True)

        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            sort_function=custom_sort,
            required_bands=["B04"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    @pytest.mark.parametrize("cloud_mask", ["OCM", "SCL"])
    def test_mosaic_cloud_mask_providers(self, cloud_mask):
        """Both OCM and SCL providers should produce a sane mosaic in grid mode."""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            required_bands=["B04", "B03", "B02"],
            cloud_mask=cloud_mask,
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert array.ndim == 3 and array.shape[0] == 3
        assert array.dtype in (np.uint8, np.uint16, np.int16)
        # Some non-zero data — June over land
        assert array.max() > 0


@pytest.mark.slow
class TestMosaicFileNaming:
    """Test file naming conventions"""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_filename_format(self, temp_output_dir):
        """Test that output filenames follow expected format"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=6,
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            sort_method="oldest",
            mosaic_method="mean",
            required_bands=["B04", "B03", "B02"],
        )

        expected_pattern = "50HMH_2023-06-01_to_2023-06-08_oldest_mean_B04_B03_B02.tif"
        assert result.name == expected_pattern

    def test_filename_different_parameters(self, temp_output_dir):
        """Test filename changes with different parameters"""
        result = mosaic(
            "50HMH",
            2022,
            start_month=12,
            start_day=15,
            duration_months=1,
            output_dir=temp_output_dir,
            sort_method="newest",
            mosaic_method="first",
            required_bands=["visual"],
        )

        expected_pattern = "50HMH_2022-12-15_to_2023-01-15_newest_first_visual.tif"

        assert result.name == expected_pattern


@pytest.mark.slow
class TestMosaicBoundsEndToEnd:
    """End-to-end coverage of bounds-mode parameter combinations.

    These hit the network (PC + COG reads + OCM). Each test uses a small AOI
    and short time range so they finish in a few seconds. Fixtures cache the
    common scene download via S2MOSAIC_DEBUG_CACHE=1 to keep wall time reasonable.
    """

    # Small AOI in 50HMH (Perth, WA) — single MGRS tile
    AOI_SMALL = (115.83, -31.97, 115.91, -31.94)
    # AOI straddling 50HMH/50HNH at ~117 deg E
    AOI_CROSS_TILE = (116.95, -32.05, 117.05, -31.95)
    # Tight date window with a couple of cloud-free passes
    DATE_KW = dict(start_year=2023, start_month=6, start_day=1, duration_days=14)
    QUERY = {"eo:cloud_cover": {"lt": 80}}

    @pytest.fixture(autouse=True)
    def enable_debug_cache(self, monkeypatch):
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

    @pytest.fixture(autouse=True)
    def time_test(self):
        start = time.time()
        yield
        elapsed = time.time() - start
        if elapsed > 60:
            print(f"\n  test took {elapsed:.1f}s")

    def _assert_basic_geotiff(self, arr, profile, expect_bands, expect_dtype=None):
        assert arr.ndim == 3, f"expected 3D, got shape {arr.shape}"
        assert arr.shape[0] == expect_bands
        if expect_dtype is not None:
            assert arr.dtype == expect_dtype
        assert "crs" in profile and profile["crs"] is not None
        assert "transform" in profile
        # At least some non-zero data — the AOI should be over land in winter
        assert arr.max() > 0

    # --- Resolution sweep ---
    @pytest.mark.parametrize("resolution", [10, 20, 30, 50, 100])
    def test_resolution_sweep(self, resolution):
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04", "B03", "B02"],
            mosaic_method="percentile",
            percentile_value=50,
            resolution=resolution,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3, expect_dtype=np.uint16)
        assert profile["transform"].a == resolution
        # Output dimensions should roughly match bounds × 1/resolution
        approx_w = (self.AOI_SMALL[2] - self.AOI_SMALL[0]) * 111000 / resolution
        assert abs(arr.shape[2] - approx_w) / approx_w < 0.2

    # --- Mosaic methods ---
    @pytest.mark.parametrize("method", ["mean", "first", "percentile", "median"])
    def test_mosaic_methods(self, method):
        kw = {}
        if method == "percentile":
            kw["percentile_value"] = 25
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method=method,
            additional_query=self.QUERY,
            **kw,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1, expect_dtype=np.uint16)

    # --- Sort methods ---
    @pytest.mark.parametrize("sort", ["valid_data", "oldest", "newest"])
    def test_sort_methods(self, sort):
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="first",
            sort_method=sort,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)

    # --- Bands at varied native resolutions ---
    @pytest.mark.parametrize(
        "bands",
        [
            ["B04"],  # 10m native
            ["B8A"],  # 20m native (native = OCM res)
            ["B01"],  # 60m native (must be upsampled)
            ["B04", "B03", "B02"],  # mixed but all 10m
            ["B04", "B8A", "B11"],  # 10m, 20m, 20m mix
            ["B02", "B03", "B04", "B08", "B11", "B12"],  # 6-band
        ],
    )
    def test_band_combinations(self, bands):
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=bands,
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=len(bands))

    # --- Resampling methods ---
    @pytest.mark.parametrize(
        "method", ["nearest", "bilinear", "cubic", "average", "lanczos"]
    )
    def test_resampling_methods(self, method):
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04", "B03", "B02"],
            mosaic_method="percentile",
            percentile_value=50,
            resolution=30,  # forces actual resampling
            resampling_method=method,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3)

    # --- Visual band in bounds mode ---
    def test_visual_band(self):
        # Visual asset is a 3-band uint8 TCI, fetched via the WarpedVRT path
        # (stackstac can't address sub-bands). Shape must match the cloud-mask
        # resize step — guards against the shape-mismatch bug where the OCM
        # mask probe and the TCI fetch snap bounds differently.
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["visual"],
            mosaic_method="first",
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3, expect_dtype=np.uint8)

    # --- Cross-MGRS-tile bounds ---
    def test_cross_tile_bounds(self):
        arr, profile = mosaic(
            bounds=self.AOI_CROSS_TILE,
            **self.DATE_KW,
            required_bands=["B04", "B03", "B02"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3)

    # --- Coverage threshold extremes ---
    @pytest.mark.parametrize("threshold", [None, 0.0, 0.1, 0.5, 0.9])
    def test_coverage_threshold(self, threshold):
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            coverage_threshold_pct=threshold,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)

    # --- no_data_threshold (early termination) ---
    @pytest.mark.parametrize("threshold", [None, 0.001, 0.5])
    def test_no_data_threshold(self, threshold):
        # Longer duration so we exercise multi-scene early termination
        date_kw = {**self.DATE_KW, "duration_days": 30}
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **date_kw,
            required_bands=["B04"],
            mosaic_method="first",  # early termination is meaningful
            no_data_threshold=threshold,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)

    # --- Custom target_crs ---
    def test_custom_target_crs(self):
        # Force WGS84 / Web Mercator output for a UTM-zone area
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            target_crs=3857,
            resolution=10,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)
        assert profile["crs"].to_epsg() == 3857

    # --- bounds_crs override (UTM bounds instead of lon/lat) ---
    def test_utm_bounds_input(self):
        # Convert AOI_SMALL to UTM 50S manually: rough bbox in EPSG:32750
        utm_bounds = (390000.0, 6463500.0, 397500.0, 6466500.0)
        arr, profile = mosaic(
            bounds=utm_bounds,
            bounds_crs=32750,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)
        assert profile["crs"].to_epsg() == 32750

    # --- Save to disk + return path ---
    def test_save_to_disk(self, tmp_path):
        result = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            output_dir=tmp_path,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        assert isinstance(result, Path) and result.exists()
        assert result.suffix == ".tif"
        assert "bounds_" in result.name  # filename uses bounds prefix

    # --- overwrite=False short-circuit ---
    def test_overwrite_false_short_circuits(self, tmp_path):
        kw = dict(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            output_dir=tmp_path,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        path1 = mosaic(**kw)
        first_mtime = path1.stat().st_mtime
        path2 = mosaic(**kw, overwrite=False)
        assert path1 == path2
        assert path2.stat().st_mtime == first_mtime

    # --- No scenes found ---
    def test_no_scenes_found(self):
        date_kw = {**self.DATE_KW, "duration_days": 1}
        with pytest.raises(Exception, match="No scenes found"):
            mosaic(
                bounds=self.AOI_SMALL,
                **date_kw,
                required_bands=["B04"],
                additional_query={"eo:cloud_cover": {"lt": 0.001}},
            )

    # --- Cloud-mask providers (OCM vs SCL) ---
    @pytest.mark.parametrize("cloud_mask", ["OCM", "SCL"])
    def test_cloud_mask_providers(self, cloud_mask):
        """Both providers should run end-to-end and produce data."""
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04", "B03", "B02"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
            cloud_mask=cloud_mask,
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3, expect_dtype=np.uint16)

    def test_scl_with_first_method(self):
        """SCL provider should work with first-mode early termination."""
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="first",
            additional_query=self.QUERY,
            cloud_mask="SCL",
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)

    def test_scl_visual_band(self):
        """SCL provider with the multi-band TCI fetch path."""
        arr, profile = mosaic(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["visual"],
            mosaic_method="first",
            additional_query=self.QUERY,
            cloud_mask="SCL",
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=3, expect_dtype=np.uint8)

    def test_scl_cross_tile(self):
        """SCL provider on an AOI that crosses MGRS tile boundaries."""
        arr, profile = mosaic(
            bounds=self.AOI_CROSS_TILE,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
            cloud_mask="SCL",
        )
        self._assert_basic_geotiff(arr, profile, expect_bands=1)

    def test_scl_cache_writes_scl_pkl(self, tmp_path, monkeypatch):
        """SCL provider should write scl_*.pkl cache files, not ocm_*.pkl."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")
        kw = dict(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
            cloud_mask="SCL",
        )
        mosaic(**kw)
        scl_caches = list((tmp_path / "cache").glob("scl_*.pkl"))
        ocm_caches = list((tmp_path / "cache").glob("ocm_*.pkl"))
        assert len(scl_caches) >= 1, "expected at least one scl cache file"
        assert len(ocm_caches) == 0, "SCL provider should not write ocm cache files"

    # --- disk cache hit on repeat ---
    def test_debug_cache_repeats_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # cache writes to ./cache (env var set by autouse)
        kw = dict(
            bounds=self.AOI_SMALL,
            **self.DATE_KW,
            required_bands=["B04"],
            mosaic_method="percentile",
            percentile_value=50,
            additional_query=self.QUERY,
        )
        arr1, _ = mosaic(**kw)
        arr2, _ = mosaic(**kw)
        # Warm-cache run must be byte-for-byte identical to the cold run
        np.testing.assert_array_equal(arr1, arr2)
        # Cache files should have been written
        cache_files = list((tmp_path / "cache").glob("stack_*.pkl"))
        assert len(cache_files) >= 1, "expected at least one stack cache file"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
