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
                "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", "B11", "B12",
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
                debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
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
            debug_cache=True,
            required_bands=["B04"],
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)

    @pytest.mark.parametrize(
        "start_month, cloud_cover_lt", [(1, 50), (6, 20), (6, 10)]
    )
    def test_mosaic_different_cloud_cover_thresholds(self, start_month, cloud_cover_lt):
        """Test mosaic with different cloud cover thresholds"""
        result = mosaic(
            "50HMH",
            2023,
            start_month=start_month,
            start_day=1,
            duration_months=1,
            debug_cache=True,
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
                debug_cache=True,
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
            debug_cache=True,
            required_bands=["B04"],
        )

        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)


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
            debug_cache=True,
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
            debug_cache=True,
            sort_method="newest",
            mosaic_method="first",
            required_bands=["visual"],
        )

        expected_pattern = "50HMH_2022-12-15_to_2023-01-15_newest_first_visual.tif"

        assert result.name == expected_pattern


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
