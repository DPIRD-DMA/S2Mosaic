import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic.masking import get_valid_mask
from s2mosaic.mosaic_utils import calculate_percentile_mosaic


class TestGetValidMask:
    """Unit tests for the no-data dilation step (cv2.dilate, MORPH_CROSS, it=4)."""

    def test_all_zero_input_returns_all_invalid(self):
        bands = np.zeros((3, 50, 50), dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert not mask.any()

    def test_all_data_input_returns_all_valid(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.all()

    def test_pixel_is_invalid_only_when_all_bands_are_zero(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        bands[0, 5, 5] = 100  # one band has data → pixel is valid
        mask = get_valid_mask(bands, dilation_count=0)
        assert mask[5, 5]
        assert not mask[0, 0]

    def test_dilation_count_zero_does_not_grow_mask(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=0)
        assert not mask[25, 25]
        # Direct neighbors still valid
        assert mask[24, 25]
        assert mask[26, 25]
        assert mask[25, 24]
        assert mask[25, 26]

    def test_dilation_grows_invalid_region_diamond(self):
        # Repeated MORPH_CROSS dilations form a diamond (Manhattan distance) region
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=4)
        # Manhattan distance <= 4 from (25, 25) is invalid
        assert not mask[25, 25]  # dist 0
        assert not mask[21, 25]  # dist 4
        assert not mask[29, 25]
        assert not mask[25, 21]
        assert not mask[25, 29]
        assert not mask[23, 27]  # dist 4 (2+2)
        # Manhattan distance 5+ remains valid
        assert mask[20, 25]
        assert mask[30, 25]
        assert mask[22, 28]  # dist 5

    def test_returns_bool_dtype(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        assert get_valid_mask(bands).dtype == bool


class TestZoomAlignmentConvention:
    """Regression test for the nearest-neighbor resample alignment.

    s2mosaic deliberately uses cv2.INTER_NEAREST (pixel-area-cell convention)
    instead of scipy.ndimage.zoom order=0 (centered-point convention) so an
    upsampled 60m S2 pixel maps exactly to the 6x6 block of 10m pixels it
    covers in the GeoTIFF transform. Reverting to scipy would shift 60m
    bands by ~30m (3 pixels at 10m).
    """

    def test_2x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        out = cv2.resize(src, (6, 4), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(2, axis=0).repeat(2, axis=1))

    def test_6x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        out = cv2.resize(src, (12, 12), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(6, axis=0).repeat(6, axis=1))

    def test_3x_upsample_uses_pixel_area_layout(self):
        # Pixel-area: input pixel i fills output cells [i*N, (i+1)*N).
        # Scipy's order=0 would yield [1, 1, 2, 2, 2, 2, 3, 3, 3] — different.
        src = np.array([[1, 2, 3]], dtype=np.uint16)
        out = cv2.resize(src, (9, 1), interpolation=cv2.INTER_NEAREST)
        expected = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uint16)
        np.testing.assert_array_equal(out, expected)


class TestCalculatePercentileMosaic:
    """Tests for the percentile mosaic codepath (numbagg.nanquantile)."""

    SCENE_SIZE = 50

    def _scene(self, value, bands=1):
        if value is None:
            return np.full(
                (bands, self.SCENE_SIZE, self.SCENE_SIZE), np.nan, dtype=np.float32
            )
        return np.full(
            (bands, self.SCENE_SIZE, self.SCENE_SIZE), value, dtype=np.float32
        )

    def test_single_scene_returns_that_scene(self):
        scene = self._scene(5.0)
        out = calculate_percentile_mosaic([scene], s2_scene_size=self.SCENE_SIZE)
        np.testing.assert_allclose(out, 5.0)
        assert out.shape == scene.shape

    def test_median_of_three_constant_scenes(self):
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=50.0
        )
        np.testing.assert_allclose(out, 5.0)

    def test_p90_of_three_constant_scenes(self):
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=90.0
        )
        # 90th percentile of [1, 5, 9] is 8.2 (linear interp between 5 and 9)
        np.testing.assert_allclose(out, 8.2, rtol=1e-5)

    def test_all_nan_input_replaced_with_zero(self):
        scene = self._scene(None)
        out = calculate_percentile_mosaic([scene], s2_scene_size=self.SCENE_SIZE)
        np.testing.assert_array_equal(out, 0.0)

    def test_nans_skipped_in_percentile(self):
        # Median of [NaN, 5, 15] should be 10 (NaN ignored)
        scenes = [self._scene(None), self._scene(5.0), self._scene(15.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=50.0
        )
        np.testing.assert_allclose(out, 10.0)

    def test_chunk_concatenation_matches_unchunked(self):
        # Output should be the same regardless of chunk_size
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out_chunked = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, chunk_size=10
        )
        out_one_chunk = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, chunk_size=self.SCENE_SIZE
        )
        np.testing.assert_array_equal(out_chunked, out_one_chunk)
        assert out_chunked.shape == (1, self.SCENE_SIZE, self.SCENE_SIZE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
