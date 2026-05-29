import cv2
import numpy as np

from s2mosaic.masking import (
    SCL_CLOUDY_CLASSES,
    compute_masks_from_scl,
    get_masks,
    get_valid_mask,
)


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


class TestComputeMasksFromScl:
    """SCL-based clear+valid mask logic."""

    def test_known_class_layout(self):
        # One pixel of every class 0..11 in a single row.
        scl = np.arange(12, dtype=np.uint8).reshape(1, 12)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        # Unsafe classes are not clear; vegetation, bare soil, water, and snow
        # remain clear. SCL no-data is invalid separately.
        for cls in range(12):
            expected_clear = cls not in SCL_CLOUDY_CLASSES
            assert bool(clear[0, cls]) == expected_clear, (
                f"class {cls} clear={clear[0, cls]}"
            )
        # No-data (class 0) is the only invalid pixel without dilation
        assert not valid[0, 0]
        for cls in range(1, 12):
            assert valid[0, cls]

    def test_cloudy_classes_match_constant(self):
        # Defensive: lock down the constant so changes are explicit
        assert SCL_CLOUDY_CLASSES == (1, 2, 3, 7, 8, 9, 10)

    def test_dilation_grows_invalid_around_no_data(self):
        # Single no-data pixel → 4-iter MORPH_CROSS dilate → diamond of invalids
        scl = np.full((50, 50), 5, dtype=np.uint8)  # all bare-soil
        scl[25, 25] = 0
        _, valid = compute_masks_from_scl(scl, dilation_count=4)
        assert not valid[25, 25]
        assert not valid[21, 25]  # Manhattan dist 4
        assert not valid[25, 21]
        assert valid[20, 25]  # dist 5 still valid
        assert valid[25, 20]

    def test_3d_input_squeezes_first_axis(self):
        # get_full_band returns (1, H, W); compute_masks_from_scl should handle that
        scl = np.array([[[0, 1, 4, 8]]], dtype=np.uint8)  # shape (1, 1, 4)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        assert clear.shape == (1, 4)
        np.testing.assert_array_equal(clear[0], [True, False, True, False])
        np.testing.assert_array_equal(valid[0], [False, True, True, True])

    def test_returns_bool_dtype(self):
        scl = np.zeros((10, 10), dtype=np.uint8)
        clear, valid = compute_masks_from_scl(scl)
        assert clear.dtype == bool
        assert valid.dtype == bool


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


class TestMaskingHelpers:
    def test_get_masks_resizes_to_rectangular_target(self, monkeypatch):
        class FakeAsset:
            href = "remote.tif"

        class FakeItem:
            assets = {"B04": FakeAsset(), "B03": FakeAsset(), "B8A": FakeAsset()}

        class FakeSource:
            def asset_name(self, canonical):
                return canonical

        seen_assets = set()
        band_values = {"B04": 4, "B03": 3, "B8A": 8}

        def fake_get_full_band(href, *, source, res, asset_name):
            seen_assets.add(asset_name)
            return np.full((1, 2, 2), band_values[asset_name], dtype=np.uint16), {}

        def fake_compute_masks_from_array(array, batch_size, inference_dtype):
            np.testing.assert_array_equal(array[:, 0, 0], np.array([4, 3, 8]))
            return (
                np.array([[True, False], [False, True]]),
                np.ones((2, 2), dtype=bool),
            )

        monkeypatch.setattr("s2mosaic.masking.get_full_band", fake_get_full_band)
        monkeypatch.setattr(
            "s2mosaic.masking.compute_masks_from_array",
            fake_compute_masks_from_array,
        )

        clear, valid = get_masks(FakeItem(), FakeSource(), target_size=(3, 5))

        assert clear.shape == (3, 5)
        assert valid.shape == (3, 5)
        assert seen_assets == {"B04", "B03", "B8A"}
