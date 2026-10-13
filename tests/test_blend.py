"""Tests for utils/blend.py — feathered mask + colour transfer + composite.

Run with:  python -m pytest livetalking/tests/test_blend.py -v
"""

import os
import time
import numpy as np
import cv2
import pytest

# Ensure the blend module picks up test-friendly defaults before import.
os.environ.setdefault("GF_BLEND", "feather")

from utils.blend import (
    feather_mask,
    reinhard_colour_transfer,
    composite,
    clear_cache,
    _mask_cache,
    BLEND_MODE,
)


class TestFeatherMask:
    def setup_method(self):
        clear_cache()

    def test_shape_and_range(self):
        m = feather_mask(100, 80)
        assert m.shape == (100, 80)
        assert m.dtype == np.float32
        assert m.min() >= 0.0
        assert m.max() <= 1.0

    def test_center_is_bright_edges_are_dark(self):
        m = feather_mask(200, 200)
        center = m[100, 100]
        corner = m[0, 0]
        assert center > 0.8
        assert corner < 0.05

    def test_cache_hit(self):
        """Same (h, w) returns the exact same array object — no per-frame alloc."""
        m1 = feather_mask(120, 90)
        m2 = feather_mask(120, 90)
        assert m1 is m2
        assert len(_mask_cache) == 1

    def test_cache_different_sizes(self):
        feather_mask(100, 80)
        feather_mask(120, 90)
        assert len(_mask_cache) == 2

    def test_custom_sigma(self):
        m = feather_mask(100, 100, sigma=2.0)
        assert m.shape == (100, 100)
        assert m.dtype == np.float32


class TestReinhardColourTransfer:
    def test_output_shape_and_dtype(self):
        src = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        gen = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = feather_mask(64, 64)
        result = reinhard_colour_transfer(src, gen, mask)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_mean_shifts_toward_source(self):
        """After transfer, the gen crop's mean should be closer to the source's."""
        rng = np.random.RandomState(42)
        src = rng.randint(180, 220, (64, 64, 3), dtype=np.uint8)  # bright
        gen = rng.randint(40, 80, (64, 64, 3), dtype=np.uint8)    # dark
        mask = feather_mask(64, 64)

        result = reinhard_colour_transfer(src, gen, mask)

        m_bool = mask > 0.1
        src_mean = src[:, :, 0][m_bool].mean()
        gen_mean_before = gen[:, :, 0][m_bool].mean()
        result_mean_after = result[:, :, 0][m_bool].mean()

        # The result should be closer to src_mean than the original gen was
        dist_before = abs(gen_mean_before - src_mean)
        dist_after = abs(result_mean_after - src_mean)
        assert dist_after < dist_before

    def test_low_mask_coverage_returns_original(self):
        """When the mask has very few valid pixels, return gen unchanged."""
        src = np.full((64, 64, 3), 200, dtype=np.uint8)
        gen = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)  # all zero
        result = reinhard_colour_transfer(src, gen, mask)
        np.testing.assert_array_equal(result, gen)


class TestComposite:
    def setup_method(self):
        clear_cache()

    def test_output_shape_matches_source(self):
        src = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gen = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = composite(src, gen, 100, 150, 180, 250)
        assert result.shape == src.shape
        assert result.dtype == np.uint8

    def test_does_not_modify_source(self):
        src = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        src_copy = src.copy()
        gen = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        composite(src, gen, 100, 150, 180, 250)
        np.testing.assert_array_equal(src, src_copy)

    def test_feather_reduces_edge_gradient_vs_hard(self):
        """The scripted acceptance test: mean edge-gradient energy along the bbox
        border drops >= 60% vs hard paste."""
        rng = np.random.RandomState(123)
        # A realistic source: smooth skin-tone gradient
        src = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(480):
            src[y, :] = [180 + y // 10, 140 + y // 15, 120 + y // 20]

        # A gen crop that's visibly different in brightness
        gen = rng.randint(60, 100, (100, 120, 3), dtype=np.uint8)

        x1, y1, x2, y2 = 200, 150, 320, 250

        # Hard paste (ground truth of the bug)
        hard = src.copy()
        gen_resized = cv2.resize(gen, (x2 - x1, y2 - y1))
        hard[y1:y2, x1:x2] = gen_resized

        # Feather composite
        feather = composite(src, gen, x1, y1, x2, y2)

        def edge_gradient_energy(frame, x1, y1, x2, y2, band=3):
            """Mean gradient magnitude along the bbox border (inside + outside)."""
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gy, gx = np.gradient(gray)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            border_vals = []
            for offset in range(-band, band + 1):
                # top and bottom edges
                for edge_y in [y1 + offset, y2 - 1 + offset]:
                    if 0 <= edge_y < gray.shape[0]:
                        border_vals.append(mag[edge_y, x1:x2])
                # left and right edges
                for edge_x in [x1 + offset, x2 - 1 + offset]:
                    if 0 <= edge_x < gray.shape[1]:
                        border_vals.append(mag[y1:y2, edge_x])
            return np.concatenate(border_vals).mean()

        e_hard = edge_gradient_energy(hard, x1, y1, x2, y2)
        e_feather = edge_gradient_energy(feather, x1, y1, x2, y2)

        reduction = 1.0 - e_feather / e_hard
        assert reduction >= 0.60, (
            f"edge gradient only dropped {reduction:.1%}, expected >= 60% "
            f"(hard={e_hard:.1f}, feather={e_feather:.1f})"
        )

    def test_performance_under_2ms(self):
        """Added cost <= 2 ms/frame at 576x768, averaged over 500 frames."""
        rng = np.random.RandomState(99)
        src = rng.randint(0, 255, (768, 576, 3), dtype=np.uint8)
        gen = rng.randint(0, 255, (120, 100, 3), dtype=np.uint8)

        # Warm up the cache
        composite(src, gen, 100, 200, 200, 320)

        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            composite(src, gen, 100, 200, 200, 320)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n

        assert elapsed_ms < 2.0, f"composite took {elapsed_ms:.2f} ms/frame, limit is 2 ms"

    def test_zero_size_region_returns_copy(self):
        src = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gen = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = composite(src, gen, 50, 50, 50, 50)  # zero-area
        np.testing.assert_array_equal(result, src)

    def test_preallocated_buffer(self):
        """Passing _buf avoids allocating a new array for the blend."""
        src = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gen = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        buf = np.empty((100, 80, 3), dtype=np.float32)
        result = composite(src, gen, 100, 150, 180, 250, _buf=buf)
        assert result.shape == src.shape


class TestBlendModeEnv:
    def test_blend_mode_default(self):
        assert BLEND_MODE in ("feather", "hard")
