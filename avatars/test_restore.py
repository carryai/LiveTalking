#!/usr/bin/env python3
"""test_restore.py — the governor and the pass-through contract.

The interesting claim in ticket 04 is not "the mouth is sharper"; it is
**restoration must never cost a frame**. That is testable without a GPU, without
a server and without the model, so this suite runs anywhere:

    python3 livetalking/avatars/test_restore.py
    ./livetalking/venv/bin/python livetalking/avatars/test_restore.py

The numpy-dependent cases skip themselves rather than failing on a bare
interpreter — the governor is the safety mechanism and must always be checked.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avatars.restore import (  # noqa: E402
    FrameBudgetGovernor, MODE_LADDER, ResidualTracker, Restorer, RestoreError,
    TorchGFPGAN, CoreMLGFPGAN, build_backend, write_stats, read_stats)

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:                                     # pragma: no cover
    HAVE_NUMPY = False

try:
    import cv2                                        # noqa: F401
    HAVE_CV2 = True
except Exception:                                     # pragma: no cover
    HAVE_CV2 = False

try:
    import torch                                      # noqa: F401
    HAVE_TORCH = True
except Exception:                                     # pragma: no cover
    HAVE_TORCH = False


class FakeBackend:
    """A backend whose cost we choose, so the budget can be argued about."""

    name = "fake"

    def __init__(self, cost_ms=0.0, on_call=None, raise_on=None):
        self.cost_ms = cost_ms
        self.size = 512
        self.calls = 0
        self.on_call = on_call
        self.raise_on = raise_on

    def run_batch(self, batch):
        self.calls += 1
        if self.raise_on is not None and self.calls >= self.raise_on:
            raise RuntimeError("the network died")
        out = batch + 10.0
        if self.on_call:
            self.on_call(self.calls, batch)
        return out


class GovernorTest(unittest.TestCase):
    """The safety machinery: it must always move towards doing less work."""

    def budget(self, **kw):
        kw.setdefault("recovery_frames", 3)
        kw.setdefault("cooldown_frames", 1)
        return FrameBudgetGovernor(budget_ms=40.0, **kw)

    def test_starts_by_restoring_every_frame(self):
        self.assertEqual(self.budget().every, 1)

    def test_degrades_when_the_frame_is_over_budget(self):
        g = self.budget()
        self.assertEqual(g.update(60.0), "every-2")

    def test_a_single_spike_does_not_walk_the_whole_ladder(self):
        """The EMA's inertia must not cascade: one spike, one step."""
        g = self.budget()
        g.update(60.0)
        for _ in range(10):
            g.update(5.0)
        self.assertEqual(len([t for t in g.transitions if t["to"] != "every-1"]), 1)
        self.assertNotEqual(g.every, None, "a lone spike must not reach passthrough")

    def test_sustained_overrun_reaches_passthrough(self):
        g = self.budget()
        for _ in range(200):
            g.update(80.0)
        self.assertIsNone(g.every)
        self.assertEqual(g.mode, "passthrough")

    def test_recovers_only_after_sustained_headroom(self):
        g = self.budget()
        for _ in range(50):
            g.update(80.0)
        self.assertIsNone(g.every)
        # one cheap frame is not headroom — that is the hysteresis working
        self.assertIsNone(g.update(5.0))
        self.assertIsNone(g.every)
        for _ in range(200):
            g.update(5.0)
        self.assertEqual(g.every, 1)

    def test_never_stops_passing_frames_through(self):
        """Whatever we do, a frame is always produced — that is the contract."""
        g = self.budget()
        for cost in (1.0, 999.0, 40.0, 0.0, 5000.0):
            g.update(cost)
            self.assertIn(g.mode, [f"every-{n}" for n in MODE_LADDER] + ["passthrough"])

    def test_ladder_only_ever_goes_one_way_at_a_time(self):
        g = FrameBudgetGovernor(budget_ms=40.0, recovery_frames=2, cooldown_frames=0)
        seen = []
        for cost in ([80.0] * 20 + [5.0] * 200 + [80.0] * 20):
            g.update(cost)
            seen.append(MODE_LADDER.index(g.every) if g.every else len(MODE_LADDER))
        # no step is bigger than one rung in either direction
        self.assertTrue(all(abs(b - a) <= 1 for a, b in zip(seen, seen[1:])), seen)

    def test_unmeasured_frames_change_nothing(self):
        g = self.budget()
        g.update(80.0)
        before = (g.mode, len(g.transitions))
        for _ in range(10):
            self.assertIsNone(g.update(None))
        self.assertEqual((g.mode, len(g.transitions)), before)

    def test_snapshot_is_json_ready(self):
        import json
        g = self.budget()
        g.update(50.0)
        self.assertEqual(json.loads(json.dumps(g.snapshot()))["mode"], "every-2")

    def test_degrade_is_idempotent_at_the_bottom(self):
        g = FrameBudgetGovernor(budget_ms=40.0)
        g._index = len(MODE_LADDER)
        self.assertEqual(g.degrade("because"), "passthrough")


class ZeroCostTest(unittest.TestCase):
    """GF_RESTORE=off must not even be reachable."""

    def test_backend_factory_returns_nothing_when_disabled(self):
        import avatars.restore as restore
        keep = restore.RESTORE
        try:
            restore.RESTORE = "off"
            self.assertIsNone(build_backend())
            self.assertFalse(restore.RESTORE_ENABLED)
        finally:
            restore.RESTORE = keep

    @unittest.skipUnless(HAVE_TORCH, "torch needed to reach the weight check")
    def test_unknown_model_raises_instead_of_silently_do_nothing(self):
        import avatars.restore as restore
        keep = restore.RESTORE
        try:
            restore.RESTORE = "gfpgan"
            with self.assertRaises(RestoreError):
                TorchGFPGAN(model_path="/nonexistent/GFPGANv1.4.pth").load()
        finally:
            restore.RESTORE = keep

    def test_coreml_says_there_is_no_model_rather_than_guessing(self):
        coreml = CoreMLGFPGAN(directory="/nonexistent/coreml")
        self.assertIsNone(coreml.available(1))
        with self.assertRaises(RestoreError):
            coreml.run_batch(None)          # not even a batch on purpose


@unittest.skipUnless(HAVE_NUMPY, "numpy needed for the array path")
class RestorerTest(unittest.TestCase):
    def crops(self, n=4, size=8, value=100.0):
        return np.full((n, size, size, 3), value, dtype=np.float32)

    def restorer(self, backend, budget_ms=40.0):
        return Restorer(backend=backend, budget_ms=budget_ms,
                        stats_path="/tmp/test_restore_stats.json")

    def test_first_frame_without_a_residual_passes_through_unchanged(self):
        """We cannot invent sharpness, and we must not invent it."""
        backend = FakeBackend()
        r = self.restorer(backend)
        crops = self.crops(1)
        out = r.process(crops)
        self.assertEqual(backend.calls, 1)                       # it did restore
        self.assertGreater(float(out[0].max()), 100.0)           # and it changed

    def test_frames_the_governor_skipped_reuse_the_residual(self):
        backend = FakeBackend()
        r = self.restorer(backend)
        r.governor._index = 1                                    # every 2nd frame
        out = r.process(self.crops(4))
        self.assertEqual(backend.calls, 1, "one network call per batch")
        self.assertEqual(r.restorations, 2, "two of four frames restored")
        # the skipped frames carry the residual rather than nothing at all
        self.assertAlmostEqual(float(out[1][0][0][0]), 110.0, places=3)
        self.assertEqual(r.reuses, 2)

    def test_a_failing_backend_never_drops_the_frame(self):
        """The one thing that must not happen. Ever."""
        backend = FakeBackend(raise_on=1)
        r = self.restorer(backend)
        crops = self.crops(2)
        out = r.process(crops)
        self.assertTrue(np.allclose(out, crops), "frames must pass through untouched")
        self.assertEqual(r.skipped, 2)
        self.assertEqual(r.governor.mode, "every-2", "and the budget learns from it")

    def test_disabled_restorer_does_not_touch_a_pixel(self):
        r = self.restorer(None)
        crops = self.crops(2)
        self.assertTrue(np.allclose(r.process(crops), crops))
        self.assertEqual(r.ms_per_frame, None)

    def test_ms_per_frame_is_none_until_measured_not_zero(self):
        r = self.restorer(FakeBackend())
        self.assertIsNone(r.ms_per_frame)
        r.process(self.crops(1))
        self.assertIsInstance(r.ms_per_frame, float)

    def test_stats_publish_only_measured_numbers(self):
        r = self.restorer(FakeBackend())
        stats = r.stats()
        self.assertIsNone(stats["ms_per_frame"])
        r.process(self.crops(2))
        stats = r.stats()
        self.assertEqual(stats["restored"], 2)
        self.assertEqual(stats["governor"]["mode"], "every-1")

    def test_amortised_cost_is_what_the_budget_compares_against(self):
        """171 ms every 16th frame is 10.7 ms/frame — how this can ever fit."""
        class Slow(FakeBackend):
            pass
        backend = Slow()
        r = self.restorer(backend, budget_ms=float("inf"))
        r.ms_per_frame = 171.0
        per_frame = r.ms_per_frame / 16
        self.assertLess(per_frame, 12.0, "amortised over 16 frames it fits")


@unittest.skipUnless(HAVE_NUMPY and HAVE_CV2, "numpy+cv2 needed for the residual")
class ResidualTest(unittest.TestCase):
    def test_residual_carries_across_frames_and_shapes(self):
        t = ResidualTracker()
        crop = np.full((8, 8, 3), 100.0, dtype=np.float32)
        t.remember(crop, crop + 12.0)
        self.assertEqual(t.residual.shape, (8, 8, 3))
        applied = t.apply(crop)
        self.assertAlmostEqual(float(applied[0][0][0]), 112.0, places=3)
        bigger = np.full((16, 16, 3), 100.0, dtype=np.float32)
        resized = t.apply(bigger)
        self.assertEqual(resized.shape, (16, 16, 3))
        self.assertAlmostEqual(float(resized[0][0][0]), 112.0, places=3)

    def test_no_residual_yet_means_no_change(self):
        t = ResidualTracker()
        crop = np.full((4, 4, 3), 50.0, dtype=np.float32)
        self.assertTrue(np.allclose(t.apply(crop), crop))


class StatsTest(unittest.TestCase):
    def test_write_and_read_round_trip(self):
        path = "/tmp/test_restore_stats_roundtrip.json"
        write_stats({"enabled": True, "ms_per_frame": None}, path)
        self.assertEqual(read_stats(path)["enabled"], True)
        self.assertIsNone(read_stats(path)["ms_per_frame"])

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(read_stats("/tmp/definitely-not-here.json"), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
