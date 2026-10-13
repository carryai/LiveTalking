"""avatars/restore.py — optional post-decoder restoration of the synthesised crop.

Why this exists
---------------
Wav2Lip's decoder is far softer than the source footage it is pasted into. The
seam tickets 02/03 fixed was never the whole problem: the generated mouth itself
is mush. A restoration network recovers plausible skin and tooth texture.

What it is NOT
--------------
Real-time, at least with the models available today, and this file would rather
say so than imply otherwise. Measured on this machine, GFPGAN v1.4:

    input size   512x512 ONLY — the network is built with out_size=512, so a
                 256x256 tensor fails in final_linear (1x1024 vs 4096x8192).
                 There is no cheap 256² mode to ask for.
    torch CPU    1260 ms/frame
    torch MPS     167 ms/frame  (batch 2: 176 ms — batching buys nothing here)
    CoreML/ANE   not convertible yet: the StyleGAN2 body carries data-dependent
                 shape ops (`out.view(b, c, *out.shape[2:4])` in basicsr's
                 ModulatedConv2d, `int(out.size(1)//2)` in GFPGANv1Clean) that
                 the MIL frontend cannot fold.
                 voice-gf/tools/convert_restore_coreml.py reaches the same wall
                 and reports it rather than pretending.

The pipeline has 40 ms to produce a frame at 25 fps, and the ANE already spends
~9 of those on Wav2Lip. So restoration cannot run on every frame: it runs
*amortised* — one frame in N is restored and its residual is reused until the
next one — which is what FrameBudgetGovernor decides and enforces. A slightly
soft mouth is invisible; a dropped frame is not.

Design rules
------------
1. `GF_RESTORE=off` costs nothing at all: no model loaded, no branch taken in
   the per-frame path. The avatar reads `RESTORE_ENABLED` once at import.
2. Nothing is imported eagerly. Torch, coremltools and the model package load
   inside the backends, so importing this module is free and
   `python3 avatars/restore.py --selftest` runs on a bare stdlib interpreter.
3. Every number published to /metrics comes from a measurement. A stage that
   did not run reports None, never 0.
"""
from __future__ import annotations

import json
import os
import time

# ── configuration ───────────────────────────────────────────────────────────
# off         nothing is loaded, nothing runs (the default, until it measures clean)
# gfpgan      GFPGAN v1.4, 512² — see the measured cost above
# codeformer  named so the config surface is stable; not implemented yet
RESTORE = os.environ.get("GF_RESTORE", "off").strip().lower()
RESTORE_BACKEND = os.environ.get("GF_RESTORE_BACKEND", "auto").strip().lower()
RESTORE_SIZE = int(os.environ.get("GF_RESTORE_SIZE", "512"))
RESTORE_MODEL = os.environ.get(
    "GF_RESTORE_MODEL", os.path.expanduser("~/models/gfpgan/GFPGANv1.4.pth"))
RESTORE_COREML_DIR = os.environ.get(
    "GF_RESTORE_COREML", os.path.expanduser("~/models/restore_coreml"))
RESTORE_BUDGET_MS = float(os.environ.get("GF_RESTORE_BUDGET_MS", "40"))
RESTORE_STATS_PATH = os.environ.get("GF_RESTORE_STATS", "/tmp/wav2lip_restore_stats.json")

RESTORE_ENABLED = RESTORE in ("gfpgan", "codeformer")

# Restore every Nth frame and reuse its residual N-1 times. The governor walks
# this ladder towards "less work" whenever the frame cost overruns.
MODE_LADDER = (1, 2, 4, 8, 16)


class RestoreError(RuntimeError):
    """Raised instead of quietly doing nothing."""


# ── the governor ────────────────────────────────────────────────────────────
class FrameBudgetGovernor:
    """Hold the frame budget by doing less restoration, never by waiting.

    Cost is smoothed with an EMA so that one slow frame (a page fault, a scene
    build running alongside) does not reconfigure the pipeline mid-utterance.
    Going cheaper is immediate; coming back needs `recovery_frames` consecutive
    frames comfortably under budget and waits out a cooldown, so a pipeline
    that sits on the line settles instead of flapping.

    `update()` takes what a frame actually cost and returns the new mode only
    when it changed. It never blocks: every decision moves towards less work,
    which is what keeps frames from being dropped.
    """

    def __init__(self, budget_ms=RESTORE_BUDGET_MS, ladder=MODE_LADDER,
                 alpha=0.2, down_ratio=1.0, up_ratio=0.7, cooldown_frames=25,
                 recovery_frames=25):
        self.budget_ms = float(budget_ms)
        self.ladder = tuple(ladder)
        self.alpha = float(alpha)
        self.down_ratio = float(down_ratio)   # at or above budget -> degrade
        self.up_ratio = float(up_ratio)       # below this -> recover
        self.cooldown_frames = int(cooldown_frames)
        self.recovery_frames = int(recovery_frames)
        self._index = 0
        self._ema = 0.0
        self._since_change = 0
        self._under = 0
        self.frames = 0
        self.transitions: list[dict] = []

    @property
    def every(self) -> int | None:
        """Every how-manyth frame gets restored; None means passthrough."""
        try:
            return self.ladder[self._index]
        except IndexError:
            return None

    @property
    def mode(self) -> str:
        every = self.every
        return f"every-{every}" if every else "passthrough"

    @property
    def ema_ms(self) -> float:
        return self._ema

    @property
    def restore_enabled(self) -> bool:
        return self.every is not None

    def should_restore(self, frame_index: int) -> bool:
        every = self.every
        return False if not every else frame_index % every == 0

    def update(self, cost_ms: float | None) -> str | None:
        """Absorb one frame's measured cost; return the new mode if it changed."""
        self.frames += 1
        self._since_change += 1
        if cost_ms is None:
            return None
        cost = float(cost_ms)
        self._ema = cost if self._ema == 0.0 else \
            self.alpha * cost + (1 - self.alpha) * self._ema

        # Degrade only when the past *and* the present agree, otherwise the EMA's
        # own inertia cascades: one spike would keep reading over budget for the
        # next several frames and walk the ladder all the way to passthrough on
        # stale data. Recovery below still uses the EMA alone, so a pipeline can
        # only ever become MORE careful quickly and comfortable slowly.
        if cost >= self.budget_ms * self.down_ratio and \
                self._ema >= self.budget_ms * self.down_ratio:
            if self._index < len(self.ladder):
                return self._move(+1, f"ema {self._ema:.1f} ms and last frame "
                                      f"{cost:.1f} ms >= "
                                      f"{self.budget_ms * self.down_ratio:.0f} ms budget")
            return None
        self._under = self._under + 1 if self._ema < self.budget_ms * self.up_ratio else 0
        if (self._under >= self.recovery_frames and self._index > 0
                and self._since_change >= self.cooldown_frames):
            return self._move(-1, f"ema {self._ema:.1f} ms under "
                                  f"{self.budget_ms * self.up_ratio:.0f} ms "
                                  f"for {self._under} frames")
        return None

    def degrade(self, reason: str) -> str:
        """Public way down the ladder — used when the backend itself fails."""
        if self._index < len(self.ladder):
            return self._move(+1, reason)
        return self.mode

    def _move(self, direction: int, reason: str) -> str:
        old = self.mode
        self._index = max(0, min(len(self.ladder), self._index + direction))
        self._since_change = 0
        self._under = 0
        self.transitions.append({"at": round(time.time(), 2), "frame": self.frames,
                                 "from": old, "to": self.mode, "reason": reason,
                                 "ema_ms": round(self._ema, 2)})
        return self.mode

    def snapshot(self) -> dict:
        return {"mode": self.mode, "every": self.every, "ema_ms": round(self._ema, 2),
                "budget_ms": self.budget_ms, "frames": self.frames,
                "transitions": len(self.transitions),
                "last_transition": self.transitions[-1] if self.transitions else None}


# ── residual reuse ──────────────────────────────────────────────────────────
class ResidualTracker:
    """Carry the sharpened detail forward across frames that skip the network.

    The interesting output of a restoration pass is not the frame, it is (restored
    - input): the texture the network invented. Adding that same residual to the
    following frames is far better than switching restoration off between them,
    and it is what makes an amortised restoration look continuous.
    """

    def __init__(self):
        self.residual = None        # HxWx3 float, the shape of the last crop
        self.shape = None
        self.reused = 0

    def remember(self, crop, restored):
        import numpy as np
        self.residual = np.asarray(restored, dtype=np.float32) - \
            np.asarray(crop, dtype=np.float32)
        self.shape = tuple(self.residual.shape)
        return self.residual

    def apply(self, crop):
        """Add the last known residual to `crop`, resized to fit if need be.

        Deliberately numpy-only: this runs inside the per-frame path, and a
        missing cv2 here would mean silently handing back unsharpened frames
        while /metrics still reported restoration as on.
        """
        import numpy as np
        crop = np.asarray(crop, dtype=np.float32)
        if self.residual is None:
            return crop
        r = self.residual
        if r.shape != crop.shape:
            r = _bilinear(r, crop.shape[:2])
        self.reused += 1
        return np.clip(crop + r, 0.0, 255.0).astype(np.float32)


def _bilinear(a, size):
    """Small bilinear resize (numpy only) so nothing here depends on cv2."""
    import numpy as np
    h, w = int(size[0]), int(size[1])
    ih, iw = a.shape[0], a.shape[1]
    if (ih, iw) == (h, w):
        return a
    ys = np.linspace(0, ih - 1, h)
    xs = np.linspace(0, iw - 1, w)
    y0, y1 = np.floor(ys).astype(int), np.minimum(np.floor(ys).astype(int) + 1, ih - 1)
    x0, x1 = np.floor(xs).astype(int), np.minimum(np.floor(xs).astype(int) + 1, iw - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


# ── backends ────────────────────────────────────────────────────────────────
class Backend:
    """Anything that turns a float NHWC batch (0..255) into a sharper one."""

    name = "none"
    size = RESTORE_SIZE

    def run_batch(self, batch):
        raise NotImplementedError


class TorchGFPGAN(Backend):
    """The reference backend: obviously correct, obviously too slow (167 ms).

    Kept because it is the only path that provably works today, because it is
    what `--benchmark` measures, and because a CoreML model can be checked
    against it — an fp16-compiled model that disagrees with this one is wrong.
    """

    name = "torch-gfpgan"

    def __init__(self, model_path=RESTORE_MODEL, size=RESTORE_SIZE, device=None):
        self.model_path = model_path
        self.size = int(size)
        self._device = device
        self._net = None

    def _torch(self):
        import torch
        return torch

    def load(self):
        if self._net is not None:
            return self._net
        torch = self._torch()
        try:
            # basicsr still reaches for a torchvision API that moved; alias it so
            # importing the architectures does not drag in the training loaders.
            import sys
            import types
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                try:
                    from torchvision.transforms import functional as _F
                    _stub = types.ModuleType("torchvision.transforms.functional_tensor")
                    for _n in dir(_F):
                        setattr(_stub, _n, getattr(_F, _n))
                    sys.modules["torchvision.transforms.functional_tensor"] = _stub
                except Exception:
                    pass
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        except Exception as e:
            raise RestoreError(
                f"GFPGAN architecture unavailable ({e}). Install gfpgan+basicsr "
                "in the LiveTalking venv, or use a CoreML model instead.")
        if not os.path.exists(self.model_path):
            raise RestoreError(f"no restoration weights at {self.model_path}")
        if self._device is None:
            mps = getattr(torch.backends, "mps", None)
            self._device = "mps" if (mps and mps.is_available()) else "cpu"
        net = GFPGANv1Clean(out_size=self.size, num_style_feat=512,
                            channel_multiplier=2, decoder_load_path=None,
                            fix_decoder=False, num_mlp=8, input_is_latent=True,
                            different_w=True, narrow=1, sft_half=True)
        sd = torch.load(self.model_path, map_location="cpu")
        net.load_state_dict(sd.get("params_ema", sd), strict=True)
        self._net = net.eval().to(self._device)
        return self._net

    def run_batch(self, batch):
        """`batch`: float NHWC array in 0..255, any HxW. Returns the same shape."""
        import cv2
        import numpy as np
        torch = self._torch()
        net = self.load()
        src_shape = batch.shape[1:3]
        if src_shape != (self.size, self.size):
            # The network only eats size x size. Upscaling costs ~0.2 ms.
            batch = np.stack([cv2.resize(f, (self.size, self.size),
                                         interpolation=cv2.INTER_LINEAR)
                              for f in batch])
        x = torch.from_numpy(np.ascontiguousarray(
            batch.transpose(0, 3, 1, 2), dtype=np.float32)).to(self._device)
        # GFPGAN was trained on [-1,1]
        with torch.no_grad():
            out = net((x / 127.5) - 1.0, randomize_noise=False)[0]
        out = ((out.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0)
        out = out.detach().cpu().numpy().transpose(0, 2, 3, 1)
        if src_shape != (self.size, self.size):
            out = np.stack([cv2.resize(f, (src_shape[1], src_shape[0]),
                                       interpolation=cv2.INTER_AREA)
                            for f in out])
        return out.astype(np.float32)


class CoreMLGFPGAN(Backend):
    """The shape restoration would need to be affordable: one compiled model per
    batch size, loaded lazily exactly like CoreMLWav2Lip.

    Nothing is convertible today (see the module docstring). Rather than failing
    obscurely at load time, `available()` answers whether a model exists and
    `build()` says why it does not.
    """

    name = "coreml-gfpgan"

    def __init__(self, directory=RESTORE_COREML_DIR, size=RESTORE_SIZE):
        self.directory = directory
        self.size = int(size)
        self._models = {}

    def available(self, batch: int) -> str | None:
        path = os.path.join(self.directory, f"restore_b{batch}_{self.size}.mlpackage")
        return path if os.path.exists(path) else None

    def run_batch(self, batch):
        try:
            n = int(batch.shape[0])
        except Exception:
            raise RestoreError("the CoreML backend wants an NHWC batch of crops")
        path = self.available(n)
        if path is None:
            raise RestoreError(
                f"no CoreML restoration model for batch {batch.shape[0]} in "
                f"{self.directory}. GF_RESTORE_BACKEND=torch uses the weights "
                "directly (167 ms/frame — not real-time).")
        import cv2
        import numpy as np
        import coremltools as ct
        model = self._models.get(path)
        if model is None:
            model = self._models[path] = ct.models.MLModel(
                path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        src_shape = batch.shape[1:3]
        if src_shape != (self.size, self.size):
            batch = np.stack([cv2.resize(f, (self.size, self.size),
                                         interpolation=cv2.INTER_LINEAR)
                              for f in batch])
        x = np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32)
        out = list(model.predict({"input": x}).values())[0]
        out = np.asarray(out).transpose(0, 2, 3, 1)
        if src_shape != (self.size, self.size):
            out = np.stack([cv2.resize(f, (src_shape[1], src_shape[0]),
                                       interpolation=cv2.INTER_AREA)
                            for f in out])
        return out.astype(np.float32)


def build_backend():
    """Choose a backend from the environment. Raises RestoreError rather than
    handing back something that quietly does nothing."""
    if not RESTORE_ENABLED:
        return None
    if RESTORE == "codeformer":
        raise RestoreError("codeformer is not implemented yet — use GF_RESTORE=gfpgan")
    coreml = CoreMLGFPGAN()
    want_coreml = RESTORE_BACKEND in ("auto", "coreml")
    if want_coreml and coreml.available(1):
        return coreml
    if RESTORE_BACKEND == "coreml":
        raise RestoreError(
            f"no CoreML restoration model in {coreml.directory}. The conversion "
            "is currently blocked (StyleGAN2's data-dependent shape ops); until "
            "it is solved use GF_RESTORE_BACKEND=torch and accept ~167 ms/frame.")
    return TorchGFPGAN()


# ── the restorer (what the avatar calls) ────────────────────────────────────
class Restorer:
    """One instance per avatar: owns the governor, the residual and the stats."""

    def __init__(self, backend=None, governor=None, budget_ms=RESTORE_BUDGET_MS,
                 stats_path=RESTORE_STATS_PATH, logger=None):
        self.backend = backend if backend is not None else build_backend()
        self.governor = governor or FrameBudgetGovernor(budget_ms=budget_ms)
        self.residual = ResidualTracker()
        self.stats_path = stats_path
        self.logger = logger
        self.frame = 0
        self.restorations = 0
        self.reuses = 0
        self.skipped = 0
        self.ms_per_frame = None        # None until something has been measured
        self._last_write = 0.0

    def enabled(self) -> bool:
        return self.backend is not None and self.governor.restore_enabled

    def process(self, crops):
        """Sharpen a batch of float HxWx3 crops (0..255).

        `crops` is whatever Wav2Lip produced for this batch; frames the governor
        did not select carry the previous residual instead. Returns a new array
        of the same shape. Never raises on a single-frame failure: an exception
        here would drop frames, which is the one thing that must not happen, so
        the batch passes through untouched and the mode is degraded.
        """
        import numpy as np
        out = [np.asarray(c, dtype=np.float32) for c in crops]
        if not self.enabled():
            return np.stack(out) if out else crops

        picked = [i for i in range(len(out)) if self.governor.should_restore(self.frame + i)]
        try:
            if picked:
                t0 = time.time()
                restored = self.backend.run_batch(np.stack([out[i] for i in picked]))
                elapsed_ms = (time.time() - t0) * 1000.0
                for k, i in enumerate(picked):
                    self.residual.remember(out[i], restored[k])
                    out[i] = restored[k]
                self.restorations += len(picked)
                self.ms_per_frame = round(elapsed_ms / max(len(picked), 1), 2)
                self._note(elapsed_ms / max(len(out), 1), restored=True)
            for i in range(len(out)):
                if i not in picked:
                    out[i] = self.residual.apply(out[i])
                    self.reuses += 1
            self.frame += len(out)
            return np.stack(out)
        except Exception as e:
            # whatever went wrong, do not stall the pipeline
            self.skipped += len(out)
            if self.logger:
                self.logger.warning(f"restore failed ({e}); passing through")
            self.governor.degrade(f"restore failed: {str(e)[:60]}")
            self.frame += len(out)
            return np.stack(out)

    def _note(self, ms, restored: bool):
        mode = self.governor.update(ms)
        if mode and self.logger:
            self.logger.info(f"restore budget: {mode}")
        now = time.time()
        if now - self._last_write > 1.0:
            self._last_write = now
            write_stats(self.stats(), self.stats_path)

    def stats(self) -> dict:
        return {"enabled": self.enabled(), "model": getattr(self.backend, "name", None),
                "size": getattr(self.backend, "size", RESTORE_SIZE),
                "ms_per_frame": self.ms_per_frame,
                "frames": self.frame, "restored": self.restorations,
                "residual_reused": self.reuses, "passed_through": self.skipped,
                "governor": self.governor.snapshot(),
                "updated": round(time.time(), 2)}


# ── telemetry ───────────────────────────────────────────────────────────────
def write_stats(payload: dict, path: str = RESTORE_STATS_PATH) -> None:
    """Atomic publish for /metrics — same pattern as the ANE stats file."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass                      # telemetry must never take the pipeline down


def read_stats(path: str = RESTORE_STATS_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── stdlib-only self check ──────────────────────────────────────────────────
def _selftest() -> int:
    """Governor invariants. Runs on a bare interpreter: numpy is not imported.

    `python3 avatars/restore.py --selftest`
    """
    checks = []

    g = FrameBudgetGovernor(budget_ms=40.0, recovery_frames=3, cooldown_frames=1)
    checks.append(("starts restoring every frame", g.every == 1))
    first = g.update(60.0)                     # one overrun
    checks.append(("degrades on the first overrun", first == "every-2"))
    for budget in (8.0,):
        for _ in range(40):
            g.update(budget * -1 + 100)        # keep it hot until passthrough
    checks.append(("reaches passthrough rather than blocking", g.every is None))
    changes = [t["to"] for t in g.transitions]
    index = [MODE_LADDER.index(int(c.split("-")[1])) if "-" in c else len(MODE_LADDER)
             for c in changes]
    checks.append(("never re-enables work while over budget",
                   all(b >= a for a, b in zip(index, index[1:]))))

    g2 = FrameBudgetGovernor(budget_ms=40.0, recovery_frames=3, cooldown_frames=1)
    g2.update(60.0)                            # -> every-2
    recovered = [g2.update(5.0) for _ in range(10)]
    checks.append(("recovers when headroom persists", g2.every == 1))
    checks.append(("announces the recovery exactly once",
                   [m for m in recovered if m].count("every-1") == 1))
    checks.append(("one spike does not cascade to passthrough",
                   all(t["to"] != "passthrough" for t in g2.transitions)))

    g3 = FrameBudgetGovernor(budget_ms=40.0, recovery_frames=50, cooldown_frames=100)
    changes_before = len(g3.transitions)
    for i in range(20):
        g3.update(80.0 if i % 2 else 1.0)      # flapping cost
    checks.append(("does not flap on alternating frames",
                   len(g3.transitions) - changes_before <= 11))

    every_frame = FrameBudgetGovernor(budget_ms=1e9)
    checks.append(("a frame with no cost samples changes nothing",
                   every_frame.update(None) is None and every_frame.every == 1))

    for label, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
