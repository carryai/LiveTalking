"""MuseTalk 1.5 (MLX) as a LiveTalking avatar backend, over a worker process.

Why: MuseTalk 1.5 inpaints in latent space, so teeth/tongue/mouth interior are real
pixels instead of wav2lip's blurred patch. The weights are the mlx-community MLX port
(fp16 measured FASTER than q8 on an M1 Pro: 3.62 vs 3.51 faces/s, and more faithful).

This module contains no MLX, no torch model, and no Metal. It is an HTTP client for
tools/musetalk_mlx_worker.py, because running MLX inside LiveTalking deadlocks the
avatar server outright (the process stays alive at 0% CPU and :8010 stops answering;
the main thread parks in libmlx's ThreadPool::start_threads). Eager pool init and a
global lock did not fix it — the process boundary did.

Registration: `musetalk-mlx`. Enable with

    GF_MUSETALK_MLX_URL=http://127.0.0.1:8040 \
    GF_AVATAR_MODEL=musetalk-mlx GF_AVATAR=<id> ./start_all.sh

WHAT THE NUMBERS ARE (M1 Pro, 256² crops, measured)
    UNet          ~107 ms/frame      VAE decode ~159 ms/frame   => ~3.6 faces/s
    full image    ~0 ms — MuseTalk only ever sees the face crop; paste-back is CPU
    crop size     fixed at 256²; 192/128 raise a broadcast error inside the pipeline
Because 3.6 faces/s is far short of 25 fps, REGEN_HZ caps regeneration and the frames
in between reuse the last mouth. At 8 Hz that measures ~11-12 fps of video.
"""
import glob
import io
import json
import os
import pickle
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from avatars.base_avatar import BaseAvatar
from avatars.audio_features.whisper import WhisperASR
from avatars.musetalk.whisper.audio2feature import Audio2Feature
from registry import register
from utils.blend import composite as _feather_composite, BLEND_MODE, BLEND_SIGMA
from utils.image import mirror_index, read_imgs
from utils.logger import logger

WORKER_URL = os.environ.get("GF_MUSETALK_MLX_URL", "http://127.0.0.1:8040").rstrip("/")
TIMEOUT = float(os.environ.get("GF_MUSETALKMLX_TIMEOUT", "60"))
# Regenerate at most this many faces per second; 0 = every frame (true MuseTalk speed).
REGEN_HZ = float(os.environ.get("GF_MUSETALKMLX_REGEN_HZ", "8"))

_warned_down = False


def load_model():
    """Nothing to load here — the weights live in the worker. We return a token so
    app.py's uniform load_model()/load_avatar()/warm_up() shape still fits."""
    return {"worker": WORKER_URL}


def warm_up(batch_size, model):
    """Fail fast and loudly if the worker is not up.

    Better a refusal at startup than a black video at the first spoken word: this
    backend cannot synthesise anything by itself.
    """
    info = _worker_health()
    if not info:
        raise RuntimeError(
            f"the MuseTalk-MLX worker is not answering at {WORKER_URL}. Start it:\n"
            "  .venv-motion/bin/python voice-gf/tools/musetalk_mlx_worker.py "
            "--weights <mlx-community/MuseTalk-1.5-fp16 snapshot> --port 8040")
    logger.info(f"[musetalk-mlx] worker ready: {info}")
    return info


def _worker_health(retries=1, timeout=4.0):
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(f"{WORKER_URL}/health", timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if attempt < retries:
                time.sleep(1.0)
    return None


def _generate(latents, audio):
    """(B,8,32,32) + (B,50,384) -> (B,256,256,3) uint8 BGR, via the worker."""
    buf = io.BytesIO()
    np.savez(buf, latents=np.ascontiguousarray(latents, np.float32),
             audio=np.ascontiguousarray(audio, np.float16))
    req = urllib.request.Request(f"{WORKER_URL}/generate", data=buf.getvalue(),
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read()
    with np.load(io.BytesIO(raw)) as z:
        return z["faces"]


def load_avatar(avatar_id):
    """Same layout as the other backends, minus the torch latents: `latents.npy`
    instead of `latents.pt`, so no torch tensor is needed on the serving path."""
    avatar_path = f"./data/avatars/{avatar_id}"
    npy = f"{avatar_path}/latents.npy"
    if not os.path.exists(npy):
        face_imgs = f"{avatar_path}/face_imgs"
        hint = (f"It has face_imgs/, so it was built for wav2lip and just needs its "
                f"crops encoded:\n  voice-gf/.venv-motion/bin/python "
                f"voice-gf/tools/build_musetalk_mlx_avatar.py --from-avatar {avatar_id}"
                if os.path.isdir(face_imgs) else
                "Rebuild it from its source clip with --clip.")
        raise RuntimeError(f"avatar '{avatar_id}' has no {npy} — {hint}")
    latents = np.load(npy)                    # (N,8,32,32) float32
    with open(f"{avatar_path}/coords.pkl", "rb") as f:
        coord_list_cycle = pickle.load(f)
    img_list = glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]"))
    img_list = sorted(img_list, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    frame_list_cycle = read_imgs(img_list)
    if len(frame_list_cycle) != len(latents):
        raise RuntimeError(f"{avatar_id}: {len(frame_list_cycle)} frames but "
                           f"{len(latents)} latents — rebuild the avatar")
    logger.info(f"[musetalk-mlx] avatar {avatar_id}: {len(frame_list_cycle)} frames, "
                f"latents {latents.shape}")
    return frame_list_cycle, coord_list_cycle, latents


@register("avatar", "musetalk-mlx")
class MuseTalkMLX(BaseAvatar):
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.frame_list_cycle, self.coord_list_cycle, self.latents = avatar
        self.batch_size = getattr(opt, "batch_size", 8) or 8
        # Seed the cache with the source frames' own mouth region so a worker hiccup
        # degrades to "no lip movement" rather than to a frozen or black video.
        self._face_cache = {}
        self._last_gen = 0.0
        self._min_period = (1.0 / REGEN_HZ) if REGEN_HZ > 0 else 0.0
        self._gen_count = self._reuse_count = 0
        self._worker_ms = 0.0
        audio_processor = Audio2Feature(model_path="./models/whisper")
        self.asr = WhisperASR(opt, self, audio_processor)
        self.asr.warm_up()
        logger.info(f"[musetalk-mlx] worker={WORKER_URL} regen<={REGEN_HZ or 'inf'} Hz")

    # ── inference ───────────────────────────────────────────────────────────
    def inference_batch(self, index, audiofeat_batch):
        global _warned_down
        batch = len(audiofeat_batch)
        idxs = [mirror_index(len(self.frame_list_cycle), index + i) for i in range(batch)]

        if self._min_period and (time.time() - self._last_gen) < self._min_period:
            if all(i in self._face_cache for i in idxs):
                self._reuse_count += len(idxs)
                return np.stack([self._face_cache[i] for i in idxs])

        aud = np.asarray(audiofeat_batch, dtype=np.float32)
        if aud.ndim == 2:                       # (B,384) -> one 50-frame chunk
            aud = aud.reshape(aud.shape[0], 1, -1)
        if aud.shape[1] != 50:
            aud = np.repeat(aud[:, :1, :], 50, axis=1)
        try:
            t0 = time.time()
            faces = _generate(self.latents[idxs], aud)
            self._worker_ms = (time.time() - t0) * 1000
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not _warned_down:
                logger.error(f"[musetalk-mlx] worker unreachable ({e}); holding the "
                             "last mouth until it returns")
                _warned_down = True
            if all(i in self._face_cache for i in idxs):
                self._reuse_count += len(idxs)
                return np.stack([self._face_cache[i] for i in idxs])
            return np.stack([np.zeros((256, 256, 3), np.uint8)] * len(idxs))

        _warned_down = False
        self._last_gen = time.time()
        self._gen_count += len(idxs)
        for i, f in zip(idxs, faces):
            self._face_cache[i] = np.asarray(f, dtype=np.uint8)
        return np.stack([self._face_cache[i] for i in idxs])

    # ── paste-back ──────────────────────────────────────────────────────────
    def paste_back_frame(self, pred_frame, idx: int):
        y1, y2, x1, x2 = self.coord_list_cycle[idx]
        combine = self.frame_list_cycle[idx].copy()
        if BLEND_MODE == "hard":
            combine[y1:y2, x1:x2] = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            return combine
        return _feather_composite(combine, pred_frame, x1, y1, x2, y2, sigma=BLEND_SIGMA)

    def stats(self):
        total = self._gen_count + self._reuse_count
        return {"generated": self._gen_count, "reused": self._reuse_count,
                "reuse_pct": round(100 * self._reuse_count / total, 1) if total else 0.0,
                "regen_hz": REGEN_HZ, "last_worker_ms": round(self._worker_ms, 1)}
