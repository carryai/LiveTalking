"""macOS system TTS provider (zh_HK Cantonese voices + any other installed voice).

LiveTalking's edge-tts backend only offers three Cantonese voices
(zh-HK-HiuGaaiNeural, zh-HK-HiuMaanNeural, zh-HK-WanLungNeural). macOS ships its
own zh_HK set — Sinji, Wing, Fung, Aasing — several at Premium/Enhanced quality.
This provider synthesises through the system `say` binary so those voices can be
used live, exactly like edge-tts voices.

Enable it either as the backend:

    tts: macsay
    REF_FILE: Sinji

or keep `tts: edgetts` — tts/edge.py falls back to this module automatically for
any voice name that is not an edge-tts id (edge ids always contain a dash).
"""
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import resampy
import soundfile as sf

from registry import register
from utils.logger import logger

from .base_tts import BaseTTS, State

DEFAULT_VOICE = "Sinji"
# the Chinese names macOS shows in System Settings -> Spoken Content
ALIASES = {
    "善怡": "Sinji",
    "阿成": "Aasing",
}
# the Cantonese voices that ship with macOS
MAC_VOICES = ["Sinji", "Wing", "Fung", "Aasing"]


def mac_voices() -> list[str]:
    """Voices actually installed on this machine (name only, first word)."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        logger.exception("macsay: could not list voices")
        return []
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("zh_HK"):
            if parts[0] not in names:
                names.append(parts[0])
    return names or MAC_VOICES


def synthesize(text: str, voice: str) -> tuple[np.ndarray, int]:
    """Run `say` and return (float32 mono stream, sample_rate). Raises on failure.

    Note: `say -o x.aiff --data-format=...` fails with "Opening output file
    failed: fmt?" — with an explicit data format the container must be one `say`
    can infer, so we write a .caf (pcm_s16le @24 kHz, mono).
    """
    voice = ALIASES.get(voice, voice)
    tmpdir = tempfile.mkdtemp(prefix="macsay_")
    path = os.path.join(tmpdir, "out.caf")
    try:
        # list form, no shell: text is passed verbatim, never interpreted
        proc = subprocess.run(
            ["say", "-v", voice, "-o", path, "--data-format=LEI16@24000", text],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"say failed ({proc.returncode}): {proc.stderr.strip()[:200]}")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError("say produced an empty file")
        stream, sr = sf.read(path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    stream = np.asarray(stream, dtype=np.float32)
    if stream.ndim > 1:
        stream = stream[:, 0]
    return stream, sr


@register("tts", "macsay")
class MacSayTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        speak_with_say(self, msg)


def speak_with_say(tts, msg: tuple[str, dict]):
    """Shared implementation, so the edge-tts backend can delegate a macOS voice
    without borrowing another class's bound method."""
    text, textevent = msg
    voice = textevent.get("tts", {}).get("ref_file") or tts.opt.REF_FILE or DEFAULT_VOICE
    t = time.time()
    try:
        stream, sample_rate = synthesize(text, voice)
    except Exception:
        logger.exception("macsay tts failed")
        return

    if sample_rate != tts.sample_rate and stream.shape[0] > 0:
        stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=tts.sample_rate)
    logger.info(f"-------macsay tts time:{time.time() - t:.4f}s voice={voice} "
                f"dur={stream.shape[0] / tts.sample_rate:.2f}s")

    if stream.shape[0] <= 0:
        logger.error("macsay produced no samples")
        return

    streamlen = stream.shape[0]
    idx = 0
    while streamlen >= tts.chunk and tts.state == State.RUNNING:
        eventpoint = {}
        streamlen -= tts.chunk
        if idx == 0:
            eventpoint = {"status": "start", "text": text}
        elif streamlen < tts.chunk:
            eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        tts.parent.put_audio_frame(stream[idx:idx + tts.chunk], eventpoint)
        idx += tts.chunk
