#!/usr/bin/env python3
"""lt_drive.py — drive a running LiveTalking digital human headlessly and record real output.

Creates a WHEP session (recvonly video+audio, which starts the server-side render
thread), tells the avatar to speak text via its own TTS (or uploads an audio file),
and records the rendered frames+audio with LiveTalking's built-in recorder.

Requires the LiveTalking venv (aiortc + aiohttp):
    ~/src/ai-girlfriend/livetalking/venv/bin/python tools/lt_drive.py --text "你好"

Output lands next to this tool by default; use --out to change it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription

DEFAULT_BASE = "http://127.0.0.1:8010"


def _default_avatar() -> str:
    """An avatar id that actually exists.

    Hardcoding one is how `lt_drive` started returning WHEP 500 — the old
    `wav2lip_avatar_female_model` directory no longer exists, and LiveTalking
    answers a missing avatar with a traceback, not a useful error.
    """
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "avatars")
    wanted = os.environ.get("GF_AVATAR", "").strip()
    if not wanted:
        # gf_server reads GF_AVATAR from the project .env; mirror that so a test
        # drives the same avatar the page does instead of an arbitrary directory.
        env_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), ".env")
        try:
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("GF_AVATAR="):
                        wanted = line.split("=", 1)[1].strip().strip('"\'')
                        break
        except OSError:
            pass
    if wanted and os.path.exists(os.path.join(root, wanted, "coords.pkl")):
        return wanted
    try:
        for name in sorted(os.listdir(root)):
            if os.path.exists(os.path.join(root, name, "coords.pkl")):
                return name
    except OSError:
        pass
    return "wav2lip_avatar_female_model"


DEFAULT_AVATAR = _default_avatar()
# Cantonese female; LiveTalking's own default is zh-CN-YunxiaNeural (male Mandarin)
DEFAULT_VOICE = "Wing"           # macOS say voice; Cantonese female ~222 Hz

stats = {"video": 0, "audio": 0}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


async def post_json(session, base, path, payload):
    async with session.post(base + path, json=payload) as r:
        body = await r.text()
        try:
            return r.status, json.loads(body).get("msg", body[:120])
        except Exception:
            return r.status, body[:120]


async def run(args) -> int:
    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read()
    else:
        text = args.text
    text = text.strip().replace("\n", "，")          # newlines -> spoken pauses
    if not text and not args.audio_file:
        log("nothing to say: pass --text, --text-file or --audio-file")
        return 2
    log("utterance:", text[:90] or f"(audio file {args.audio_file})")

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    # order matters: LiveTalking patches codec preferences on transceiver index 1
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    @pc.on("track")
    def on_track(track):
        log("remote track:", track.kind)

        async def pump():
            try:
                while True:
                    await track.recv()
                    stats[track.kind] += 1
            except Exception as e:
                log("track closed:", track.kind, type(e).__name__)

        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())

    async with aiohttp.ClientSession() as s:
        async with s.post(f"{args.base}/whep?avatar={args.avatar}",
                          data=pc.localDescription.sdp,
                          headers={"Content-Type": "application/sdp"}) as r:
            code, sid, answer = r.status, r.headers.get("X-Session-ID"), await r.text()
        log("WHEP", code, "sessionid =", sid)
        if code not in (200, 201):
            log("WHEP rejected:", answer[:200])
            await pc.close()
            return 1
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))

        # the render thread only starts once a client subscribes; wait for real frames
        for _ in range(60):
            await asyncio.sleep(0.5)
            if stats["video"] > 0:
                break
        log("frames flowing:", stats)
        if stats["video"] == 0:
            log("no video frames arrived; is the avatar loaded?")
            await pc.close()
            return 1

        log("start_record:", await post_json(s, args.base, "/record",
                                             {"sessionid": sid, "type": "start_record"}))
        await asyncio.sleep(0.8)

        if args.audio_file:
            with open(args.audio_file, "rb") as fh:
                form = aiohttp.FormData()
                form.add_field("file", fh, filename=os.path.basename(args.audio_file),
                               content_type="audio/wav")
                form.add_field("sessionid", str(sid))
                async with s.post(f"{args.base}/humanaudio", data=form) as r:
                    log("humanaudio:", r.status, (await r.text())[:120])
        else:
            log("speak:", await post_json(s, args.base, "/human", {
                "sessionid": sid, "type": "echo", "text": text,
                "interrupt": args.interrupt, "tts": {"ref_file": args.voice},
            }))

        saw_speaking = False
        for _ in range(int(args.max_wait * 2)):
            await asyncio.sleep(0.5)
            async with s.post(args.base + "/is_speaking", json={"sessionid": sid}) as r:
                speaking = json.loads(await r.text()).get("data")
            if speaking:
                saw_speaking = True
            elif saw_speaking:
                break
        log("speech seen:", saw_speaking, "| frames:", stats)
        await asyncio.sleep(args.tail)

        log("end_record:", await post_json(s, args.base, "/record",
                                           {"sessionid": sid, "type": "end_record"}))
        await asyncio.sleep(2.5)

        async with s.get(f"{args.base}/record/{sid}") as r:
            log("download:", r.status)
            data = await r.read()
        if not data:
            log("server returned an empty recording")
            await pc.close()
            return 1
        with open(args.out, "wb") as fh:
            fh.write(data)
        log(f"saved {args.out} ({len(data)} bytes), {stats['video']} video frames")

    await pc.close()
    return 0


def main():
    p = argparse.ArgumentParser(description="Drive LiveTalking and record a proof video.")
    p.add_argument("--text", default="", help="text for the avatar to speak")
    p.add_argument("--text-file", default="", help="read the utterance from a file")
    p.add_argument("--audio-file", default="", help="use your own 16k mono wav instead of TTS")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice name")
    p.add_argument("--avatar", default=DEFAULT_AVATAR, help="LiveTalking avatar_id")
    p.add_argument("--base", default=DEFAULT_BASE, help="LiveTalking base URL")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "gf_recording.mp4"))
    p.add_argument("--max-wait", type=float, default=45.0, help="max seconds to wait for speech")
    p.add_argument("--tail", type=float, default=1.5, help="seconds to keep rolling after speech")
    p.add_argument("--interrupt", action="store_true", help="flush any in-progress speech")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
