###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import json
import os
import time
import cv2
import glob
import pickle

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from utils.blend import composite as _feather_composite, BLEND_MODE, BLEND_SIGMA
from registry import register

# ── inference backend ────────────────────────────────────────────────────────
# torch  = PyTorch on MPS/CPU (original path)
# coreml = the same model compiled for the Apple Neural Engine. Wav2Lip is
#          fixed-shape pure conv, which is what the ANE is for: measured 8.7 ms /
#          frame at batch 16 vs 30.2 on MPS (~3.5x), and it frees the GPU.
#          Build the models with voice-gf/tools/convert_wav2lip_coreml.py.
BACKEND = os.environ.get("GF_WAV2LIP_BACKEND", "torch").strip().lower()
COREML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "wav2lip_coreml")
COREML_FALLBACK_BATCH = 16
# where the ANE path publishes its own rolling inference timing for /metrics
ANE_STATS_PATH = "/tmp/wav2lip_ane_stats.json"

# With the ANE backend the tensors stay on the CPU deliberately: inference_batch
# does `.to(device)`, and copying to the Metal GPU only to hand the data back
# would add a pointless round trip per frame.
device = torch.device("cpu") if BACKEND == "coreml" else initialize_device()
logger.info('Using {} for inference (backend={}).'.format(device, BACKEND))


class CoreMLWav2Lip:
    """Drop-in replacement for the torch model, running on the Neural Engine.

    Returns a CPU tensor of the same (B,3,H,W) shape the caller expects, so
    `inference_batch` needs no changes. The compiled graph has a fixed batch, so
    other sizes are padded by repeating the last sample and trimmed afterwards —
    that keeps any `batch_size` in config.yaml working.
    """

    def __init__(self, res=256):
        self.res = res
        self._models = {}
        self._frames = 0
        self._last_ms = 0.0
        self._last_write = 0.0

    def _pick(self, batch):
        exact = os.path.join(COREML_DIR, f"wav2lip_b{batch}.mlpackage")
        if os.path.exists(exact):
            return batch, exact
        fallback = os.path.join(COREML_DIR,
                                f"wav2lip_b{COREML_FALLBACK_BATCH}.mlpackage")
        if os.path.exists(fallback):
            return COREML_FALLBACK_BATCH, fallback
        raise FileNotFoundError(
            f"no CoreML model in {COREML_DIR}. Build one with:\n"
            f"  ./livetalking/venv/bin/python voice-gf/tools/convert_wav2lip_coreml.py "
            f"--batches 1,{batch}")

    def _model(self, batch):
        key, path = self._pick(batch)
        if key not in self._models:
            import coremltools as ct
            logger.info(f"Loading CoreML wav2lip (batch {key}) from {path}")
            self._models[key] = ct.models.MLModel(
                path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        return key, self._models[key]

    def _write_stats(self):
        """Publish rolling inference timing for /metrics. Atomic (tmp + rename) and
        JSON-safe: an MLModel object is not serialisable, and a mid-write failure
        would otherwise leave a truncated file behind."""
        try:
            payload = {"backend": "coreml", "device": "ane",
                       "ms_per_frame": round(self._last_ms, 2),
                       "fps": round(1000.0 / self._last_ms, 1) if self._last_ms else None,
                       "frames_total": self._frames,
                       "batches": sorted(int(k) for k in self._models),
                       "updated": round(time.time(), 2)}
            tmp = ANE_STATS_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, ANE_STATS_PATH)
        except Exception as e:
            logger.warning(f"ANE stats write failed: {e}")

    def __call__(self, mel, img):
        target, model = self._model(int(mel.shape[0]))
        m = np.ascontiguousarray(mel.detach().cpu().numpy(), dtype=np.float32)
        i = np.ascontiguousarray(img.detach().cpu().numpy(), dtype=np.float32)
        b = m.shape[0]
        if b != target:
            pad = target - b
            m = np.concatenate([m, np.repeat(m[-1:], pad, axis=0)], axis=0)
            i = np.concatenate([i, np.repeat(i[-1:], pad, axis=0)], axis=0)
        # The compiled graph is fixed at res x res. An avatar built with a different
        # img_size (300, 400 ...) would otherwise fail the shape check, and the
        # failure surfaces as a frozen mouth with no obvious error. Resize instead
        # of refusing, so any avatar still lip-syncs.
        if i.shape[2] != self.res or i.shape[3] != self.res:
            if not getattr(self, "_warned_res", False):
                logger.warning(f"CoreML wav2lip: avatar crops are {i.shape[3]}x{i.shape[2]}, "
                               f"resizing to {self.res}x{self.res}. Rebuild the avatar "
                               f"with img_size={self.res} for best quality.")
                self._warned_res = True
            # 6 channels = masked face (3) + reference face (3); cv2.resize tops out
            # at 4 channels, so resize the two halves as separate images
            i = np.stack([np.concatenate([
                cv2.resize(f[:3].transpose(1, 2, 0), (self.res, self.res),
                           interpolation=cv2.INTER_AREA).transpose(2, 0, 1),
                cv2.resize(f[3:].transpose(1, 2, 0), (self.res, self.res),
                           interpolation=cv2.INTER_AREA).transpose(2, 0, 1)], axis=0)
                for f in i], axis=0)
        t0 = time.time()
        out = list(model.predict({"mel": m, "img": i}).values())[0]
        self._last_ms = (time.time() - t0) * 1000.0 / max(b, 1)
        self._frames += b
        now = time.time()
        if now - self._last_write > 1.0:
            self._last_write = now
            self._write_stats()
        return torch.from_numpy(np.asarray(out)[:b])

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    if BACKEND == "coreml":
        logger.info("wav2lip backend=coreml — using the Neural Engine")
        return CoreMLWav2Lip()
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = self.frame_list_cycle[idx].copy()
        y1, y2, x1, x2 = bbox

        if BLEND_MODE == "hard":
            res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
            combine_frame[y1:y2, x1:x2] = res_frame
            return combine_frame

        return _feather_composite(combine_frame, pred_frame, x1, y1, x2, y2,
                                  sigma=BLEND_SIGMA)

