"""TTS synthesis with CosyVoice 2 zero-shot voice cloning per SRT segment."""
import os, sys, re

ROOT = r"E:/视频处理/CosyVoice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

PROMPT_AUDIO = r"E:/视频处理/reference_audio_16k.wav"
PROMPT_TEXT  = "可以输入文本或者链接以及图片都可以"
SRT_PATH     = r"E:/视频处理/transcript_merged.srt"
OUT_DIR      = r"E:/视频处理/tts_segments"
MODEL_DIR    = r"E:/视频处理/CosyVoice/pretrained_models/CosyVoice2-0.5B"

os.makedirs(OUT_DIR, exist_ok=True)

print("[1/3] Loading CosyVoice2 model...", flush=True)
cosyvoice = AutoModel(model_dir=MODEL_DIR)
sr = cosyvoice.sample_rate
print(f"  sample_rate = {sr}", flush=True)

def parse_srt(raw):
    pat = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*$",
        re.M,
    )
    matches = list(pat.finditer(raw))
    segs = []
    for i, m in enumerate(matches):
        text_start = m.end()
        text_end = matches[i+1].start() if i+1 < len(matches) else len(raw)
        block = raw[text_start:text_end]
        lines = [l.strip() for l in block.split("\n")]
        while lines and (lines[-1] == "" or lines[-1].isdigit()):
            lines.pop()
        while lines and lines[0] == "":
            lines.pop(0)
        text = " ".join(l for l in lines if l).strip()
        if not text:
            continue
        segs.append((m.group(1), m.group(2), text))
    return segs

with open(SRT_PATH, encoding="utf-8") as f:
    raw = f.read()
segs = parse_srt(raw)
print(f"[2/3] {len(segs)} segments to synthesize", flush=True)

print("[3/3] Synthesizing per segment...", flush=True)
for i, (start_ts, end_ts, text) in enumerate(segs, 1):
    out = os.path.join(OUT_DIR, f"seg_{i:03d}.wav")
    if os.path.exists(out):
        print(f"  #{i:3d} skip (cached)", flush=True)
        continue
    print(f"  #{i:3d} -> {text[:40]}", flush=True)
    chunks = []
    for result in cosyvoice.inference_zero_shot(
        text, PROMPT_TEXT, PROMPT_AUDIO, stream=False
    ):
        chunks.append(result["tts_speech"])
    if not chunks:
        print(f"    WARNING: no audio for seg {i}", flush=True)
        continue
    audio = torch.cat(chunks, dim=-1)
    torchaudio.save(out, audio, sr)

print("Done.", flush=True)
