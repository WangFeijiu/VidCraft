"""Compose final video: TTS segments aligned to SRT timestamps + burned-in subtitles."""
import os, sys, re, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

ROOT = Path(r"E:/视频处理")
SRT = ROOT / "transcript_merged.srt"
SEG_DIR = ROOT / "tts_segments"
VIDEO_IN = ROOT / "flowbrain_cut_30_627.mp4"
AUDIO_OUT = ROOT / "final_audio.wav"
VIDEO_OUT = ROOT / "flowbrain_final.mp4"

SR = 24000  # CosyVoice2 native sample rate
TOTAL_SEC = 597.0  # 627 - 30
MAX_SPEEDUP = 1.4   # never speed up more than 1.4x
MIN_SLOWDOWN = 0.85 # if shorter than 85% of slot, leave silence after

FFMPEG = r"D:/Tech/program/python/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"

def ts_to_sec(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

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
        segs.append((ts_to_sec(m.group(1)), ts_to_sec(m.group(2)), text))
    return segs

raw = SRT.read_text(encoding="utf-8")
segs = parse_srt(raw)
print(f"{len(segs)} segments")

total_samples = int(TOTAL_SEC * SR)
out = np.zeros(total_samples, dtype=np.float32)

for i, (start, end, text) in enumerate(segs, 1):
    seg_path = SEG_DIR / f"seg_{i:03d}.wav"
    if not seg_path.exists():
        print(f"  #{i:3d} MISSING -> {seg_path}")
        continue
    audio, sr_in = sf.read(str(seg_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr_in != SR:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=SR)
    target_dur = end - start
    actual_dur = len(audio) / SR

    if actual_dur > target_dur and target_dur > 0:
        speed = actual_dur / target_dur
        if speed > MAX_SPEEDUP:
            speed = MAX_SPEEDUP
            print(f"  #{i:3d} OVERFLOW: a={actual_dur:.2f}s slot={target_dur:.2f}s (cap@{speed:.2f}x)")
        audio = librosa.effects.time_stretch(audio, rate=speed)

    start_idx = int(start * SR)
    end_idx = start_idx + len(audio)
    if end_idx > total_samples:
        audio = audio[: total_samples - start_idx]
        end_idx = total_samples
    out[start_idx:end_idx] = audio
    print(f"  #{i:3d} [{start:6.2f}-{end:6.2f}] slot={end-start:5.2f}s actual={actual_dur:5.2f}s placed")

m = float(np.max(np.abs(out)))
if m > 0:
    out = out / m * 0.95

sf.write(str(AUDIO_OUT), out, SR)
print(f"Audio saved: {AUDIO_OUT}")

print("Muxing video + audio + burning subtitles...")
srt_for_filter = str(SRT).replace("\\", "/").replace(":", r"\:")
cmd = [
    FFMPEG, "-y",
    "-i", str(VIDEO_IN),
    "-i", str(AUDIO_OUT),
    "-vf", f"subtitles='{srt_for_filter}':force_style='FontName=Microsoft YaHei,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=1,Shadow=0,Alignment=2,MarginV=30'",
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    str(VIDEO_OUT),
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Video saved: {VIDEO_OUT}")
