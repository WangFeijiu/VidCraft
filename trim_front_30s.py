"""Trim front 30s: drop SRT segs with start<30, shift remaining timestamps, re-cut video to [30, 627]."""
import re, subprocess
from pathlib import Path

SRT = Path(r"E:/视频处理/transcript_merged.srt")
TXT = Path(r"E:/视频处理/transcript_merged.txt")
VIDEO_IN = Path(r"E:/视频处理/flowbrain_combined.mp4")
VIDEO_OUT = Path(r"E:/视频处理/flowbrain_cut_30_627.mp4")

START_OFFSET = 30.0
END_CUTOFF = 627.0
FFMPEG = r"D:/Tech/program/python/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"

def ts_to_sec(ts):
    ts = ts.replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def sec_to_ts(x):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    ms = int(round((x - int(x)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

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
        segs.append([ts_to_sec(m.group(1)), ts_to_sec(m.group(2)), text])
    return segs

raw = SRT.read_text(encoding="utf-8")
segs = parse_srt(raw)
print(f"Original segs: {len(segs)}")

filtered = []
for s, e, t in segs:
    if s < START_OFFSET:
        print(f"  drop: [{s:6.2f}-{e:6.2f}] {t[:40]}")
        continue
    filtered.append([s - START_OFFSET, e - START_OFFSET, t])

print(f"After trim: {len(filtered)} segs, last ends @ {filtered[-1][1]:.2f}s")

with open(SRT, "w", encoding="utf-8") as fs, open(TXT, "w", encoding="utf-8") as ft:
    for n, (s, e, t) in enumerate(filtered, 1):
        fs.write(f"{n}\n{sec_to_ts(s)} --> {sec_to_ts(e)}\n{t}\n\n")
        ft.write(f"{t}\n")
print(f"Wrote {SRT}, {TXT}")

cmd = [
    FFMPEG, "-y",
    "-ss", str(START_OFFSET), "-to", str(END_CUTOFF),
    "-i", str(VIDEO_IN),
    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    str(VIDEO_OUT),
]
print("Cutting video:", " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"Video saved: {VIDEO_OUT}")
