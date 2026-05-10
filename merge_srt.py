"""Merge over-segmented SRT into 5-12s chunks and cut at 10:27.

Robust parser: use timestamp lines as anchors. Tolerates missing blank lines and stray indices."""
import re
from pathlib import Path

SRT_IN = Path(r"E:/视频处理/transcript.srt")
SRT_OUT = Path(r"E:/视频处理/transcript_merged.srt")
TXT_OUT = Path(r"E:/视频处理/transcript_merged.txt")
CUTOFF = 627.0  # 10:27

MAX_DUR = 12.0
MAX_GAP = 1.8
MIN_DUR = 5.0

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
        start_ts, end_ts = m.group(1), m.group(2)
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
        segs.append([ts_to_sec(start_ts), ts_to_sec(end_ts), text])
    return segs

raw = SRT_IN.read_text(encoding="utf-8")
segs = parse_srt(raw)
print(f"Parsed: {len(segs)} segs")

trimmed = []
for s, e, t in segs:
    if s >= CUTOFF:
        continue
    if e > CUTOFF:
        e = CUTOFF
    trimmed.append([s, e, t])
segs = trimmed
print(f"After cutoff @{CUTOFF}s: {len(segs)} segs")

def can_merge(cur, nxt):
    cur_dur = cur[1] - cur[0]
    nxt_dur = nxt[1] - nxt[0]
    merged_dur = nxt[1] - cur[0]
    gap = nxt[0] - cur[1]
    if gap > MAX_GAP:
        return False
    if merged_dur > MAX_DUR:
        return False
    if cur_dur < MIN_DUR:
        return True
    if nxt_dur < 1.5 and merged_dur <= MAX_DUR:
        return True
    return False

def join_text(a, b):
    if a and a[-1] in "。！？.!?":
        return a + b
    if a and a[-1] in ",，;；":
        return a + b
    return a + "，" + b

merged = []
i = 0
while i < len(segs):
    cur = list(segs[i]); i += 1
    while i < len(segs) and can_merge(cur, segs[i]):
        cur[1] = segs[i][1]
        cur[2] = join_text(cur[2], segs[i][2])
        i += 1
    merged.append(cur)

print(f"After merge: {len(merged)} segs")

with open(SRT_OUT, "w", encoding="utf-8") as fs, open(TXT_OUT, "w", encoding="utf-8") as ft:
    for n, (s, e, t) in enumerate(merged, 1):
        fs.write(f"{n}\n{sec_to_ts(s)} --> {sec_to_ts(e)}\n{t}\n\n")
        ft.write(f"{t}\n")

print("\n--- Preview ---")
for n, (s, e, t) in enumerate(merged, 1):
    print(f"  #{n:3d} [{s:6.2f}-{e:6.2f}] dur={e-s:5.2f}s | {t}")
