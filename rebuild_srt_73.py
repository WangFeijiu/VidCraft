"""Rebuild 73-segment SRT from edited txt: front 6 from original transcript.srt, back 67 shifted +30s."""
import re
from pathlib import Path

TXT = Path(r"E:/视频处理/transcript_merged.txt")
SRT_CUR = Path(r"E:/视频处理/transcript_merged.srt")  # current 67 segs (already -30s)
SRT_ORIG = Path(r"E:/视频处理/transcript.srt")        # raw 240-seg ASR / user-edited
SRT_OUT = Path(r"E:/视频处理/transcript_merged.srt")  # overwrite

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
    return [(ts_to_sec(m.group(1)), ts_to_sec(m.group(2)))
            for m in pat.finditer(raw)]

txt_lines = [l.strip() for l in TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
cur_segs = parse_srt(SRT_CUR.read_text(encoding="utf-8"))
orig_segs = parse_srt(SRT_ORIG.read_text(encoding="utf-8"))

print(f"TXT lines: {len(txt_lines)}")
print(f"Current SRT segs: {len(cur_segs)}")
print(f"Original SRT segs: {len(orig_segs)}")

front_count = len(txt_lines) - len(cur_segs)
print(f"Need {front_count} new front segments + {len(cur_segs)} shifted back segments")
assert front_count >= 0, "txt has fewer lines than current SRT"

new_segs = []
for i in range(front_count):
    new_segs.append(orig_segs[i])
for s, e in cur_segs:
    new_segs.append((s + 30.0, e + 30.0))

with open(SRT_OUT, "w", encoding="utf-8") as f:
    for n, ((s, e), text) in enumerate(zip(new_segs, txt_lines), 1):
        f.write(f"{n}\n{sec_to_ts(s)} --> {sec_to_ts(e)}\n{text}\n\n")

print(f"Wrote {SRT_OUT} with {len(new_segs)} segs (last ends @ {new_segs[-1][1]:.2f}s)")
