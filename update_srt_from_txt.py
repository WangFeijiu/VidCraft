"""Inject edited text from transcript_merged.txt back into transcript_merged.srt timestamps."""
import re
from pathlib import Path

SRT = Path(r"E:/视频处理/transcript_merged.srt")
TXT = Path(r"E:/视频处理/transcript_merged.txt")

raw = SRT.read_text(encoding="utf-8")
pat = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*$",
    re.M,
)
matches = list(pat.finditer(raw))
txt_lines = [l.strip() for l in TXT.read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"SRT segs: {len(matches)} | TXT lines: {len(txt_lines)}")
if len(matches) != len(txt_lines):
    raise SystemExit(f"Mismatch — refuse to overwrite. Restore txt to {len(matches)} non-empty lines.")

with open(SRT, "w", encoding="utf-8") as f:
    for i, (m, text) in enumerate(zip(matches, txt_lines), 1):
        f.write(f"{i}\n{m.group(1)} --> {m.group(2)}\n{text}\n\n")
print(f"Updated {SRT}")
