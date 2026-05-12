"""FFmpeg subprocess wrapper — unified video/audio processing interface."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


def run_ffmpeg(ffmpeg_path: str, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    cmd = [ffmpeg_path] + args
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "ignore")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0 and kwargs.get("check", False):
        logger.error(f"FFmpeg failed: {result.stderr[-500:] if result.stderr else ''}")
    return result


def get_video_info(ffmpeg_path: str, path: str) -> tuple[int, int, float]:
    probe = subprocess.run([ffmpeg_path, "-i", str(path)],
                          capture_output=True, encoding="utf-8", errors="ignore")
    m = re.search(r",\s*(\d+)x(\d+)\s", probe.stderr or "")
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
    m2 = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
    dur = int(m2.group(1)) * 3600 + int(m2.group(2)) * 60 + float(m2.group(3)) if m2 else 600
    return w, h, dur


def has_audio(ffmpeg_path: str, path: str) -> bool:
    r = subprocess.run([ffmpeg_path, "-i", str(path)],
                      capture_output=True, encoding="utf-8", errors="ignore")
    return "Audio:" in (r.stderr or "")


def ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    sec = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        sec += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def hex_to_ass_color(hex_color: str) -> str:
    h = (hex_color or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b.upper()}{g.upper()}{r.upper()}&"


DEFAULT_SUBTITLE_STYLE = {
    "font_name": "Microsoft YaHei",
    "font_size": 20,
    "primary_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline": 1,
    "position": "bottom",
    "margin_v": 30,
}


def force_style_from(style: dict | None) -> str:
    s = {**DEFAULT_SUBTITLE_STYLE, **(style or {})}
    pos = s.get("position", "bottom")
    alignment = {"bottom": 2, "middle": 5, "top": 8}.get(pos, 2)
    return (
        f"FontName={s['font_name']},FontSize={s['font_size']},"
        f"PrimaryColour={hex_to_ass_color(s['primary_color'])},"
        f"OutlineColour={hex_to_ass_color(s['outline_color'])},"
        f"Outline={s['outline']},Shadow=0,Alignment={alignment},MarginV={s['margin_v']}"
    )


def ffmpeg_concat(ffmpeg_path: str, segment_paths: list, output_path: str) -> None:
    tmpdir = tempfile.mkdtemp(prefix="vs_")
    try:
        lines = []
        for i, seg in enumerate(segment_paths):
            tmp_seg = os.path.join(tmpdir, f"s{i:04d}.mp4")
            shutil.copy2(str(seg), tmp_seg)
            lines.append(f"file 's{i:04d}.mp4'")
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        tmp_out = os.path.join(tmpdir, "out.mp4")
        subprocess.run(
            [ffmpeg_path, "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file, "-c", "copy", tmp_out],
            check=True, capture_output=True, encoding="utf-8", errors="ignore",
        )
        shutil.copy2(tmp_out, str(output_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def normalize_video(ffmpeg_path: str, path_in: str, path_out: str) -> None:
    subprocess.run(
        [ffmpeg_path, "-y", "-i", str(path_in),
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", "25",
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart",
         str(path_out)],
        check=True, capture_output=True, encoding="utf-8", errors="ignore",
    )


def enc_for_concat(ffmpeg_path: str, src_path: str, dst_path: str,
                   w: int, h: int, *, ss=None, t=None, fps=25, sr=44100) -> None:
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    has_a = has_audio(ffmpeg_path, src_path)
    cmd = [ffmpeg_path, "-y", "-i", str(src_path)]
    if not has_a:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={sr}"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-vf", vf, "-r", str(fps),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2"]
    if not has_a:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += [str(dst_path)]
    subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8", errors="ignore")


FORMATS = {
    "mp4": {"ext": ".mp4", "codec": "libx264", "audio": "aac", "mime": "video/mp4"},
    "avi": {"ext": ".avi", "codec": "mpeg4", "audio": "mp3", "mime": "video/x-msvideo"},
    "mkv": {"ext": ".mkv", "codec": "libx264", "audio": "aac", "mime": "video/x-matroska"},
    "mov": {"ext": ".mov", "codec": "libx264", "audio": "aac", "mime": "video/quicktime"},
    "webm": {"ext": ".webm", "codec": "libvpx-vp9", "audio": "libopus", "mime": "video/webm"},
}

PRESETS = {
    "1080p": {"w": 1920, "h": 1080},
    "720p": {"w": 1280, "h": 720},
    "480p": {"w": 854, "h": 480},
    "360p": {"w": 640, "h": 360},
    "original": None,
}
