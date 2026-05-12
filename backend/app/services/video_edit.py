"""Video editing pipelines for main project — delete, insert, speedup, audio replace."""
import json
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.utils.ffmpeg import get_video_info, ffmpeg_concat, enc_for_concat
from app.utils.paths import project_dir, input_video
from app.utils.state import save_state


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """Parse '10-11, 20-21, 30-32' or '00:01:20-00:02:30' into [(start, end), ...]."""
    ranges = []
    for part in re.split(r"[,;，；\n\r]+", text):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"([\d:.]+)\s*[-~]\s*([\d:.]+)", part)
        if not m:
            raise ValueError(f"时间段格式错误: {part}")
        ranges.append((_parse_ts(m.group(1)), _parse_ts(m.group(2))))
    return sorted(ranges, key=lambda x: x[0])


def _parse_ts(s: str) -> float:
    parts = s.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def pipeline_ve_delete(name: str, ranges: list[tuple[float, float]],
                      settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        inp = input_video(name, settings)
        _, _, total = get_video_info(ffmpeg, str(inp))

        keep = []
        prev_end = 0.0
        for start, end in ranges:
            if start > prev_end:
                keep.append((prev_end, start))
            prev_end = max(prev_end, end)
        if prev_end < total:
            keep.append((prev_end, total))
        if not keep:
            raise ValueError("删除后没有剩余内容")

        seg_files = []
        for i, (s, e) in enumerate(keep):
            seg = d / f"_seg_{i:03d}.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(inp), "-ss", str(s), "-to", str(e),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                 "-c:a", "aac", "-b:a", "192k", str(seg)],
                check=True, capture_output=True, encoding="utf-8", errors="ignore",
            )
            seg_files.append(seg)

        out = d / "input_edited.mp4"
        ffmpeg_concat(ffmpeg, seg_files, str(out))

        inp.rename(d / "input_backup.mp4")
        out.rename(d / "input.mp4")

        for f in seg_files:
            f.unlink(missing_ok=True)

        save_state(name, settings, sio, stage="editing", sub="video_edit_done",
                   msg="视频裁剪完成")
    except Exception as e:
        save_state(name, settings, sio, stage="error",
                   msg=f"视频裁剪失败: {e}\n{traceback.format_exc()}")


def pipeline_ve_speedup(name: str, start: float, end: float, rate: float,
                       settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        inp = input_video(name, settings)
        w, h, total_dur = get_video_info(ffmpeg, str(inp))

        part1 = d / "_sp_part1.mp4"
        part2 = d / "_sp_part2.mp4"
        speed_raw = d / "_sp_raw.mp4"
        speed_up = d / "_sp_fast.mp4"
        out = d / "input_edited.mp4"

        if start > 0:
            enc_for_concat(ffmpeg, str(inp), str(part1), w, h, t=start)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(inp), "-ss", str(start), "-to", str(end),
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-r", "25", "-pix_fmt", "yuv420p", str(speed_raw)],
            check=True, capture_output=True, encoding="utf-8", errors="ignore",
        )
        subprocess.run(
            [ffmpeg, "-y", "-i", str(speed_raw),
             "-vf", f"setpts=PTS/{rate}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(speed_up)],
            check=True, capture_output=True, encoding="utf-8", errors="ignore",
        )
        if end < total_dur - 0.1:
            enc_for_concat(ffmpeg, str(inp), str(part2), w, h, ss=end)

        _, _, new_seg_dur = get_video_info(ffmpeg, str(speed_up))

        segments = []
        if start > 0:
            segments.append(part1)
        segments.append(speed_up)
        if end < total_dur - 0.1:
            segments.append(part2)
        ffmpeg_concat(ffmpeg, segments, str(out))

        meta = {"start": start, "end": end, "rate": rate,
                "new_start": start, "new_end": start + new_seg_dur,
                "new_seg_duration": new_seg_dur}
        (d / "speedup_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        if (d / "input.mp4").exists():
            (d / "input.mp4").rename(d / "input_backup_sp.mp4")
        out.rename(d / "input.mp4")

        for f in [part1, part2, speed_raw, speed_up]:
            f.unlink(missing_ok=True)

        save_state(name, settings, sio, stage="editing", sub="speedup_done",
                   msg="变速完成！请录制新音频")
    except Exception as e:
        save_state(name, settings, sio, stage="error",
                   msg=f"变速失败: {e}\n{traceback.format_exc()}")
