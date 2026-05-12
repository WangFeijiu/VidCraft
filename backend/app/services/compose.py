"""Video composition service — FFmpeg pipeline for final video."""
import json
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from app.config import Settings
from app.utils.ffmpeg import ts, force_style_from, FORMATS, PRESETS
from app.utils.paths import project_dir, input_video, active_sentences_path
from app.utils.state import save_state, load_state


def pipeline_compose(name: str, settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    inp = input_video(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        sentences = json.loads(active_sentences_path(name, settings).read_text("utf-8"))
        state = load_state(name, settings)
        deleted = set(state.get("deleted_sentences", []))
        SR = 24000

        active = [(i, seg) for i, seg in enumerate(sentences, 1) if i not in deleted]
        if not active:
            save_state(name, settings, sio, stage="error", msg="没有可合成的内容（所有句子已被删除）")
            return

        save_state(name, settings, sio, stage="composing", msg="提取片段...")
        tmp_dir = d / "_compose_segments"
        tmp_dir.mkdir(exist_ok=True)

        segment_files = []
        srt_entries = []
        time_offset = 0.0

        for seg_idx, (i, seg) in enumerate(active):
            seg_start, seg_end = seg["start"], seg["end"]
            seg_dur = seg_end - seg_start
            seg_video = tmp_dir / f"seg_{seg_idx:04d}.mp4"

            rec = d / "recordings" / f"s_{i:03d}.webm"
            if not rec.exists():
                rec = d / "recordings" / f"s_{i:03d}_clone.webm"

            if rec.exists():
                wav_tmp = tmp_dir / f"rec_{seg_idx}.wav"
                subprocess.run([ffmpeg, "-y", "-i", str(rec), "-ar", str(SR),
                                "-ac", "1", "-c:a", "pcm_s16le", str(wav_tmp)],
                               check=True, capture_output=True)
                audio, _ = sf.read(str(wav_tmp))
                wav_tmp.unlink(missing_ok=True)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                mx = float(np.max(np.abs(audio)))
                if mx > 0:
                    audio = (audio / mx * 0.95).astype(np.float32)
                rec_wav = tmp_dir / f"rec_{seg_idx}_norm.wav"
                sf.write(str(rec_wav), audio, SR)

                subprocess.run([
                    ffmpeg, "-y",
                    "-ss", str(seg_start), "-to", str(seg_end), "-i", str(inp),
                    "-i", str(rec_wav),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(seg_video),
                ], check=True, capture_output=True)
                rec_wav.unlink(missing_ok=True)
            else:
                subprocess.run([
                    ffmpeg, "-y",
                    "-ss", str(seg_start), "-to", str(seg_end), "-i", str(inp),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k",
                    str(seg_video),
                ], check=True, capture_output=True)

            segment_files.append(seg_video)
            srt_entries.append({"idx": seg_idx + 1, "start": time_offset,
                                "end": time_offset + seg_dur, "text": seg["text"]})
            time_offset += seg_dur
            save_state(name, settings, sio, stage="composing",
                       msg=f"提取片段 {seg_idx + 1}/{len(active)}...")

        save_state(name, settings, sio, stage="composing", msg="拼接视频...")
        from app.utils.ffmpeg import ffmpeg_concat
        concat_video = tmp_dir / "concat.mp4"
        ffmpeg_concat(ffmpeg, segment_files, str(concat_video))

        srt = d / "final.srt"
        with open(str(srt), "w", encoding="utf-8") as f:
            for entry in srt_entries:
                f.write(f"{entry['idx']}\n{ts(entry['start'])} --> {ts(entry['end'])}\n{entry['text']}\n\n")

        save_state(name, settings, sio, stage="composing", msg="合成视频（编码中）...")
        srt_p = str(srt).replace("\\", "/").replace(":", "\\:")
        style = force_style_from(state.get("subtitle_style"))
        subprocess.run([
            ffmpeg, "-y",
            "-i", str(concat_video),
            "-vf", f"subtitles='{srt_p}':force_style='{style}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(d / "final.mp4"),
        ], check=True, capture_output=True, encoding="utf-8", errors="ignore")

        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        save_state(name, settings, sio, stage="done", msg="完成！点击下载")
    except Exception as e:
        save_state(name, settings, sio, stage="error", msg=f"{e}\n{traceback.format_exc()}")


def pipeline_convert(name: str, fmt: str, res: str, settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        inp = input_video(name, settings)
        cfg = FORMATS[fmt]
        out = d / f"converted{cfg['ext']}"

        vf_parts = []
        if PRESETS[res]:
            w, h = PRESETS[res]["w"], PRESETS[res]["h"]
            vf_parts.append(
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
            )

        cmd = [ffmpeg, "-y", "-i", str(inp)]
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        cmd += ["-c:v", cfg["codec"], "-preset", "medium", "-crf", "20",
                "-c:a", cfg["audio"], "-b:a", "192k",
                "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8", errors="ignore")

        save_state(name, settings, sio, stage="done", msg=f"转换完成 ({fmt} {res})")
    except Exception as e:
        save_state(name, settings, sio, stage="error", msg=f"转换失败: {e}\n{traceback.format_exc()}")
