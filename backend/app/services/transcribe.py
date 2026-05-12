"""Transcription service — audio extraction + Whisper orchestration."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.utils.paths import project_dir, input_video
from app.utils.state import save_state


def pipeline_transcribe(name: str, settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    inp = input_video(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        save_state(name, settings, sio, stage="processing", msg="提取音频...")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(inp),
             "-vn", "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", str(d / "audio_16k.wav")],
            check=True, capture_output=True, encoding="utf-8", errors="ignore",
        )

        probe = subprocess.run(
            [ffmpeg, "-i", str(inp)],
            capture_output=True, encoding="utf-8", errors="ignore",
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 600
        (d / "_duration.txt").write_text(str(dur))

        save_state(name, settings, sio, stage="processing", msg="语音转写中...",
                   duration=dur, transcribe_progress=[0, int(dur)])

        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent.parent / "workers" / "transcribe_worker.py"),
             name, str(d), str(settings.HF_CACHE_DIR)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        last_progress = None
        while worker.poll() is None:
            time.sleep(2)
            try:
                st = json.loads((d / "state.json").read_text("utf-8"))
                progress = st.get("transcribe_progress")
                if progress != last_progress:
                    last_progress = progress
                    if sio:
                        import asyncio
                        try:
                            asyncio.get_event_loop().run_until_complete(
                                sio.emit("project_update", {"project": name, "state": st}, namespace="/")
                            )
                        except RuntimeError:
                            pass
            except Exception:
                pass

        (d / "_duration.txt").unlink(missing_ok=True)

        if d.exists():
            from app.utils.state import load_state
            st = load_state(name, settings)
            if sio:
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(
                        sio.emit("project_update", {"project": name, "state": st}, namespace="/")
                    )
                except RuntimeError:
                    pass
    except Exception as e:
        import traceback
        save_state(name, settings, sio, stage="error", msg=f"{e}\n{traceback.format_exc()}")
