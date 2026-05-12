"""State management with file locking for concurrent access."""
import json
from pathlib import Path

from filelock import FileLock
from loguru import logger

from app.config import Settings
from app.utils.paths import project_dir


def load_state(name: str, settings: Settings) -> dict:
    p = project_dir(name, settings) / "state.json"
    if not p.exists():
        return {"stage": "new"}
    lock = FileLock(str(p) + ".lock", timeout=5)
    with lock:
        return json.loads(p.read_text("utf-8"))


def save_state(name: str, settings: Settings, sio=None, **kw) -> None:
    d = project_dir(name, settings)
    if not d.exists():
        return
    p = d / "state.json"
    lock = FileLock(str(p) + ".lock", timeout=5)
    with lock:
        state = json.loads(p.read_text("utf-8")) if p.exists() else {}
        state.update(kw)
        p.write_text(json.dumps(state, ensure_ascii=False), "utf-8")

    if sio is not None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    sio.emit("project_update", {"project": name, "state": state}, namespace="/")
                )
            else:
                loop.run_until_complete(
                    sio.emit("project_update", {"project": name, "state": state}, namespace="/")
                )
        except Exception as e:
            logger.debug(f"SocketIO emit failed: {e}")


def i2v_state(name: str, settings: Settings) -> dict:
    p = settings.img2vid_dir / name / "state.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return {"stage": "new"}


def i2v_save(name: str, settings: Settings, sio=None, **kw) -> None:
    d = settings.img2vid_dir / name
    p = d / "state.json"
    lock = FileLock(str(p) + ".lock", timeout=5)
    with lock:
        state = json.loads(p.read_text("utf-8")) if p.exists() else {}
        state.update(kw)
        p.write_text(json.dumps(state, ensure_ascii=False), "utf-8")
