"""Path utilities — cross-platform project path resolution."""
import json
from pathlib import Path

from app.config import Settings


def project_dir(name: str, settings: Settings) -> Path:
    return settings.projects_dir / name


def input_video(name: str, settings: Settings) -> Path:
    d = project_dir(name, settings)
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
        p = d / f"input{ext}"
        if p.exists():
            return p
    for f in d.glob("input.*"):
        if f.suffix in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
            return f
    return d / "input.mp4"


def active_sentences_path(name: str, settings: Settings) -> Path:
    d = project_dir(name, settings)
    state = load_state_raw(name, settings)
    v = state.get("recording_version", "")
    if v == "uploaded" and (d / "sentences_uploaded.json").exists():
        return d / "sentences_uploaded.json"
    if v == "optimized" and (d / "sentences_optimized.json").exists():
        return d / "sentences_optimized.json"
    if v == "original" and (d / "sentences.json").exists():
        return d / "sentences.json"
    if (d / "sentences_uploaded.json").exists():
        return d / "sentences_uploaded.json"
    if (d / "sentences_optimized.json").exists():
        return d / "sentences_optimized.json"
    return d / "sentences.json"


def load_state_raw(name: str, settings: Settings) -> dict:
    """Read state.json without locking (for path resolution only)."""
    p = project_dir(name, settings) / "state.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return {"stage": "new"}
