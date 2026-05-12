"""Sentence/subtitle management routes."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, FileResponse

from app.config import Settings
from app.dependencies import get_settings
from app.models.sentences import SentencesSave, OptimizeRequest, MatchSubtitlesRequest
from app.utils.paths import project_dir, active_sentences_path
from app.utils.state import load_state, save_state
from app.services.optimize import pipeline_optimize
from app.main import sio

router = APIRouter(prefix="/api/project/{name}", tags=["sentences"])


def _settings() -> Settings:
    return get_settings()


@router.get("/sentences")
async def get_sentences(name: str):
    settings = _settings()
    d = project_dir(name, settings)
    versions = {}
    for v in ("original", "optimized", "uploaded"):
        p = d / f"sentences_{v}.json" if v != "original" else d / "sentences.json"
        if p.exists():
            versions[v] = json.loads(p.read_text("utf-8"))
    active = active_sentences_path(name, settings)
    active_name = "original"
    if "uploaded" in active.name:
        active_name = "uploaded"
    elif "optimized" in active.name:
        active_name = "optimized"
    return {"active": active_name, "versions": versions,
            "sentences": json.loads(active.read_text("utf-8")) if active.exists() else []}


@router.put("/sentences")
async def save_sentences(name: str, data: SentencesSave):
    settings = _settings()
    d = project_dir(name, settings)
    version = data.version or "original"
    if version == "original":
        p = d / "sentences.json"
    else:
        p = d / f"sentences_{version}.json"
    p.write_text(json.dumps([s.model_dump() for s in data.sentences],
                            ensure_ascii=False, indent=2), "utf-8")
    if data.clear_after:
        for v in ("optimized", "uploaded"):
            (d / f"sentences_{v}.json").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/export")
async def export_sentences(name: str, format: str = "srt"):
    settings = _settings()
    p = active_sentences_path(name, settings)
    if not p.exists():
        raise HTTPException(404)
    sentences = json.loads(p.read_text("utf-8"))
    if format == "json":
        return sentences
    if format == "txt":
        return PlainTextResponse("\n".join(s["text"] for s in sentences))
    from app.utils.ffmpeg import ts
    lines = []
    for i, s in enumerate(sentences, 1):
        lines.append(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n")
    return PlainTextResponse("\n".join(lines))


@router.post("/optimize")
async def optimize_sentences(name: str, data: OptimizeRequest, background_tasks: BackgroundTasks):
    settings = _settings()
    d = project_dir(name, settings)
    version = data.version or "original"
    if version == "original":
        p = d / "sentences.json"
    else:
        p = d / f"sentences_{version}.json"
    if not p.exists():
        raise HTTPException(404, detail="未找到字幕文件")
    background_tasks.add_task(pipeline_optimize, name, str(p), data.description, settings, sio)
    return {"ok": True}


@router.post("/match-subtitles")
async def match_subtitles(name: str, data: MatchSubtitlesRequest):
    settings = _settings()
    return {"ok": True, "msg": "字幕匹配功能暂未实现"}


@router.get("/deleted-sentences")
async def get_deleted_sentences(name: str):
    settings = _settings()
    state = load_state(name, settings)
    deleted = state.get("deleted_sentences", [])
    return {"deleted": deleted}


@router.post("/delete-sentence/{idx}")
async def delete_sentence(name: str, idx: int):
    settings = _settings()
    state = load_state(name, settings)
    deleted = set(state.get("deleted_sentences", []))
    deleted.add(idx)
    save_state(name, settings, sio, deleted_sentences=sorted(deleted))
    return {"ok": True}


@router.post("/restore-sentence/{idx}")
async def restore_sentence(name: str, idx: int):
    settings = _settings()
    state = load_state(name, settings)
    deleted = set(state.get("deleted_sentences", []))
    deleted.discard(idx)
    save_state(name, settings, sio, deleted_sentences=sorted(deleted))
    return {"ok": True}


@router.get("/sentence-clip/{idx}")
async def get_sentence_clip(name: str, idx: int):
    settings = _settings()
    d = project_dir(name, settings)
    p = active_sentences_path(name, settings)
    if not p.exists():
        raise HTTPException(404)
    sentences = json.loads(p.read_text("utf-8"))
    if idx < 1 or idx > len(sentences):
        raise HTTPException(404)
    seg = sentences[idx - 1]
    import subprocess
    import tempfile
    from app.utils.paths import input_video
    inp = input_video(name, settings)
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    subprocess.run([
        settings.FFMPEG_PATH, "-y",
        "-ss", str(seg["start"]), "-to", str(seg["end"]),
        "-i", str(inp),
        "-c", "copy", str(tmp),
    ], check=True, capture_output=True)
    return FileResponse(str(tmp), media_type="video/mp4",
                       filename=f"{name}_s{idx:03d}.mp4",
                       background=lambda: tmp.unlink(missing_ok=True))
