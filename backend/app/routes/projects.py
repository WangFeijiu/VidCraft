"""Project management routes — CRUD operations."""
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.config import Settings
from app.dependencies import get_settings
from app.utils.paths import project_dir, input_video
from app.utils.state import load_state, save_state
from app.services.transcribe import pipeline_transcribe
from app.main import sio

router = APIRouter(prefix="/api", tags=["projects"])


def _settings() -> Settings:
    return get_settings()


@router.get("/projects")
async def list_projects():
    settings = _settings()
    out = []
    if settings.projects_dir.exists():
        for d in sorted(settings.projects_dir.iterdir()):
            if d.is_dir() and (d / "state.json").exists():
                out.append({"name": d.name, **load_state(d.name, settings)})
    return out


@router.post("/projects")
async def create_project(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    video: UploadFile = File(...),
):
    settings = _settings()
    if not name.strip():
        raise HTTPException(400, detail="项目名必填")
    d = settings.projects_dir / name
    if d.exists():
        raise HTTPException(400, detail="项目已存在")
    d.mkdir(parents=True)
    (d / "recordings").mkdir()
    ext = Path(video.filename).suffix or ".mp4"
    video_path = d / f"input{ext}"
    content = await video.read()
    video_path.write_bytes(content)
    save_state(name, settings, sio, stage="processing", msg="提取音频...", orig_ext=ext)
    background_tasks.add_task(pipeline_transcribe, name, settings, sio)
    return {"name": name}


@router.delete("/project/{name}")
async def delete_project(name: str):
    settings = _settings()
    d = project_dir(name, settings)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@router.post("/project/{name}/reset")
async def reset_project(name: str):
    settings = _settings()
    d = project_dir(name, settings)
    for f in d.iterdir():
        if f.name == "state.json":
            continue
        if f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
        else:
            f.unlink(missing_ok=True)
    (d / "recordings").mkdir(exist_ok=True)
    save_state(name, settings, sio, stage="editing", msg="数据已清空，请上传视频")
    return {"ok": True}


@router.get("/project/{name}")
async def project_status(name: str):
    settings = _settings()
    s = load_state(name, settings)
    rec_dir = project_dir(name, settings) / "recordings"
    s["recorded"] = len([f for f in rec_dir.glob("s_*.webm") if "_clone" not in f.name]) if rec_dir.exists() else 0
    return s


@router.put("/project/{name}/stage")
async def set_stage(name: str, data: dict):
    settings = _settings()
    stage = data.get("stage", "")
    if stage in ("editing", "recording"):
        updates = {"stage": stage, "msg": ""}
        if stage == "recording":
            version = data.get("version", "")
            if version in ("original", "optimized", "uploaded"):
                updates["recording_version"] = version
        save_state(name, settings, sio, **updates)
    return {"ok": True}


@router.get("/project/{name}/has-recordings")
async def has_recordings(name: str):
    settings = _settings()
    rec_dir = project_dir(name, settings) / "recordings"
    has = rec_dir.exists() and any(f.suffix in (".webm", ".wav", ".mp3") for f in rec_dir.iterdir())
    return {"has_recordings": has}


@router.get("/project/{name}/has-video")
async def has_video(name: str):
    settings = _settings()
    return {"has": input_video(name, settings).exists()}


@router.post("/project/{name}/reupload")
async def reupload_video(
    name: str,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    settings = _settings()
    d = project_dir(name, settings)
    for old in d.glob("input.*"):
        old.unlink(missing_ok=True)
    ext = Path(video.filename).suffix or ".mp4"
    content = await video.read()
    (d / f"input{ext}").write_bytes(content)
    save_state(name, settings, sio, stage="processing", msg="视频上传成功，开始转写...")
    background_tasks.add_task(pipeline_transcribe, name, settings, sio)
    return {"ok": True}


@router.get("/project/{name}/video")
async def stream_video(name: str):
    settings = _settings()
    inp = input_video(name, settings)
    if not inp.exists():
        raise HTTPException(404)
    mime_map = {".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
                ".mov": "video/quicktime", ".webm": "video/webm"}
    mime = mime_map.get(inp.suffix, "video/mp4")
    return FileResponse(str(inp), media_type=mime)


@router.get("/project/{name}/final-video")
async def stream_final_video(name: str):
    settings = _settings()
    p = project_dir(name, settings) / "final.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4")


@router.get("/project/{name}/download")
async def download_video(name: str):
    settings = _settings()
    d = project_dir(name, settings)
    p = d / "final.mp4"
    if p.exists():
        return FileResponse(str(p), media_type="video/mp4",
                           filename=f"{name}_final.mp4")
    inp = input_video(name, settings)
    if inp.exists():
        return FileResponse(str(inp), media_type="video/mp4",
                           filename=inp.name)
    raise HTTPException(404)
