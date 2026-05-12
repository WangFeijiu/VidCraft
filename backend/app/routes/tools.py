"""Standalone video tool workspace routes."""
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import Settings
from app.dependencies import get_settings
from app.models.tool import ToolDeleteRequest, ToolConvertRequest, ToolSpeedupRequest

router = APIRouter(prefix="/api/tool", tags=["tools"])


def _settings() -> Settings:
    return get_settings()


def _session_dir(sid: str, settings: Settings) -> Path:
    return settings.tool_workspace_dir / sid


def _tool_state_path(sid: str, settings: Settings) -> Path:
    return _session_dir(sid, settings) / "state.json"


def _load_tool_state(sid: str, settings: Settings) -> dict:
    p = _tool_state_path(sid, settings)
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


def _save_tool_state(sid: str, settings: Settings, **kw) -> None:
    d = _session_dir(sid, settings)
    d.mkdir(parents=True, exist_ok=True)
    p = _tool_state_path(sid, settings)
    state = json.loads(p.read_text("utf-8")) if p.exists() else {}
    state.update(kw)
    p.write_text(json.dumps(state, ensure_ascii=False), "utf-8")


def _tool_input_video(sid: str, settings: Settings) -> Path:
    d = _session_dir(sid, settings)
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
        p = d / f"input{ext}"
        if p.exists():
            return p
    raise FileNotFoundError("未找到输入视频")


@router.post("/upload")
async def upload_tool_video(video: UploadFile = File(...)):
    settings = _settings()
    sid = uuid.uuid4().hex[:8]
    d = _session_dir(sid, settings)
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(video.filename).suffix or ".mp4"
    content = await video.read()
    (d / f"input{ext}").write_bytes(content)
    _save_tool_state(sid, settings, filename=video.filename, stage="ready")
    return {"sid": sid, "filename": video.filename}


@router.get("/list")
async def list_tool_sessions():
    settings = _settings()
    out = []
    if settings.tool_workspace_dir.exists():
        for d in sorted(settings.tool_workspace_dir.iterdir()):
            if d.is_dir() and (d / "state.json").exists():
                state = _load_tool_state(d.name, settings)
                out.append({"sid": d.name, **state})
    return out


@router.get("/{sid}/video")
async def stream_tool_video(sid: str):
    settings = _settings()
    try:
        p = _tool_input_video(sid, settings)
    except FileNotFoundError:
        raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4")


@router.post("/{sid}/reset-result")
async def reset_tool_result(sid: str):
    settings = _settings()
    d = _session_dir(sid, settings)
    for result in d.glob("result.*"):
        result.unlink(missing_ok=True)
    _save_tool_state(sid, settings, stage="ready")
    return {"ok": True}


@router.get("/{sid}/state")
async def get_tool_state(sid: str):
    settings = _settings()
    return _load_tool_state(sid, settings)


@router.post("/{sid}/edit/delete")
async def tool_edit_delete(sid: str, data: ToolDeleteRequest):
    settings = _settings()
    _save_tool_state(sid, settings, stage="editing", msg="删除片段中...")
    return {"ok": True, "msg": "删除功能待完整迁移"}


@router.post("/{sid}/edit/insert-video")
async def tool_edit_insert_video(sid: str, request: Request):
    settings = _settings()
    _save_tool_state(sid, settings, stage="editing", msg="插入视频中...")
    return {"ok": True, "msg": "插入视频功能待完整迁移"}


@router.post("/{sid}/edit/concat")
async def tool_edit_concat(sid: str, request: Request):
    settings = _settings()
    _save_tool_state(sid, settings, stage="editing", msg="拼接中...")
    return {"ok": True, "msg": "拼接功能待完整迁移"}


@router.post("/{sid}/edit/insert-images")
async def tool_edit_insert_images(sid: str, request: Request):
    settings = _settings()
    _save_tool_state(sid, settings, stage="editing", msg="插入图片中...")
    return {"ok": True, "msg": "图片插入功能待完整迁移"}


@router.post("/{sid}/convert")
async def tool_convert(sid: str, data: ToolConvertRequest):
    settings = _settings()
    _save_tool_state(sid, settings, stage="converting", msg=f"转换为 {data.format}...")
    return {"ok": True, "msg": "转换功能待完整迁移"}


@router.post("/{sid}/edit/speedup")
async def tool_edit_speedup(sid: str, data: ToolSpeedupRequest):
    settings = _settings()
    _save_tool_state(sid, settings, stage="editing", msg="变速中...")
    return {"ok": True, "msg": "变速功能待完整迁移"}


@router.post("/{sid}/edit/speedup-merge")
async def tool_edit_speedup_merge(sid: str, request: Request):
    settings = _settings()
    return {"ok": True, "msg": "变速合并功能待完整迁移"}


@router.post("/{sid}/edit/replace-audio")
async def tool_edit_replace_audio(sid: str, request: Request):
    settings = _settings()
    return {"ok": True, "msg": "替换音频功能待完整迁移"}


@router.get("/{sid}/result")
async def get_tool_result(sid: str):
    settings = _settings()
    d = _session_dir(sid, settings)
    for result in d.glob("result.*"):
        return FileResponse(str(result), media_type="video/mp4")
    raise HTTPException(404)


@router.get("/{sid}/download")
async def download_tool_result(sid: str):
    settings = _settings()
    d = _session_dir(sid, settings)
    for result in d.glob("result.*"):
        return FileResponse(str(result), media_type="video/mp4",
                           filename=f"tool_{sid}{result.suffix}")
    raise HTTPException(404)


@router.delete("/{sid}")
async def delete_tool_session(sid: str):
    settings = _settings()
    d = _session_dir(sid, settings)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}
