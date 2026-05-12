"""Video composition and subtitle styling routes."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.config import Settings
from app.dependencies import get_settings, get_executor
from app.models.project import SubtitleStyle
from app.models.tool import ConvertRequest
from app.services.compose import pipeline_compose, pipeline_convert
from app.utils.paths import project_dir
from app.utils.state import load_state, save_state
from app.main import sio

router = APIRouter(prefix="/api/project/{name}", tags=["compose"])


def _settings() -> Settings:
    return get_settings()


@router.post("/compose")
async def compose_video(name: str, background_tasks: BackgroundTasks):
    settings = _settings()
    executor = get_executor()
    executor.submit(pipeline_compose, name, settings, sio)
    return {"ok": True}


@router.get("/subtitle-style")
async def get_subtitle_style(name: str):
    settings = _settings()
    state = load_state(name, settings)
    style = state.get("subtitle_style", {})
    from app.utils.ffmpeg import DEFAULT_SUBTITLE_STYLE
    return {**DEFAULT_SUBTITLE_STYLE, **style}


@router.post("/subtitle-style")
async def set_subtitle_style(name: str, style: SubtitleStyle):
    settings = _settings()
    save_state(name, settings, sio, subtitle_style=style.model_dump())
    return {"ok": True}


@router.get("/preview-frame")
async def get_preview_frame(name: str, time: float = 0):
    settings = _settings()
    d = project_dir(name, settings)
    from app.utils.paths import input_video
    inp = input_video(name, settings)
    if not inp.exists():
        raise HTTPException(404)
    import subprocess
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".jpg"))
    subprocess.run([
        settings.FFMPEG_PATH, "-y",
        "-ss", str(time),
        "-i", str(inp),
        "-vframes", "1",
        str(tmp),
    ], check=True, capture_output=True)
    return FileResponse(str(tmp), media_type="image/jpeg",
                       background=lambda: tmp.unlink(missing_ok=True))


@router.post("/convert")
async def convert_video(name: str, data: ConvertRequest, background_tasks: BackgroundTasks):
    settings = _settings()
    executor = get_executor()
    executor.submit(pipeline_convert, name, data.format, data.resolution, settings, sio)
    return {"ok": True}


@router.get("/converted")
async def list_converted(name: str):
    settings = _settings()
    d = project_dir(name, settings)
    files = []
    for f in d.glob("converted.*"):
        files.append({"name": f.name, "size": f.stat().st_size})
    return {"files": files}


@router.get("/converted-download/{fmt}")
async def download_converted(name: str, fmt: str):
    settings = _settings()
    from app.utils.ffmpeg import FORMATS
    if fmt not in FORMATS:
        raise HTTPException(400, detail="不支持的格式")
    p = project_dir(name, settings) / f"converted{FORMATS[fmt]['ext']}"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type=FORMATS[fmt]["mime"],
                       filename=f"{name}_converted{FORMATS[fmt]['ext']}")
