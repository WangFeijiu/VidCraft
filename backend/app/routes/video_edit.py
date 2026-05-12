"""Video editing routes for main project (delete/insert/speedup within project context)."""
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from app.config import Settings
from app.dependencies import get_settings, get_executor
from app.services.video_edit import pipeline_ve_delete, pipeline_ve_speedup, parse_ranges
from app.utils.state import save_state
from app.main import sio

router = APIRouter(prefix="/api/project/{name}/video-edit", tags=["video-edit"])


def _settings() -> Settings:
    return get_settings()


@router.post("/delete")
async def project_ve_delete(name: str, data: dict):
    settings = _settings()
    ranges_text = data.get("ranges", "")
    if not ranges_text.strip():
        raise HTTPException(400, detail="请输入要删除的时间段")
    try:
        ranges = parse_ranges(ranges_text)
    except Exception as e:
        raise HTTPException(400, detail=str(e))
    save_state(name, settings, sio, sub="video_edit", msg="视频裁剪中...")
    executor = get_executor()
    executor.submit(pipeline_ve_delete, name, ranges, settings, sio)
    return {"ok": True}


@router.post("/insert-video")
async def project_ve_insert_video(name: str, request: Request):
    settings = _settings()
    save_state(name, settings, sio, sub="video_edit", msg="插入视频中...")
    return {"ok": True, "msg": "项目视频插入功能待完整迁移"}


@router.post("/insert-images")
async def project_ve_insert_images(name: str, request: Request):
    settings = _settings()
    save_state(name, settings, sio, sub="video_edit", msg="插入图片中...")
    return {"ok": True, "msg": "项目图片插入功能待完整迁移"}


@router.post("/speedup")
async def project_ve_speedup(name: str, data: dict):
    settings = _settings()
    try:
        start = float(data.get("start", 0))
        end = float(data.get("end", 0))
        rate = float(data.get("rate", 2))
    except (ValueError, TypeError):
        raise HTTPException(400, detail="参数格式错误")
    if end <= start or rate <= 1:
        raise HTTPException(400, detail="结束时间需大于起始时间，倍速需大于1")
    save_state(name, settings, sio, sub="video_edit", msg="变速中...")
    executor = get_executor()
    executor.submit(pipeline_ve_speedup, name, start, end, rate, settings, sio)
    return {"ok": True}


@router.post("/speedup-merge")
async def project_ve_speedup_merge(name: str, request: Request):
    settings = _settings()
    return {"ok": True, "msg": "项目变速合并功能待完整迁移"}
