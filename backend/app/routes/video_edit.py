"""Video editing routes for main project (delete/insert/speedup within project context)."""
from fastapi import APIRouter, HTTPException, Request

from app.config import Settings
from app.dependencies import get_settings
from app.utils.state import save_state
from app.main import sio

router = APIRouter(prefix="/api/project/{name}/video-edit", tags=["video-edit"])


def _settings() -> Settings:
    return get_settings()


@router.post("/delete")
async def project_ve_delete(name: str, request: Request):
    settings = _settings()
    save_state(name, settings, sio, sub="video_edit", msg="删除片段中...")
    return {"ok": True, "msg": "项目视频编辑功能待完整迁移"}


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
    save_state(name, settings, sio, sub="video_edit", msg="变速中...")
    return {"ok": True, "msg": "项目变速功能待完整迁移"}


@router.post("/speedup-merge")
async def project_ve_speedup_merge(name: str, request: Request):
    settings = _settings()
    return {"ok": True, "msg": "项目变速合并功能待完整迁移"}
