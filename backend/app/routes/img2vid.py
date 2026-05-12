"""Image-to-Video project routes."""
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse

from app.config import Settings
from app.dependencies import get_settings
from app.models.img2vid import (
    Img2VidAnalyzeRequest,
    Img2VidGenerateRequest,
    Img2VidStageUpdate,
    Img2VidThemeUpdate,
)
from app.utils.state import i2v_state, i2v_save
from app.main import sio

router = APIRouter(prefix="/api/img2vid", tags=["img2vid"])


def _settings() -> Settings:
    return get_settings()


def _i2v_dir(name: str, settings: Settings) -> Path:
    return settings.img2vid_dir / name


@router.get("")
async def list_i2v_projects():
    settings = _settings()
    out = []
    if settings.img2vid_dir.exists():
        for d in sorted(settings.img2vid_dir.iterdir()):
            if d.is_dir() and (d / "state.json").exists():
                out.append({"name": d.name, **i2v_state(d.name, settings)})
    return out


@router.post("")
async def create_i2v_project(request: Request):
    settings = _settings()
    form = await request.form()
    name = form.get("name", "").strip()
    theme = form.get("theme", "")
    if not name:
        raise HTTPException(400, detail="项目名必填")
    d = _i2v_dir(name, settings)
    if d.exists():
        raise HTTPException(400, detail="项目已存在")
    d.mkdir(parents=True)
    images_dir = d / "images"
    images_dir.mkdir()
    (d / "recordings").mkdir()

    idx = 0
    for key in sorted(form.keys()):
        f = form[key]
        if hasattr(f, "filename") and f.filename:
            ext = Path(f.filename).suffix or ".jpg"
            content = await f.read()
            (images_dir / f"img_{idx:03d}{ext}").write_bytes(content)
            idx += 1
    if idx == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, detail="请上传至少一张图片")
    i2v_save(name, settings, stage="uploading", theme=theme, image_count=idx)
    return {"name": name, "image_count": idx}


@router.get("/{name}")
async def get_i2v_status(name: str):
    settings = _settings()
    return i2v_state(name, settings)


@router.get("/{name}/images")
async def list_i2v_images(name: str):
    settings = _settings()
    d = _i2v_dir(name, settings) / "images"
    if not d.exists():
        return {"images": []}
    images = []
    for f in sorted(d.glob("img_*")):
        images.append({"name": f.name, "url": f"/api/img2vid/{name}/image/{f.name}"})
    return {"images": images}


@router.get("/{name}/image/{fname}")
async def get_i2v_image(name: str, fname: str):
    settings = _settings()
    p = _i2v_dir(name, settings) / "images" / fname
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))


@router.post("/{name}/theme")
async def update_i2v_theme(name: str, data: Img2VidThemeUpdate):
    settings = _settings()
    i2v_save(name, settings, theme=data.theme)
    return {"ok": True}


@router.post("/{name}/add-images")
async def add_i2v_images(name: str, request: Request):
    settings = _settings()
    d = _i2v_dir(name, settings) / "images"
    existing = sorted(d.glob("img_*"))
    start_idx = len(existing)
    form = await request.form()
    added = 0
    for key in sorted(form.keys()):
        f = form[key]
        if hasattr(f, "filename") and f.filename:
            ext = Path(f.filename).suffix or ".jpg"
            content = await f.read()
            (d / f"img_{start_idx + added:03d}{ext}").write_bytes(content)
            added += 1
    return {"ok": True, "added": added}


@router.post("/{name}/reorder")
async def reorder_i2v_images(name: str, data: dict):
    settings = _settings()
    order = data.get("order", [])
    d = _i2v_dir(name, settings) / "images"
    tmp_dir = d / "_reorder_tmp"
    tmp_dir.mkdir(exist_ok=True)
    try:
        for new_idx, old_name in enumerate(order):
            old_path = d / old_name
            if old_path.exists():
                ext = old_path.suffix
                shutil.move(str(old_path), str(tmp_dir / f"img_{new_idx:03d}{ext}"))
        for f in sorted(tmp_dir.iterdir()):
            shutil.move(str(f), str(d / f.name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"ok": True}


@router.post("/{name}/analyze")
async def analyze_i2v(name: str, data: Img2VidAnalyzeRequest):
    settings = _settings()
    i2v_save(name, settings, stage="analyzing", msg="分析图片中...")
    return {"ok": True, "msg": "图片分析任务已启动（完整实现待迁移）"}


@router.get("/{name}/narration")
async def get_i2v_narration(name: str):
    settings = _settings()
    p = _i2v_dir(name, settings) / "narration.json"
    if not p.exists():
        return {"items": []}
    return {"items": json.loads(p.read_text("utf-8"))}


@router.put("/{name}/narration")
async def save_i2v_narration(name: str, data: dict):
    settings = _settings()
    p = _i2v_dir(name, settings) / "narration.json"
    p.write_text(json.dumps(data.get("items", []), ensure_ascii=False, indent=2), "utf-8")
    return {"ok": True}


@router.post("/{name}/voice-sample")
async def upload_i2v_voice_sample(name: str, sample: UploadFile = File(...)):
    settings = _settings()
    d = _i2v_dir(name, settings)
    content = await sample.read()
    (d / "voice_sample.wav").write_bytes(content)
    return {"ok": True}


@router.post("/{name}/preview-audio")
async def preview_i2v_audio(name: str):
    settings = _settings()
    i2v_save(name, settings, stage="preview", msg="生成预览音频中...")
    return {"ok": True}


@router.put("/{name}/stage")
async def update_i2v_stage(name: str, data: Img2VidStageUpdate):
    settings = _settings()
    i2v_save(name, settings, stage=data.stage)
    return {"ok": True}


@router.get("/{name}/audio/{idx}")
async def get_i2v_audio(name: str, idx: int):
    settings = _settings()
    p = _i2v_dir(name, settings) / "audios" / f"a_{idx:03d}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


@router.post("/{name}/generate")
async def generate_i2v(name: str, data: Img2VidGenerateRequest):
    settings = _settings()
    i2v_save(name, settings, stage="generating", msg="生成视频中...")
    return {"ok": True}


@router.get("/{name}/download")
async def download_i2v(name: str):
    settings = _settings()
    p = _i2v_dir(name, settings) / "final.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4", filename=f"{name}.mp4")


@router.get("/{name}/video")
async def stream_i2v_video(name: str):
    settings = _settings()
    p = _i2v_dir(name, settings) / "final.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4")


@router.delete("/{name}")
async def delete_i2v(name: str):
    settings = _settings()
    d = _i2v_dir(name, settings)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@router.post("/{name}/reset")
async def reset_i2v(name: str):
    settings = _settings()
    d = _i2v_dir(name, settings)
    for f in d.iterdir():
        if f.name == "state.json":
            continue
        if f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
        else:
            f.unlink(missing_ok=True)
    (d / "images").mkdir(exist_ok=True)
    (d / "recordings").mkdir(exist_ok=True)
    i2v_save(name, settings, stage="uploading", msg="数据已清空")
    return {"ok": True}
