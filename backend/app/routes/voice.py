"""Voice cloning and recording routes."""
import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.config import Settings
from app.dependencies import get_settings, get_executor
from app.models.voice import VoicePreset, VoicePreviewRequest, RegenerateCloneRequest, SelectSourceRequest, RenameVoiceRequest
from app.services.voice_clone import VOICE_LIBRARY, clear_stale_clones, pipeline_voice_clone
from app.utils.paths import project_dir, active_sentences_path
from app.utils.state import load_state, save_state
from app.main import sio

router = APIRouter(prefix="/api", tags=["voice"])


def _settings() -> Settings:
    return get_settings()


@router.get("/voices")
async def list_voices():
    settings = _settings()
    out = [VoicePreset(**v) for v in VOICE_LIBRARY]
    custom_dir = settings.custom_voices_dir
    if custom_dir.exists():
        for d in sorted(custom_dir.iterdir()):
            if d.is_dir() and (d / "sample.wav").exists():
                meta = {}
                if (d / "meta.json").exists():
                    meta = json.loads((d / "meta.json").read_text("utf-8"))
                out.append(VoicePreset(
                    id=d.name,
                    name=meta.get("name", d.name),
                    desc=meta.get("desc", "自定义音色"),
                    instruct="",
                ))
    return out


@router.post("/project/{name}/voice-clone")
async def start_voice_clone(name: str, background_tasks: BackgroundTasks,
                            voice_id: str = Form(...), prompt_text: str = Form("")):
    settings = _settings()
    d = project_dir(name, settings)
    if not (d / "voice_sample.wav").exists() and not voice_id.startswith("custom_"):
        if not any(v["id"] == voice_id for v in VOICE_LIBRARY):
            raise HTTPException(400, detail="未找到音色")
    clear_stale_clones(name, settings)
    save_state(name, settings, sio, voice_id=voice_id, clone_prompt_text=prompt_text)
    executor = get_executor()
    executor.submit(pipeline_voice_clone, name, prompt_text, voice_id, settings, sio)
    return {"ok": True}


@router.post("/project/{name}/cancel-clone")
async def cancel_clone(name: str):
    from app.services.voice_clone import _cancel_clone
    _cancel_clone.add(name)
    return {"ok": True}


@router.post("/project/{name}/resume-clone")
async def resume_clone(name: str, background_tasks: BackgroundTasks):
    settings = _settings()
    state = load_state(name, settings)
    voice_id = state.get("voice_id", "")
    prompt_text = state.get("clone_prompt_text", "")
    if not voice_id:
        raise HTTPException(400, detail="未找到音色配置")
    executor = get_executor()
    executor.submit(pipeline_voice_clone, name, prompt_text, voice_id, settings, sio)
    return {"ok": True}


@router.post("/project/{name}/regenerate-clone/{idx}")
async def regenerate_clone(name: str, idx: int, data: RegenerateCloneRequest):
    settings = _settings()
    return {"ok": True, "msg": "单句重新生成功能暂未实现"}


@router.get("/project/{name}/voice-preview")
async def get_voice_preview(name: str):
    settings = _settings()
    p = project_dir(name, settings) / "voice_preview.webm"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/webm")


@router.post("/project/{name}/voice-preview")
async def generate_voice_preview(name: str, data: VoicePreviewRequest):
    settings = _settings()
    return {"ok": True, "msg": "音色预览生成功能暂未实现"}


@router.post("/custom-voices")
async def create_custom_voice(
    name: str = Form(...),
    desc: str = Form(""),
    sample: UploadFile = File(...),
):
    settings = _settings()
    import uuid
    voice_id = f"custom_{uuid.uuid4().hex[:8]}"
    d = settings.custom_voices_dir / voice_id
    d.mkdir(parents=True, exist_ok=True)
    content = await sample.read()
    (d / "sample.wav").write_bytes(content)
    (d / "meta.json").write_text(json.dumps({"name": name, "desc": desc}, ensure_ascii=False), "utf-8")
    return {"ok": True, "voice_id": voice_id}


@router.post("/custom-voices/{voice_id}/rename")
async def rename_custom_voice(voice_id: str, data: RenameVoiceRequest):
    settings = _settings()
    d = settings.custom_voices_dir / voice_id
    if not d.exists():
        raise HTTPException(404)
    meta = json.loads((d / "meta.json").read_text("utf-8")) if (d / "meta.json").exists() else {}
    meta["name"] = data.name
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    return {"ok": True}


@router.delete("/custom-voices/{voice_id}")
async def delete_custom_voice(voice_id: str):
    settings = _settings()
    d = settings.custom_voices_dir / voice_id
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@router.get("/custom-voices/{voice_id}/sample")
async def get_custom_voice_sample(voice_id: str):
    settings = _settings()
    p = settings.custom_voices_dir / voice_id / "sample.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


@router.get("/custom-voices/{voice_id}/preview")
async def get_custom_voice_preview(voice_id: str):
    settings = _settings()
    p = settings.custom_voices_dir / voice_id / "preview.webm"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/webm")


@router.post("/project/{name}/record/{idx}")
async def upload_recording(name: str, idx: int, audio: UploadFile = File(...)):
    settings = _settings()
    d = project_dir(name, settings) / "recordings"
    d.mkdir(exist_ok=True)
    content = await audio.read()
    (d / f"s_{idx:03d}.webm").write_bytes(content)
    return {"ok": True}


@router.get("/project/{name}/record/{idx}")
async def get_recording(name: str, idx: int):
    settings = _settings()
    p = project_dir(name, settings) / "recordings" / f"s_{idx:03d}.webm"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/webm")


@router.delete("/project/{name}/record/{idx}")
async def delete_recording(name: str, idx: int):
    settings = _settings()
    p = project_dir(name, settings) / "recordings" / f"s_{idx:03d}.webm"
    p.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/project/{name}/recorded")
async def get_recorded_list(name: str):
    settings = _settings()
    rec_dir = project_dir(name, settings) / "recordings"
    recorded = []
    if rec_dir.exists():
        for f in sorted(rec_dir.glob("s_*.webm")):
            if "_clone" not in f.name:
                idx = int(f.stem.split("_")[1])
                recorded.append(idx)
    return {"recorded": recorded}


@router.get("/project/{name}/cloned")
async def get_cloned_list(name: str):
    settings = _settings()
    rec_dir = project_dir(name, settings) / "recordings"
    cloned = []
    if rec_dir.exists():
        for f in sorted(rec_dir.glob("s_*_clone.webm")):
            idx = int(f.stem.split("_")[1])
            cloned.append(idx)
    return {"cloned": cloned}


@router.get("/project/{name}/selected-sources")
async def get_selected_sources(name: str):
    settings = _settings()
    state = load_state(name, settings)
    return {"selected": state.get("selected_sources", {})}


@router.post("/project/{name}/select-source/{idx}")
async def select_source(name: str, idx: int, data: SelectSourceRequest):
    settings = _settings()
    state = load_state(name, settings)
    selected = state.get("selected_sources", {}) or {}
    selected[str(idx)] = data.source or "manual"
    save_state(name, settings, sio, selected_sources=selected)
    return {"ok": True}


@router.get("/project/{name}/clone-audio/{idx}")
async def get_clone_audio(name: str, idx: int):
    settings = _settings()
    p = project_dir(name, settings) / "recordings" / f"s_{idx:03d}_clone.webm"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/webm")


@router.post("/project/{name}/accept-clone/{idx}")
async def accept_clone(name: str, idx: int):
    settings = _settings()
    d = project_dir(name, settings) / "recordings"
    clone = d / f"s_{idx:03d}_clone.webm"
    manual = d / f"s_{idx:03d}.webm"
    if clone.exists():
        import shutil
        shutil.copy2(clone, manual)
    state = load_state(name, settings)
    selected = state.get("selected_sources", {}) or {}
    selected[str(idx)] = "clone"
    save_state(name, settings, sio, selected_sources=selected)
    return {"ok": True}


@router.post("/project/{name}/reject-clone/{idx}")
async def reject_clone(name: str, idx: int):
    settings = _settings()
    d = project_dir(name, settings) / "recordings"
    (d / f"s_{idx:03d}_clone.webm").unlink(missing_ok=True)
    state = load_state(name, settings)
    selected = state.get("selected_sources", {}) or {}
    selected.pop(str(idx), None)
    save_state(name, settings, sio, selected_sources=selected)
    return {"ok": True}


@router.post("/project/{name}/accept-all-clones")
async def accept_all_clones(name: str):
    settings = _settings()
    d = project_dir(name, settings) / "recordings"
    if not d.exists():
        return {"ok": True, "count": 0}
    count = 0
    for clone in d.glob("s_*_clone.webm"):
        idx = int(clone.stem.split("_")[1])
        manual = d / f"s_{idx:03d}.webm"
        import shutil
        shutil.copy2(clone, manual)
        count += 1
    state = load_state(name, settings)
    selected = state.get("selected_sources", {}) or {}
    for clone in d.glob("s_*_clone.webm"):
        idx = int(clone.stem.split("_")[1])
        selected[str(idx)] = "clone"
    save_state(name, settings, sio, selected_sources=selected)
    return {"ok": True, "count": count}


@router.get("/project/{name}/clone-preview-all")
async def clone_preview_all(name: str):
    settings = _settings()
    return {"ok": True, "msg": "批量预览功能暂未实现"}
