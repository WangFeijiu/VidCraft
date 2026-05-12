"""LLM configuration routes."""
from fastapi import APIRouter, HTTPException

from app.config import Settings
from app.dependencies import get_settings
from app.services.llm import load_config, save_config, mask_config, call_llm, get_active_config

router = APIRouter(prefix="/api", tags=["llm"])


def _settings() -> Settings:
    return get_settings()


@router.get("/llm-config")
async def get_llm_config():
    settings = _settings()
    cfg = load_config(settings)
    return mask_config(cfg)


@router.put("/llm-config")
async def update_llm_config(data: dict):
    settings = _settings()
    save_config(settings, data)
    return {"ok": True}


@router.post("/llm-test")
async def test_llm(data: dict):
    settings = _settings()
    cfg_dict = data.get("config")
    if not cfg_dict:
        raise HTTPException(400, detail="缺少配置")
    from app.services.llm import resolve_key
    cfg_dict = resolve_key(settings, cfg_dict)
    try:
        result = call_llm(settings, "你好，请回复一句话。", config=cfg_dict)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/narration-styles")
async def get_narration_styles():
    return {
        "styles": [
            {"id": "default", "name": "默认", "desc": "平实叙述"},
            {"id": "documentary", "name": "纪录片", "desc": "娓娓道来"},
            {"id": "storytelling", "name": "故事讲述", "desc": "引人入胜"},
            {"id": "educational", "name": "教学", "desc": "清晰易懂"},
        ]
    }
