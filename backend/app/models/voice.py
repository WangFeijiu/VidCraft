"""Pydantic models for voice-related endpoints."""
from pydantic import BaseModel


class VoicePreset(BaseModel):
    id: str
    name: str
    desc: str
    instruct: str


class VoicePreviewRequest(BaseModel):
    voice_id: str
    text: str = "这是一段音色预览，你可以听一下这个声音的效果。"


class RegenerateCloneRequest(BaseModel):
    prompt_text: str = ""


class SelectSourceRequest(BaseModel):
    source: str = ""


class RenameVoiceRequest(BaseModel):
    name: str
