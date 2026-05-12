"""Pydantic models for project-related endpoints."""
from pydantic import BaseModel


class ProjectStatus(BaseModel):
    name: str
    stage: str = "new"
    msg: str = ""
    recorded: int = 0
    duration: float | None = None
    clone_progress: list[int] | None = None
    voice_id: str = ""
    recording_version: str = ""


class StageUpdate(BaseModel):
    stage: str
    version: str = ""


class SubtitleStyle(BaseModel):
    font_name: str = "Microsoft YaHei"
    font_size: int = 20
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: int = 1
    position: str = "bottom"
    margin_v: int = 30
