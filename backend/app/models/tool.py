"""Pydantic models for video tool endpoints."""
from pydantic import BaseModel


class ToolDeleteRequest(BaseModel):
    ranges: str


class ToolConvertRequest(BaseModel):
    format: str = "mp4"
    resolution: str = "original"


class ToolSpeedupRequest(BaseModel):
    start: float
    end: float
    rate: float = 2.0


class ConvertRequest(BaseModel):
    format: str = "mp4"
    resolution: str = "original"
