"""Pydantic models for image-to-video endpoints."""
from pydantic import BaseModel


class Img2VidAnalyzeRequest(BaseModel):
    style: str = ""


class Img2VidGenerateRequest(BaseModel):
    animate: bool = True


class Img2VidStageUpdate(BaseModel):
    stage: str


class Img2VidThemeUpdate(BaseModel):
    theme: str = ""


class NarrationItem(BaseModel):
    image_idx: int
    narration: str
    analysis: str = ""
