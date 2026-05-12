"""Common response models."""
from pydantic import BaseModel


class OkResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


class OkCountResponse(BaseModel):
    ok: bool = True
    count: int = 0
