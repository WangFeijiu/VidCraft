"""Pydantic models for sentence/subtitle endpoints."""
from pydantic import BaseModel


class Sentence(BaseModel):
    text: str
    start: float
    end: float
    source: list[int] | None = None


class SentencesSave(BaseModel):
    version: str = "original"
    sentences: list[Sentence]
    clear_after: bool = False


class OptimizeRequest(BaseModel):
    version: str = "original"
    description: str = ""


class MatchSubtitlesRequest(BaseModel):
    subtitles: list[str]
