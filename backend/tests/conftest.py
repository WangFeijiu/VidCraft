"""Pytest fixtures for backend tests."""
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path, monkeypatch):
    """Ensure every test uses an isolated temp data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("PROJECTS_DIR", str(data_dir / "projects"))
    monkeypatch.setenv("IMG2VID_DIR", str(data_dir / "img2vid"))
    monkeypatch.setenv("TOOL_WORKSPACE_DIR", str(data_dir / "tool_workspace"))
    monkeypatch.setenv("VOICE_CACHE_DIR", str(data_dir / "voice_cache"))
    monkeypatch.setenv("CUSTOM_VOICES_DIR", str(data_dir / "custom_voices"))
    monkeypatch.setenv("FFMPEG_PATH", "ffmpeg")
    monkeypatch.setenv("COSYVOICE_DIR", str(data_dir / "CosyVoice"))
    monkeypatch.setenv("HF_CACHE_DIR", str(data_dir / "hf_cache"))
    monkeypatch.setenv("FRONTEND_DIR", str(data_dir))

    from app.dependencies import get_settings
    get_settings.cache_clear()

    from app.config import Settings
    settings = Settings()
    settings.ensure_dirs()

    yield settings

    get_settings.cache_clear()


@pytest.fixture
def client(isolate_settings, monkeypatch):
    def fake_transcribe(*args, **kwargs):
        pass

    monkeypatch.setattr("app.services.transcribe.pipeline_transcribe", fake_transcribe)

    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_video_bytes() -> bytes:
    return b"FAKE_VIDEO_DATA_FOR_TESTING"
