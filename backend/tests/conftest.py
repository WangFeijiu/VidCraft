"""Pytest fixtures for backend tests."""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir():
    d = Path(tempfile.mkdtemp(prefix="vs_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_settings(temp_data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(temp_data_dir))
    monkeypatch.setenv("FFMPEG_PATH", "ffmpeg")
    monkeypatch.setenv("COSYVOICE_DIR", str(temp_data_dir / "CosyVoice"))
    monkeypatch.setenv("HF_CACHE_DIR", str(temp_data_dir / "hf_cache"))

    from app.dependencies import get_settings
    get_settings.cache_clear()
    from app.config import Settings
    settings = Settings()
    settings.ensure_dirs()
    return settings


@pytest.fixture
def client(mock_settings, monkeypatch):
    def fake_transcribe(*args, **kwargs):
        pass

    monkeypatch.setattr("app.services.transcribe.pipeline_transcribe", fake_transcribe)

    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_video_bytes() -> bytes:
    return b"FAKE_VIDEO_DATA_FOR_TESTING"
