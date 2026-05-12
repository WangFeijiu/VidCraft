"""Voice Studio Backend — Shared dependencies and singletons."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.config import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vs_worker")
    return _executor


_cosyvoice_model = None
_cosyvoice_model_name: str | None = None


def get_cosyvoice_model():
    global _cosyvoice_model, _cosyvoice_model_name
    if _cosyvoice_model is not None:
        return _cosyvoice_model, _cosyvoice_model_name

    from app.utils.gpu import load_cosyvoice_model
    settings = get_settings()
    _cosyvoice_model, _cosyvoice_model_name = load_cosyvoice_model(settings)
    return _cosyvoice_model, _cosyvoice_model_name


_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    from faster_whisper import WhisperModel
    settings = get_settings()
    _whisper_model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
        download_root=str(settings.HF_CACHE_DIR),
    )
    return _whisper_model
