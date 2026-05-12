"""GPU utilities — CUDA DLL setup and model loading."""
import sys
from pathlib import Path

from loguru import logger


def setup_cuda_dlls(settings) -> None:
    import os
    venv_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    for subdir in ("cublas", "cudnn", "cuda_nvrtc"):
        dll_dir = venv_packages / "nvidia" / subdir / "bin"
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
            logger.debug(f"Added CUDA DLL dir: {dll_dir}")


def load_cosyvoice_model(settings):
    cosyvoice_dir = Path(settings.COSYVOICE_DIR)
    sys.path.insert(0, str(cosyvoice_dir))
    sys.path.insert(0, str(cosyvoice_dir / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir, model_name = _find_cosyvoice_model(cosyvoice_dir)
    if not model_dir:
        raise FileNotFoundError("CosyVoice model not found")

    logger.info(f"Loading CosyVoice model: {model_name}")
    model = AutoModel(model_dir=model_dir)

    _warmup_model(model, model_name, cosyvoice_dir)

    return model, model_name


def _find_cosyvoice_model(cosyvoice_dir: Path) -> tuple[str | None, str | None]:
    for name in ("Fun-CosyVoice3-0.5B", "CosyVoice2-0.5B", "CosyVoice-300M"):
        path = cosyvoice_dir / "pretrained_models" / name
        if path.exists():
            return str(path), name
    return None, None


def _warmup_model(model, model_name: str, cosyvoice_dir: Path) -> None:
    try:
        warm_ref = cosyvoice_dir / "asset" / "zero_shot_prompt.wav"
        if not warm_ref.exists():
            return
        warm_prompt = "希望你以后能够做的比我还好呦。"
        warm_text = "收到好友从远方寄来的生日礼物。"
        if "CosyVoice3" in model_name:
            full_prompt = "You are a helpful assistant.<|endofprompt|>" + warm_prompt
        else:
            full_prompt = warm_prompt
        for _ in model.inference_zero_shot(warm_text, full_prompt, str(warm_ref), stream=False):
            pass
        logger.info("CosyVoice warmup complete")
    except Exception:
        pass
