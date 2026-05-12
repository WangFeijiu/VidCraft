"""Voice cloning service — CosyVoice orchestration."""
import shutil
import subprocess
import traceback
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.utils.paths import project_dir, active_sentences_path
from app.utils.state import save_state

import json

VOICE_LIBRARY = [
    {"id": "standard", "name": "标准播音", "desc": "清晰平稳的专业播报",
     "instruct": "用标准播音的语气朗读，声音清晰、平稳、专业。"},
    {"id": "deep", "name": "低沉磁性", "desc": "沉稳有力的浑厚风格",
     "instruct": "用低沉浑厚的语气朗读，声音沉稳、有力、充满磁性，语速稍慢。"},
    {"id": "humor", "name": "幽默风趣", "desc": "轻松有趣的讲述风格",
     "instruct": "用轻松幽默的语气朗读，声音活泼、有趣，带有调侃的感觉，语速稍快。"},
    {"id": "narrative", "name": "纪录片旁白", "desc": "娓娓道来的叙事风格",
     "instruct": "用纪录片旁白的语气朗读，声音平和、客观，娓娓道来，像在讲述一段故事。"},
    {"id": "warm", "name": "温柔知性", "desc": "温暖舒缓的叙事风格",
     "instruct": "用温柔知性的语气朗读，声音温暖、舒缓，像知心朋友在轻声细语。"},
    {"id": "lively", "name": "活泼俏皮", "desc": "轻快可爱的青春风格",
     "instruct": "用活泼俏皮的语气朗读，声音轻快、可爱，充满青春活力，语速较快。"},
    {"id": "serious", "name": "严肃正式", "desc": "字正腔圆的新闻风格",
     "instruct": "用严肃正式的语气朗读，字正腔圆，像新闻联播的播音员。"},
    {"id": "emotional", "name": "情感朗读", "desc": "富有感情的起伏风格",
     "instruct": "用富有感情的语气朗读，声音中带有情感起伏，时而激昂时而舒缓。"},
]

_cancel_clone: set = set()


def clear_stale_clones(name: str, settings: Settings) -> None:
    rec_dir = project_dir(name, settings) / "recordings"
    if not rec_dir.exists():
        return
    for f in rec_dir.glob("s_*_clone.webm"):
        f.unlink(missing_ok=True)
    for f in rec_dir.glob("s_*_clone.wav"):
        f.unlink(missing_ok=True)
    from app.utils.state import load_state
    state = load_state(name, settings)
    selected = state.get("selected_sources", {}) or {}
    kept = {}
    for k, v in selected.items():
        if v == "manual":
            kept[k] = v
        else:
            (rec_dir / f"s_{int(k):03d}.webm").unlink(missing_ok=True)
    save_state(name, settings, selected_sources=kept)


def transcribe_sample(wav_path: Path, settings: Settings) -> str:
    try:
        from app.dependencies import get_whisper_model
        model = get_whisper_model()
        segs_gen, info = model.transcribe(
            str(wav_path), language="zh", beam_size=5,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        )
        texts = [s.text.strip() for s in segs_gen if s.text.strip()]
        return " ".join(texts)
    except Exception:
        return ""


def pipeline_voice_clone(name: str, prompt_text: str, voice_id: str,
                         settings: Settings, sio) -> None:
    d = project_dir(name, settings)
    ffmpeg = settings.FFMPEG_PATH
    try:
        import torchaudio
        from app.dependencies import get_cosyvoice_model
        model, model_name = get_cosyvoice_model()
        sentences = json.loads(active_sentences_path(name, settings).read_text("utf-8"))
        total = len(sentences)

        rec_dir = d / "recordings"
        done = sum(1 for i in range(total) if (rec_dir / f"s_{i + 1:03d}_clone.webm").exists())
        if done >= total:
            save_state(name, settings, sio, stage="recording", msg="音色克隆完成，可试听或直接生成视频")
            return
        save_state(name, settings, sio, stage="cloning",
                   msg=f"音色克隆中 {done}/{total}..." + ("（恢复中）" if done > 0 else ""),
                   clone_progress=[done, total])

        voice = next((v for v in VOICE_LIBRARY if v["id"] == voice_id), None)
        is_custom = voice_id.startswith("custom_") and (settings.custom_voices_dir / voice_id / "sample.wav").exists()
        use_instruct = voice is not None and "CosyVoice" in model_name and "300M" not in model_name

        if not use_instruct:
            sample_wav = d / "voice_sample.wav"
            if sample_wav.exists():
                transcribed = transcribe_sample(sample_wav, settings)
                if transcribed:
                    prompt_text = transcribed
                    save_state(name, settings, clone_prompt_text=prompt_text)

        cosyvoice_dir = Path(settings.COSYVOICE_DIR)

        if use_instruct:
            ref_wav = str(cosyvoice_dir / "asset" / voice.get("ref", "zero_shot_prompt.wav"))
            for i, seg in enumerate(sentences):
                if name in _cancel_clone:
                    _cancel_clone.discard(name)
                    return
                if (rec_dir / f"s_{i + 1:03d}_clone.webm").exists():
                    continue
                for j, result in enumerate(model.inference_instruct2(
                        seg["text"], voice["instruct"], ref_wav, stream=False)):
                    wav_out = d / "recordings" / f"s_{i + 1:03d}_clone.wav"
                    torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                    webm_out = d / "recordings" / f"s_{i + 1:03d}_clone.webm"
                    subprocess.run([ffmpeg, "-y", "-i", str(wav_out),
                                    "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                                   check=True, capture_output=True)
                    wav_out.unlink(missing_ok=True)
                save_state(name, settings, sio, stage="cloning",
                           msg=f"音色克隆中 {i + 1}/{total}...",
                           clone_progress=[i + 1, total])
        else:
            if "CosyVoice3" in model_name:
                full_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
            else:
                full_prompt = prompt_text
            sample_path = str(d / "voice_sample.wav")
            for i, seg in enumerate(sentences):
                if name in _cancel_clone:
                    _cancel_clone.discard(name)
                    return
                if (rec_dir / f"s_{i + 1:03d}_clone.webm").exists():
                    continue
                for j, result in enumerate(model.inference_zero_shot(
                        seg["text"], full_prompt, sample_path, stream=False)):
                    wav_out = d / "recordings" / f"s_{i + 1:03d}_clone.wav"
                    torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                    webm_out = d / "recordings" / f"s_{i + 1:03d}_clone.webm"
                    subprocess.run([ffmpeg, "-y", "-i", str(wav_out),
                                    "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                                   check=True, capture_output=True)
                    wav_out.unlink(missing_ok=True)
                save_state(name, settings, sio, stage="cloning",
                           msg=f"音色克隆中 {i + 1}/{total}...",
                           clone_progress=[i + 1, total])

        save_state(name, settings, sio, stage="recording", msg="音色克隆完成，可试听或直接生成视频")
    except Exception as e:
        save_state(name, settings, sio, stage="error",
                   msg=f"音色克隆失败: {e}\n{traceback.format_exc()}")
