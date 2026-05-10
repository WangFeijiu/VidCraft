"""Pre-generate all voice library preview audio files."""
import sys, os
from pathlib import Path

if sys.platform == "win32":
    vs = Path(r"E:/视频处理/.venv/Lib/site-packages")
    for s in ("cublas", "cudnn", "cuda_nvrtc"):
        d = vs / "nvidia" / s / "bin"
        if d.is_dir(): os.add_dll_directory(str(d))
    sys.path.insert(0, str(vs))

COSYVOICE_DIR = Path(r"E:/视频处理/CosyVoice")
CACHE_DIR = Path(r"E:/视频处理/voice_studio/voice_cache")
CACHE_DIR.mkdir(exist_ok=True)
FFMPEG = r"D:/Tech/program/python/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
import subprocess

sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# Use the SAME reference audio for all (the model's base voice is fixed by ref)
# Instruction text controls style/prosody only
REF_WAV = str(COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav")
PREVIEW_TEXT = "大家好，欢迎收听本期节目，今天我们将一起探讨一个有趣的话题。"

# Style-focused instructions (not gender claims)
VOICE_LIBRARY = [
    {"id": "standard",   "instruct": "用标准播音的语气朗读，声音清晰、平稳、专业。"},
    {"id": "deep",       "instruct": "用低沉浑厚的语气朗读，声音沉稳、有力、充满磁性，语速稍慢。"},
    {"id": "humor",      "instruct": "用轻松幽默的语气朗读，声音活泼、有趣，带有调侃的感觉，语速稍快。"},
    {"id": "narrative",  "instruct": "用纪录片旁白的语气朗读，声音平和、客观，娓娓道来，像在讲述一段故事。"},
    {"id": "warm",       "instruct": "用温柔知性的语气朗读，声音温暖、舒缓，像知心朋友在轻声细语。"},
    {"id": "lively",     "instruct": "用活泼俏皮的语气朗读，声音轻快、可爱，充满青春活力，语速较快。"},
    {"id": "serious",    "instruct": "用严肃正式的语气朗读，字正腔圆，像新闻联播的播音员。"},
    {"id": "emotional",  "instruct": "用富有感情的语气朗读，声音中带有情感起伏，时而激昂时而舒缓。"},
]

print("Loading CosyVoice model...")
model_dir = None
for name in ["Fun-CosyVoice3-0.5B", "CosyVoice2-0.5B", "CosyVoice-300M"]:
    p = COSYVOICE_DIR / "pretrained_models" / name
    if p.exists():
        model_dir = str(p)
        print(f"Using model: {name}")
        break

if not model_dir:
    print("ERROR: No CosyVoice model found!")
    sys.exit(1)

model = AutoModel(model_dir=model_dir)

for voice in VOICE_LIBRARY:
    cache_webm = CACHE_DIR / f"{voice['id']}.webm"
    print(f"[gen]  {voice['id']} ...", end=" ", flush=True)

    for j, result in enumerate(model.inference_instruct2(
            PREVIEW_TEXT, voice["instruct"], REF_WAV, stream=False)):
        wav_out = CACHE_DIR / f"{voice['id']}.wav"
        torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
        subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                        "-c:a", "libopus", "-b:a", "64k", str(cache_webm)],
                       check=True, capture_output=True)
        wav_out.unlink(missing_ok=True)

    size = cache_webm.stat().st_size
    print(f"done ({size // 1024} KB)")

print("\nAll voice previews generated!")
for f in sorted(CACHE_DIR.glob("*.webm")):
    if f.name.startswith('_'): continue
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
