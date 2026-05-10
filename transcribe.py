import os, sys
if sys.platform == 'win32':
    for sub in ['cublas', 'cudnn', 'cuda_nvrtc']:
        d = rf"E:/视频处理/.venv/Lib/site-packages/nvidia/{sub}/bin"
        if os.path.isdir(d):
            os.add_dll_directory(d)
os.environ['HF_HOME'] = r'E:/视频处理/.cache/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
import time

CACHE_DIR = r"E:/视频处理/.cache/huggingface"
REPO_ID = "Systran/faster-whisper-large-v3"

print("[0/3] Ensuring model files (resume on flaky networks) ...", flush=True)
last_err = None
for attempt in range(1, 11):
    try:
        snapshot_download(repo_id=REPO_ID, cache_dir=CACHE_DIR, max_workers=2)
        print("  model files complete.", flush=True)
        break
    except Exception as e:
        last_err = e
        print(f"  [retry {attempt}] {type(e).__name__}: {e}", flush=True)
        time.sleep(3)
else:
    raise last_err

def fmt_ts(seconds):
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    ms = int(round((seconds - total) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

print("[1/3] Loading model: large-v3 ...", flush=True)
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    download_root=r"E:/视频处理/.cache/huggingface",
)

audio_path = r"E:/视频处理/flowbrain_audio_16k.wav"
print(f"[2/3] Transcribing: {audio_path}", flush=True)
segments, info = model.transcribe(
    audio_path,
    language="zh",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=400),
    word_timestamps=False,
)
print(f"  language={info.language} prob={info.language_probability:.2f} duration={info.duration:.1f}s", flush=True)

srt_path = r"E:/视频处理/transcript.srt"
txt_path = r"E:/视频处理/transcript.txt"
with open(srt_path, "w", encoding="utf-8") as f_srt, open(txt_path, "w", encoding="utf-8") as f_txt:
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        f_srt.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n")
        f_txt.write(f"{text}\n")
        print(f"  [{seg.start:6.1f}-{seg.end:6.1f}] {text}", flush=True)

print(f"[3/3] Done. SRT={srt_path}  TXT={txt_path}", flush=True)
