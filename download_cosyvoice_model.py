from modelscope import snapshot_download
import time, sys

LOCAL = r"E:/视频处理/CosyVoice/pretrained_models/CosyVoice2-0.5B"
last = None
for i in range(1, 11):
    try:
        path = snapshot_download("iic/CosyVoice2-0.5B", local_dir=LOCAL)
        print(f"Done -> {path}", flush=True)
        sys.exit(0)
    except Exception as e:
        last = e
        print(f"[retry {i}] {type(e).__name__}: {e}", flush=True)
        time.sleep(3)
raise last
