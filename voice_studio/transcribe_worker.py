"""Whisper transcription worker - runs as a separate process to avoid blocking Flask."""
import sys, json
from pathlib import Path

def main():
    name = sys.argv[1]
    project_dir = Path(sys.argv[2])
    hf_cache = sys.argv[3]
    audio_path = project_dir / "audio_16k.wav"
    state_path = project_dir / "state.json"

    def update_state(**kw):
        s = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {}
        s.update(kw)
        state_path.write_text(json.dumps(s, ensure_ascii=False), "utf-8")

    try:
        dur_file = project_dir / "_duration.txt"
        dur = float(dur_file.read_text()) if dur_file.exists() else 600

        update_state(stage="processing", msg="语音转写中...", duration=dur, transcribe_progress=[0, int(dur)])

        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                             download_root=hf_cache)
        segs_gen, info = model.transcribe(
            str(audio_path), language="zh",
            beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500))

        sentences = []
        for s in segs_gen:
            if s.text.strip():
                sentences.append({"text": s.text.strip(), "start": round(s.start, 3), "end": round(s.end, 3)})
                if len(sentences) % 3 == 0:
                    update_state(stage="processing", msg="语音转写中...",
                                 duration=dur, transcribe_progress=[int(s.end), int(dur)])

        (project_dir / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=2), "utf-8")
        update_state(stage="editing", msg="转写完成，请编辑字幕", duration=dur)
    except Exception as e:
        import traceback
        update_state(stage="error", msg=f"{e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
