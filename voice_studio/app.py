"""Voice Studio — upload a video, edit transcript, record voice-over, produce final video."""
import os, sys, json, re, base64, shutil, tempfile, threading, subprocess, traceback, uuid
from pathlib import Path

# Tracks which projects have been cancelled by user
_cancel_clone = set()

# ── Windows CUDA DLLs ──────────────────────────────────────────────
if sys.platform == "win32":
    _vs = Path(r"E:/视频处理/.venv/Lib/site-packages")
    for _s in ("cublas", "cudnn", "cuda_nvrtc"):
        _d = _vs / "nvidia" / _s / "bin"
        if _d.is_dir():
            os.add_dll_directory(str(_d))
    sys.path.insert(0, str(_vs))

from flask import Flask, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
from llm import load_config, save_config, mask_config, resolve_key, call_llm, call_llm_vision

# ── Paths ───────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.resolve()
PROJECTS     = BASE / "projects"
PROJECTS.mkdir(exist_ok=True)
FFMPEG       = (r"D:/Tech/program/python/Lib/site-packages"
                r"/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")
HF_CACHE     = r"E:/视频处理/.cache/huggingface"
COSYVOICE_DIR = Path(r"E:/视频处理/CosyVoice")
IMG2VID    = BASE / "img2vid"
IMG2VID.mkdir(exist_ok=True)

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

# Global cache for voice previews
VOICE_CACHE = BASE / "voice_cache"
VOICE_CACHE.mkdir(exist_ok=True)

# Custom voices saved by user
CUSTOM_VOICES_DIR = BASE / "custom_voices"
CUSTOM_VOICES_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=600, ping_interval=120)

# ── Whisper cache for transcribing voice samples ─────────────────────
_whisper_model = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16",
                                      download_root=HF_CACHE)
    return _whisper_model

def _transcribe_sample(wav_path):
    """Transcribe a voice sample to use as prompt_text for zero-shot cloning."""
    try:
        model = _get_whisper_model()
        segs_gen, info = model.transcribe(str(wav_path), language="zh",
                                          beam_size=5, vad_filter=True,
                                          vad_parameters=dict(min_silence_duration_ms=500))
        texts = [s.text.strip() for s in segs_gen if s.text.strip()]
        return " ".join(texts)
    except Exception:
        return ""

# ── Helpers ─────────────────────────────────────────────────────────
def pd(name):           return PROJECTS / name
def _input_video(name):
    """Find the input video file in a project (any supported extension)."""
    d = pd(name)
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
        p = d / f"input{ext}"
        if p.exists():
            return p
    # Fallback: find any file starting with "input"
    for f in d.glob("input.*"):
        if f.suffix in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
            return f
    return d / "input.mp4"
def _active_sentences_path(name):
    """Resolve the active sentences file based on recording_version in state."""
    d = pd(name)
    state = load_state(name)
    v = state.get("recording_version", "")
    if v == "uploaded" and (d / "sentences_uploaded.json").exists():
        return d / "sentences_uploaded.json"
    if v == "optimized" and (d / "sentences_optimized.json").exists():
        return d / "sentences_optimized.json"
    if v == "original" and (d / "sentences.json").exists():
        return d / "sentences.json"
    # Fallback to priority: uploaded > optimized > original
    if (d / "sentences_uploaded.json").exists():
        return d / "sentences_uploaded.json"
    if (d / "sentences_optimized.json").exists():
        return d / "sentences_optimized.json"
    return d / "sentences.json"

def load_state(name):
    p = pd(name) / "state.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {"stage": "new"}
def save_state(name, **kw):
    d = pd(name)
    if not d.exists():
        return
    s = load_state(name); s.update(kw)
    (d / "state.json").write_text(json.dumps(s, ensure_ascii=False), "utf-8")
    try:
        socketio.emit('project_update', {'project': name, 'state': s}, namespace='/')
    except Exception:
        pass
def ts(s):
    h, m = int(s // 3600), int(s % 3600 // 60); sec = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    if ms == 1000: sec += 1; ms = 0
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

# ── Routes: pages ───────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file("templates/index.html")

# ── Routes: projects ────────────────────────────────────────────────
@app.route("/api/projects")
def api_list():
    out = []
    for d in sorted(PROJECTS.iterdir()):
        if d.is_dir() and (d / "state.json").exists():
            out.append({"name": d.name, **load_state(d.name)})
    return jsonify(out)

@app.route("/api/projects", methods=["POST"])
def api_create():
    name = request.form.get("name", "").strip()
    if not name: return jsonify({"error": "项目名必填"}), 400
    f = request.files.get("video")
    if not f: return jsonify({"error": "视频必传"}), 400
    d = PROJECTS / name
    if d.exists(): return jsonify({"error": "项目已存在"}), 400
    d.mkdir(parents=True); (d / "recordings").mkdir()
    # Save as-is, keep original format
    orig_name = f.filename or "video.mp4"
    ext = Path(orig_name).suffix or ".mp4"
    f.save(str(d / f"input{ext}"))
    save_state(name, stage="processing", msg="提取音频...", orig_ext=ext)
    threading.Thread(target=_pipeline_transcribe, args=(name,), daemon=True).start()
    return jsonify({"name": name})

@app.route("/api/project/<name>", methods=["DELETE"])
def api_delete_project(name):
    d = PROJECTS / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return jsonify({"ok": True})

@app.route("/api/project/<name>/reset", methods=["POST"])
def api_reset_project(name):
    d = pd(name)
    # Delete everything in the project directory
    for f in d.iterdir():
        if f.name == "state.json":
            continue
        if f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
        else:
            f.unlink(missing_ok=True)
    # Recreate recordings dir
    (d / "recordings").mkdir(exist_ok=True)
    # Reset state
    save_state(name, stage="editing", msg="数据已清空，请上传视频")
    return jsonify({"ok": True})

@app.route("/api/project/<name>")
def api_status(name):
    s = load_state(name)
    s["recorded"] = len([f for f in (pd(name) / "recordings").glob("s_*.webm") if "_clone" not in f.name])
    return jsonify(s)

@app.route("/api/project/<name>/sentences")
def api_get_sents(name):
    """Get sentences with version info. Returns {active: 'original'|'optimized'|'uploaded', versions: {original: [...], ...}, sentences: [...]}"""
    d = pd(name)
    versions = {}
    # Build in display-priority order: uploaded > optimized > original
    if (d / "sentences_uploaded.json").exists():
        versions['uploaded'] = json.loads((d / "sentences_uploaded.json").read_text("utf-8"))
    if (d / "sentences_optimized.json").exists():
        versions['optimized'] = json.loads((d / "sentences_optimized.json").read_text("utf-8"))
    if (d / "sentences.json").exists():
        versions['original'] = json.loads((d / "sentences.json").read_text("utf-8"))

    # Priority: uploaded > optimized > original (editing stage)
    # In recording stage, use the locked recording_version if set
    state = load_state(name)
    if state.get("stage") == "recording" and state.get("recording_version") in versions:
        active = state["recording_version"]
    else:
        active = 'uploaded' if 'uploaded' in versions else ('optimized' if 'optimized' in versions else 'original')
    sentences = versions.get(active, [])

    return jsonify({'active': active, 'versions': versions, 'sentences': sentences})

@app.route("/api/project/<name>/sentences", methods=["PUT"])
def api_put_sents(name):
    """Save sentences. Accepts {version: 'original'|'optimized'|'uploaded', sentences: [...], clear_after: bool}"""
    data = request.json
    version = data.get('version', 'original')
    sentences = data.get('sentences', [])
    clear_after = data.get('clear_after', False)

    # Check if we need to clear recordings due to content changes
    filename = f"sentences_{version}.json" if version != 'original' else "sentences.json"
    filepath = pd(name) / filename

    changed_indices = []
    if filepath.exists() and not clear_after:
        # Load old sentences and compare
        old_sentences = json.loads(filepath.read_text("utf-8"))
        for i, (old, new) in enumerate(zip(old_sentences, sentences)):
            if old.get('text') != new.get('text'):
                changed_indices.append(i)
        # Check if length changed
        if len(old_sentences) != len(sentences):
            # If length changed, clear all recordings after the shorter length
            min_len = min(len(old_sentences), len(sentences))
            changed_indices.extend(range(min_len, max(len(old_sentences), len(sentences))))

    # Save to appropriate file
    filepath.write_text(json.dumps(sentences, ensure_ascii=False, indent=2), "utf-8")

    # Clear affected recordings
    if changed_indices:
        rec_dir = pd(name) / "recordings"
        if rec_dir.exists():
            for idx in changed_indices:
                # Remove recordings for changed sentences
                for pattern in [f"audio_{idx}.*", f"cloned_{idx}.*"]:
                    for f in rec_dir.glob(pattern):
                        f.unlink(missing_ok=True)
                        print(f"[Edit] Cleared recording: {f.name}")

    if clear_after:
        rec_dir = pd(name) / "recordings"
        if rec_dir.exists():
            for f in rec_dir.iterdir():
                f.unlink(missing_ok=True)

    if load_state(name).get("stage") == "processing":
        save_state(name, stage="editing", msg="转写完成，请编辑字幕")

    return jsonify({"ok": True, "changed_indices": changed_indices})

@app.route("/api/project/<name>/has-recordings")
def api_has_recordings(name):
    rec_dir = pd(name) / "recordings"
    has = rec_dir.exists() and any(f.suffix in (".webm", ".wav", ".mp3") for f in rec_dir.iterdir())
    return jsonify({"has_recordings": has})

@app.route("/api/project/<name>/has-video")
def api_has_video(name):
    has = _input_video(name).exists()
    return jsonify({"has": has})

@app.route("/api/project/<name>/reupload", methods=["POST"])
def api_reupload_video(name):
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "请上传视频"}), 400
    d = pd(name)
    # Delete old video
    for old in d.glob("input.*"):
        old.unlink(missing_ok=True)
    ext = Path(f.filename).suffix or ".mp4"
    f.save(str(d / f"input{ext}"))
    save_state(name, stage="processing", msg="视频上传成功，开始转写...")
    threading.Thread(target=_pipeline_transcribe, args=(name,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/stage", methods=["PUT"])
def api_set_stage(name):
    data = request.json or {}
    stage = data.get("stage", "")
    if stage in ("editing", "recording"):
        updates = {"stage": stage, "msg": ""}
        # Lock the recording version so refreshWork uses the correct sentences
        if stage == "recording":
            version = data.get("version", "")
            if version in ("original", "optimized", "uploaded"):
                updates["recording_version"] = version
        save_state(name, **updates)
    return jsonify({"ok": True})

@app.route("/api/project/<name>/record/<int:idx>", methods=["POST"])
def api_upload_rec(name, idx):
    f = request.files.get("audio")
    if not f: return jsonify({"error": "no audio"}), 400
    f.save(str(pd(name) / "recordings" / f"s_{idx:03d}.webm"))
    return jsonify({"ok": True})

@app.route("/api/project/<name>/record/<int:idx>")
def api_get_rec(name, idx):
    p = pd(name) / "recordings" / f"s_{idx:03d}.webm"
    return send_file(str(p), mimetype="audio/webm") if p.exists() else ("", 404)

@app.route("/api/project/<name>/record/<int:idx>", methods=["DELETE"])
def api_delete_rec(name, idx):
    p = pd(name) / "recordings" / f"s_{idx:03d}.webm"
    p.unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/api/project/<name>/recorded")
def api_recorded(name):
    return jsonify([int(f.name.split("_")[1].split(".")[0])
                    for f in (pd(name) / "recordings").glob("s_*.webm")
                    if "_clone" not in f.name])

@app.route("/api/project/<name>/cloned")
def api_cloned(name):
    return jsonify([int(f.name.split("_")[1])
                    for f in (pd(name) / "recordings").glob("s_*_clone.webm")])

@app.route("/api/project/<name>/selected-sources")
def api_selected_sources(name):
    state = load_state(name)
    return jsonify(state.get("selected_sources", {}))

@app.route("/api/project/<name>/select-source/<int:idx>", methods=["POST"])
def api_select_source(name, idx):
    state = load_state(name)
    selected = state.get("selected_sources", {})
    data = request.get_json(silent=True) or {}
    src = data.get("source", "")
    if src in ("clone", "manual"):
        selected[str(idx)] = src
    elif src == "":
        selected.pop(str(idx), None)
    save_state(name, selected_sources=selected)
    return jsonify({"ok": True})

@app.route("/api/project/<name>/clone-audio/<int:idx>")
def api_get_clone(name, idx):
    p = pd(name) / "recordings" / f"s_{idx:03d}_clone.webm"
    return send_file(str(p), mimetype="audio/webm") if p.exists() else ("", 404)

@app.route("/api/project/<name>/accept-clone/<int:idx>", methods=["POST"])
def api_accept_clone(name, idx):
    src = pd(name) / "recordings" / f"s_{idx:03d}_clone.webm"
    dst = pd(name) / "recordings" / f"s_{idx:03d}.webm"
    if src.exists():
        shutil.copy2(str(src), str(dst))
        return jsonify({"ok": True})
    return jsonify({"error": "克隆音频不存在"}), 404

@app.route("/api/project/<name>/reject-clone/<int:idx>", methods=["POST"])
def api_reject_clone(name, idx):
    # Remove accepted recording if it was a clone, but keep clone file
    dst = pd(name) / "recordings" / f"s_{idx:03d}.webm"
    dst.unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/api/project/<name>/accept-all-clones", methods=["POST"])
def api_accept_all_clones(name):
    d = pd(name) / "recordings"
    state = load_state(name)
    selected = state.get("selected_sources", {})
    count = 0
    for f in d.glob("s_*_clone.webm"):
        idx_str = f.name.split("_")[1]
        dst = d / f"s_{idx_str}.webm"
        shutil.copy2(str(f), str(dst))
        selected[idx_str] = "clone"
        count += 1
    save_state(name, selected_sources=selected)
    return jsonify({"ok": True, "count": count})

@app.route("/api/project/<name>/clone-preview-all")
def api_clone_preview_all(name):
    """Concatenate all clone recordings into one audio for preview."""
    d = pd(name) / "recordings"
    out = d / "_all_clones_preview.webm"
    # Build list of clone files in order
    files = sorted(d.glob("s_*_clone.webm"))
    if not files:
        return jsonify({"error": "没有克隆录音"}), 404
    # Concatenate using ffmpeg concat demuxer
    tmpdir = tempfile.mkdtemp(prefix="vs_")
    try:
        lines = []
        for i, f in enumerate(files):
            tmp_seg = os.path.join(tmpdir, f"s{i:04d}.webm")
            shutil.copy2(str(f), tmp_seg)
            lines.append(f"file 's{i:04d}.webm'")
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        tmp_out = os.path.join(tmpdir, "out.webm")
        subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_file, "-c", "copy", tmp_out],
                       check=True, capture_output=True, encoding='utf-8', errors='ignore')
        shutil.copy2(tmp_out, str(out))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return send_file(str(out), mimetype="audio/webm")
def api_clear_clones(name):
    """Delete all clone audio files so cloning can restart with a new voice."""
    d = pd(name) / "recordings"
    count = 0
    for f in d.glob("s_*_clone.webm"):
        f.unlink(missing_ok=True)
        count += 1
    save_state(name, stage="recording", msg="", clone_progress=[0, 0], voice_id="", clone_prompt_text="")
    return jsonify({"ok": True, "deleted": count})

@app.route("/api/project/<name>/regenerate-clone/<int:idx>", methods=["POST"])
def api_regen_clone(name, idx):
    d = pd(name)
    sentences = json.loads(_active_sentences_path(name).read_text("utf-8"))
    if idx < 1 or idx > len(sentences):
        return jsonify({"error": "序号越界"}), 400
    prompt_text = request.json.get("prompt_text", "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题，希望你们能够喜欢这个内容，也希望大家能够多多支持，谢谢你们的关注和鼓励。")
    state = load_state(name)
    voice_id = state.get("voice_id", "")
    try:
        sys.path.insert(0, str(COSYVOICE_DIR))
        sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio

        model_dir, model_name = _get_cosyvoice_model()
        if not model_dir:
            return jsonify({"error": "未找到 CosyVoice 模型"}), 500
        model = AutoModel(model_dir=model_dir)

        seg = sentences[idx - 1]
        voice = next((v for v in VOICE_LIBRARY if v["id"] == voice_id), None)
        use_instruct = voice is not None and "CosyVoice" in model_name and "300M" not in model_name

        if use_instruct:
            ref_wav = str(COSYVOICE_DIR / "asset" / voice.get("ref", "zero_shot_prompt.wav"))
            for j, result in enumerate(model.inference_instruct2(
                    seg["text"], voice["instruct"], ref_wav, stream=False)):
                wav_out = d / "recordings" / f"s_{idx:03d}_clone.wav"
                torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                webm_out = d / "recordings" / f"s_{idx:03d}_clone.webm"
                subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                                "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                               check=True, capture_output=True)
                wav_out.unlink(missing_ok=True)
        else:
            if "CosyVoice3" in model_name:
                full_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
            else:
                full_prompt = prompt_text
            sample_path = str(d / "voice_sample.wav")
            for j, result in enumerate(model.inference_zero_shot(
                seg["text"], full_prompt, sample_path, stream=False)):
                wav_out = d / "recordings" / f"s_{idx:03d}_clone.wav"
                torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                webm_out = d / "recordings" / f"s_{idx:03d}_clone.webm"
                subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                                "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                               check=True, capture_output=True)
                wav_out.unlink(missing_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def api_compose(name):
    save_state(name, stage="composing", msg="拼接音频...")
    threading.Thread(target=_pipeline_compose, args=(name,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/download")
def api_download(name):
    p = pd(name) / "final.mp4"
    if p.exists():
        return send_file(str(p), mimetype="video/mp4",
                         as_attachment=True, attachment_filename=f"{name}_final.mp4")
    inp = _input_video(name)
    if inp.exists():
        mt = {"avi": "video/x-msvideo", "mkv": "video/x-matroska", "mov": "video/quicktime",
              "webm": "video/webm", "flv": "video/x-flv", "wmv": "video/x-ms-wmv"}.get(inp.suffix.lstrip("."), "video/mp4")
        return send_file(str(inp), mimetype=mt,
                         as_attachment=True, attachment_filename=inp.name)
    return ("", 404)

@app.route("/api/project/<name>/video")
def api_project_video(name):
    inp = _input_video(name)
    if inp.exists():
        mt = {"avi": "video/x-msvideo", "mkv": "video/x-matroska", "mov": "video/quicktime",
              "webm": "video/webm", "flv": "video/x-flv", "wmv": "video/x-ms-wmv"}.get(inp.suffix.lstrip("."), "video/mp4")
        return send_file(str(inp), mimetype=mt)
    return ("", 404)

@app.route("/api/project/<name>/export")
def api_export(name):
    fmt = request.args.get("format", "json")
    version = request.args.get("version", "")
    d = pd(name)
    # Use specified version, fallback to priority
    if version == 'optimized' and (d / "sentences_optimized.json").exists():
        p = d / "sentences_optimized.json"
    elif version == 'uploaded' and (d / "sentences_uploaded.json").exists():
        p = d / "sentences_uploaded.json"
    elif version == 'original':
        p = d / "sentences.json"
    elif (d / "sentences_uploaded.json").exists():
        p = d / "sentences_uploaded.json"
    elif (d / "sentences_optimized.json").exists():
        p = d / "sentences_optimized.json"
    else:
        p = d / "sentences.json"
    if not p.exists():
        return ("", 404)
    sentences = json.loads(p.read_text("utf-8"))

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(sentences, 1):
            lines.append(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text']}\n")
        content = "\n".join(lines)
        return Response(content, mimetype="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}.srt"'})

    elif fmt == "txt":
        content = "\n".join(seg["text"] for seg in sentences)
        return Response(content, mimetype="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}.txt"'})

    else:  # json
        content = json.dumps(sentences, ensure_ascii=False, indent=2)
        return Response(content, mimetype="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})

# ── Routes: LLM config ─────────────────────────────────────────────
@app.route("/api/llm-config")
def api_llm_get():
    return jsonify(load_config())

@app.route("/api/llm-config", methods=["PUT"])
def api_llm_save():
    data = request.json
    save_config(data)
    return jsonify({"ok": True})

@app.route("/api/llm-test", methods=["POST"])
def api_llm_test():
    try:
        cfg = resolve_key(request.json)
        resp = call_llm("请回复「连接成功」四个字。", config=cfg)
        return jsonify({"ok": True, "response": resp})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ── Routes: LLM optimize ───────────────────────────────────────────
@app.route("/api/project/<name>/optimize", methods=["POST"])
def api_optimize(name):
    d = pd(name)
    req_data = request.json or {}
    version = req_data.get('version', 'original')
    user_description = req_data.get('description', '').strip()
    if version == 'optimized' and (d / "sentences_optimized.json").exists():
        p = d / "sentences_optimized.json"
    elif version == 'uploaded' and (d / "sentences_uploaded.json").exists():
        p = d / "sentences_uploaded.json"
    else:
        p = d / "sentences.json"
    if not p.exists():
        return jsonify({"error": "字幕文件不存在"}), 400
    threading.Thread(target=_pipeline_optimize, args=(name, str(p), user_description), daemon=True).start()
    return jsonify({"ok": True, "async": True})

def _pipeline_optimize(name, sentences_path, user_description=''):
    try:
        sentences = json.loads(Path(sentences_path).read_text("utf-8"))
        total = len(sentences)

        # Step 0: Understand context
        socketio.emit('optimize_progress', {
            'project': name,
            'step': 0,
            'msg': '正在理解整体语境...',
            'progress': 5
        }, namespace='/')

        if user_description:
            # User provided description, use it as primary context
            context_summary = user_description
        else:
            # Auto-analyze from first 60 sentences
            all_text = ' '.join(s['text'] for s in sentences[:60])
            context_prompt = (
                f"以下是一段视频语音转写的前60句内容：\n{all_text}\n\n"
                "请用2-3句话总结这段内容的主题和语境（是教程、演讲、对话、还是其他？讲的是什么？）"
            )
            context_summary = call_llm(context_prompt, system="简洁总结，不超过100字。")

        socketio.emit('optimize_progress', {
            'project': name,
            'step': 0,
            'msg': f'语境分析完成，开始润色...',
            'progress': 10
        }, namespace='/')

        # Step 1: Text polish (batch processing, 30 sentences per batch)
        batch_size = 30
        polished = []
        for i in range(0, total, batch_size):
            batch = sentences[i:i+batch_size]

            prompt1 = (
                f"【语境】{context_summary}\n\n"
                f"以下是该视频第 {i+1} 到第 {i+len(batch)} 句语音转写（共{len(batch)}句）：\n"
                f"{json.dumps([s['text'] for s in batch], ensure_ascii=False)}\n\n"
                "请基于上述语境，逐句润色为适合配音的书面语：\n"
                "- 去掉口头禅和填充词（这个、那个、就是、然后、对、嗯、啊）\n"
                "- 修正错别字，让语句通顺\n"
                "- 断断续续的表达整合成完整句子\n"
                "- 保留所有有实际意义的内容\n"
                "- 保持原来的句数不变\n"
                "输出JSON数组，只包含润色后的文本字符串，顺序和数量必须与输入完全一致。"
            )
            try:
                resp1 = call_llm(prompt1, system="只输出纯JSON数组，不要解释。")
                cleaned1 = re.sub(r'```json\s*|\s*```', '', resp1).strip()
                texts = json.loads(cleaned1)
                if len(texts) != len(batch):
                    texts = [s['text'] for s in batch]
            except Exception as batch_err:
                raise RuntimeError(f"第 {i+1}-{i+len(batch)} 句润色失败: {batch_err}")
            for j, s in enumerate(batch):
                polished.append({"text": texts[j] or s['text'], "start": s["start"], "end": s["end"]})

            # Emit progress AFTER batch completes
            done = min(i + len(batch), total)
            socketio.emit('optimize_progress', {
                'project': name,
                'step': 1,
                'msg': '润色中...',
                'progress': 10 + int((done / total) * 70)
            }, namespace='/')

        # Step 2: Merge/split (also batched)
        merged = []
        merge_batch_size = 40
        total_polished = len(polished)
        for i in range(0, total_polished, merge_batch_size):
            batch = polished[i:i+merge_batch_size]
            socketio.emit('optimize_progress', {
                'project': name,
                'step': 2,
                'msg': '语义整合中...',
                'progress': 80 + int(((i+len(batch)) / total_polished) * 15)
            }, namespace='/')

            prompt2 = (
                f"【语境】{context_summary}\n\n"
                f"以下是第 {i+1} 到第 {i+len(batch)} 句润色后的字幕（共{len(batch)}句）：\n"
                f"{json.dumps(batch, ensure_ascii=False)}\n\n"
                "请根据语义进行适度的合并：\n"
                "- 只合并明显是同一句话被切断的碎片\n"
                "- 大部分句子保持不动\n"
                "- 过长的句子（超过50字）可以拆分\n"
                "- 合并后：start取第一句的start，end取最后一句的end\n"
                "- 拆分后：按比例分配时间段\n\n"
                "输出JSON数组，每个元素：\n"
                "- text: 文本\n"
                "- start: 浮点数\n"
                "- end: 浮点数\n"
                "- source: 数组，对应输入的句子序号（1-based，相对于本批），如 [1,2] 表示合并本批第1、2句"
            )
            try:
                resp2 = call_llm(prompt2, system="只输出纯JSON数组，不要解释。")
                cleaned2 = re.sub(r'```json\s*|\s*```', '', resp2).strip()
                batch_result = json.loads(cleaned2)
                if isinstance(batch_result, list):
                    merged.extend(batch_result)
                else:
                    merged.extend(batch)
            except Exception:
                merged.extend(batch)

        result = merged

        # Save result
        d = pd(name)
        (d / "sentences_optimized.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")

        socketio.emit('optimize_progress', {
            'project': name,
            'step': 'done',
            'msg': f'优化完成：{total} 句 → {len(result)} 句',
            'progress': 100,
            'result_count': len(result)
        }, namespace='/')
    except Exception as e:
        socketio.emit('optimize_progress', {
            'project': name,
            'step': 'error',
            'msg': str(e),
            'progress': 0
        }, namespace='/')

# ── Routes: match subtitles ─────────────────────────────────────────
@app.route("/api/project/<name>/match-subtitles", methods=["POST"])
def api_match_subtitles(name):
    user_text = (request.json or {}).get("subtitles", [])
    if not user_text:
        return jsonify({"error": "字幕内容为空"}), 400
    threading.Thread(target=_pipeline_match, args=(name, user_text), daemon=True).start()
    return jsonify({"ok": True, "async": True})

def _pipeline_match(name, user_text):
    """Sliding window semantic matching with LLM."""
    try:
        d = pd(name)
        if (d / "sentences_optimized.json").exists():
            sentences = json.loads((d / "sentences_optimized.json").read_text("utf-8"))
        else:
            sentences = json.loads((d / "sentences.json").read_text("utf-8"))

        total_user = len(user_text)
        total_orig = len(sentences)

        if total_orig == 0:
            raise ValueError("原始字幕为空")

        WINDOW_SIZE = 10
        result_map = {}  # user_idx -> {start, end, orig_start, orig_end}

        orig_cursor = 0
        user_cursor = 0

        while user_cursor < total_user:
            # Take 10 user subtitles
            user_end = min(user_cursor + WINDOW_SIZE, total_user)
            user_window = user_text[user_cursor:user_end]

            # Take 10-15 original subtitles (with buffer for splits/merges)
            orig_end = min(orig_cursor + WINDOW_SIZE + 5, total_orig)
            orig_window = sentences[orig_cursor:orig_end]

            socketio.emit('match_progress', {
                'project': name,
                'msg': f'匹配中... {user_end}/{total_user} 句',
                'progress': int((user_end / total_user) * 90)
            }, namespace='/')

            # Build prompt for LLM
            orig_lines = "\n".join(
                f"[{orig_cursor + i}] {s['text'][:50]}"
                for i, s in enumerate(orig_window)
            )
            user_lines = "\n".join(
                f"[{user_cursor + i}] {t[:50]}"
                for i, t in enumerate(user_window)
            )

            prompt = (
                f"原始字幕（序号{orig_cursor}-{orig_end-1}）：\n{orig_lines}\n\n"
                f"上传字幕（序号{user_cursor}-{user_end-1}）：\n{user_lines}\n\n"
                "请匹配每句上传字幕对应的原始字幕区间。\n"
                "输出格式：`上传序号:原始起始-原始结束`（一行一个）\n"
                "例如：\n"
                "0:0-2  （上传第0句对应原始0-2句，合并）\n"
                "1:3-3  （上传第1句对应原始第3句）\n"
                "2:4-4  （上传第2句对应原始第4句，拆分1/2）\n"
                "3:4-4  （上传第3句对应原始第4句，拆分2/2）\n\n"
                "规则：\n"
                "1. 序号使用全局序号（不是窗口内序号）\n"
                "2. 如果原始某句被拆分，多个上传句可以引用同一原始句\n"
                "3. 只输出匹配结果，不要解释"
            )

            resp = call_llm(prompt, system="你是字幕匹配专家，只输出匹配结果。", max_tokens=4000)

            # Parse results
            max_orig_matched = orig_cursor - 1
            for line in resp.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                try:
                    user_idx_str, range_str = line.split(":", 1)
                    user_idx = int(user_idx_str.strip())
                    orig_s, orig_e = range_str.strip().split("-", 1)
                    orig_start = int(orig_s.strip())
                    orig_end = int(orig_e.strip())

                    if 0 <= orig_start <= orig_end < total_orig and 0 <= user_idx < total_user:
                        result_map[user_idx] = {
                            "start": sentences[orig_start]["start"],
                            "end": sentences[orig_end]["end"],
                            "orig_start": orig_start,
                            "orig_end": orig_end,
                        }
                        max_orig_matched = max(max_orig_matched, orig_end)
                except Exception as e:
                    print(f"Parse error on line '{line}': {e}")
                    continue

            # Move cursors
            orig_cursor = max_orig_matched + 1
            user_cursor = user_end

        socketio.emit('match_progress', {
            'project': name,
            'msg': '整合结果中...',
            'progress': 95
        }, namespace='/')

        # Build final result
        final = []
        for i in range(total_user):
            if i in result_map:
                r = result_map[i]
                final.append({
                    "text": user_text[i],
                    "start": r["start"],
                    "end": r["end"],
                    "source": list(range(r["orig_start"]+1, r["orig_end"]+2))  # 1-based
                })
            else:
                # Fallback: interpolate
                prev_end = 0.0
                next_start = sentences[-1]["end"] if sentences else 0.0
                for j in range(i - 1, -1, -1):
                    if j in result_map:
                        prev_end = result_map[j]["end"]
                        break
                for j in range(i + 1, total_user):
                    if j in result_map:
                        next_start = result_map[j]["start"]
                        break
                est_start = (prev_end + next_start) / 2
                est_end = est_start + 1.0
                final.append({
                    "text": user_text[i],
                    "start": est_start,
                    "end": est_end,
                    "source": []
                })

        (pd(name) / "sentences_uploaded.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), "utf-8")

        socketio.emit('match_progress', {
            'project': name,
            'step': 'done',
            'msg': f'匹配完成：{total_user} 句',
            'progress': 100,
            'result': final
        }, namespace='/')
    except Exception as e:
        socketio.emit('match_progress', {
            'project': name,
            'step': 'error',
            'msg': str(e),
            'progress': 0
        }, namespace='/')

# ── Routes: voice clone ─────────────────────────────────────────────
@app.route("/api/project/<name>/voice-clone", methods=["POST"])
def api_voice_clone(name):
    d = pd(name)
    voice_id = request.form.get("voice_id", "")
    prompt_text = request.form.get("prompt_text", "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题，希望你们能够喜欢这个内容，也希望大家能够多多支持，谢谢你们的关注和鼓励。")

    # Voice library mode: no sample upload needed
    if voice_id and any(v["id"] == voice_id for v in VOICE_LIBRARY):
        save_state(name, stage="cloning", msg="音色克隆中...", clone_progress=[0, 0],
                   voice_id=voice_id, clone_prompt_text=prompt_text)
        threading.Thread(target=_pipeline_voice_clone, args=(name, prompt_text, voice_id), daemon=True).start()
        return jsonify({"ok": True})

    # Custom voice mode: copy sample.wav to project dir
    custom_voice_dir = CUSTOM_VOICES_DIR / voice_id if voice_id else None
    if custom_voice_dir and (custom_voice_dir / "sample.wav").exists():
        shutil.copy2(str(custom_voice_dir / "sample.wav"), str(d / "voice_sample.wav"))
        save_state(name, stage="cloning", msg="音色克隆中...", clone_progress=[0, 0],
                   voice_id=voice_id, clone_prompt_text=prompt_text)
        threading.Thread(target=_pipeline_voice_clone, args=(name, prompt_text, voice_id), daemon=True).start()
        return jsonify({"ok": True})

    # User-uploaded sample mode
    f = request.files.get("sample")
    if not f:
        return jsonify({"error": "请上传音色样本或选择一个预设音色"}), 400
    raw_ext = Path(f.filename).suffix if f.filename else ".webm"
    raw_path = d / f"voice_sample_raw{raw_ext}"
    wav_path = d / "voice_sample.wav"
    f.save(str(raw_path))
    subprocess.run([FFMPEG, "-y", "-i", str(raw_path),
                    "-ar", "24000", "-ac", "1", "-f", "wav", str(wav_path)],
                   check=True, capture_output=True)
    raw_path.unlink(missing_ok=True)
    save_state(name, stage="cloning", msg="音色克隆中...", clone_progress=[0, 0], clone_prompt_text=prompt_text)
    threading.Thread(target=_pipeline_voice_clone, args=(name, prompt_text, ""), daemon=True).start()
    return jsonify({"ok": True})

def _load_custom_voices():
    """Load all custom voice metadata from disk."""
    voices = []
    for d in sorted(CUSTOM_VOICES_DIR.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            v = json.loads(meta.read_text("utf-8"))
            v["custom"] = True
            voices.append(v)
    return voices

@app.route("/api/voices")
def api_list_voices():
    return jsonify(VOICE_LIBRARY + _load_custom_voices())

@app.route("/api/project/<name>/resume-clone", methods=["POST"])
def api_resume_clone(name):
    """Resume a stalled cloning task from where it left off."""
    state = load_state(name)
    if state.get("stage") != "cloning":
        return jsonify({"error": "当前不在克隆阶段"}), 400
    voice_id = state.get("voice_id", "")
    prompt_text = state.get("clone_prompt_text",
        "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题，希望你们能够喜欢这个内容，也希望大家能够多多支持，谢谢你们的关注和鼓励。")
    save_state(name, stage="cloning", msg="恢复克隆中...", clone_progress=state.get("clone_progress", [0, 0]))
    threading.Thread(target=_pipeline_voice_clone, args=(name, prompt_text, voice_id), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/cancel-clone", methods=["POST"])
def api_cancel_clone(name):
    """Cancel an in-progress cloning task and clear all clone data."""
    _cancel_clone.add(name)
    # Delete all clone files
    rec_dir = pd(name) / "recordings"
    count = 0
    for f in rec_dir.glob("s_*_clone.webm"):
        f.unlink(missing_ok=True)
        count += 1
    for f in rec_dir.glob("s_*_clone.wav"):
        f.unlink(missing_ok=True)
    save_state(name, stage="recording", msg="", clone_progress=[0, 0], voice_id="", clone_prompt_text="")
    return jsonify({"ok": True, "deleted": count})
def api_voice_preview_get(voice_id):
    """Stream a cached voice preview. Returns 404 if not yet generated."""
    voice = next((v for v in VOICE_LIBRARY if v["id"] == voice_id), None)
    if not voice:
        return jsonify({"error": "未找到该音色"}), 404
    cache_path = VOICE_CACHE / f"{voice_id}.webm"
    if not cache_path.exists():
        return jsonify({"error": "未缓存", "cached": False}), 404
    return send_file(str(cache_path), mimetype="audio/webm")

@app.route("/api/project/<name>/voice-preview", methods=["POST"])
def api_voice_preview(name):
    data = request.json or {}
    voice_id = data.get("voice_id", "")
    preview_text = data.get("text", "这是一段音色预览，你可以听一下这个声音的效果。")
    voice = next((v for v in VOICE_LIBRARY if v["id"] == voice_id), None)
    if not voice:
        return jsonify({"error": "未找到该音色"}), 400

    # Global cache
    cache_path = VOICE_CACHE / f"{voice_id}.webm"
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="audio/webm")

    try:
        sys.path.insert(0, str(COSYVOICE_DIR))
        sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio

        model_dir, model_name = _get_cosyvoice_model()
        if not model_dir:
            return jsonify({"error": "未找到 CosyVoice 模型"}), 500
        model = AutoModel(model_dir=model_dir)

        ref_wav = str(COSYVOICE_DIR / "asset" / voice.get("ref", "zero_shot_prompt.wav"))
        for j, result in enumerate(model.inference_instruct2(
                preview_text, voice["instruct"], ref_wav, stream=False)):
            wav_out = VOICE_CACHE / f"{voice_id}.wav"
            torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
            subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                            "-c:a", "libopus", "-b:a", "64k", str(cache_path)],
                           check=True, capture_output=True)
            wav_out.unlink(missing_ok=True)
        if cache_path.exists():
            return send_file(str(cache_path), mimetype="audio/webm")
        return jsonify({"error": "生成失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/custom-voices", methods=["POST"])
def api_save_voice():
    """Save a voice sample as a reusable custom voice. Accepts FormData with sample audio + name."""
    voice_name = request.form.get("name", "").strip()
    prompt_text = request.form.get("prompt_text", "")
    if not voice_name:
        return jsonify({"error": "请输入音色名称"}), 400

    f = request.files.get("sample")
    if not f:
        return jsonify({"error": "请上传音色样本"}), 400

    # Create custom voice directory
    voice_id = f"custom_{uuid.uuid4().hex[:8]}"
    voice_dir = CUSTOM_VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)

    # Save and convert to wav
    raw_ext = Path(f.filename).suffix if f.filename else ".webm"
    raw_path = voice_dir / f"sample_raw{raw_ext}"
    wav_path = voice_dir / "sample.wav"
    f.save(str(raw_path))
    subprocess.run([FFMPEG, "-y", "-i", str(raw_path),
                    "-ar", "24000", "-ac", "1", "-f", "wav", str(wav_path)],
                   check=True, capture_output=True)
    raw_path.unlink(missing_ok=True)

    # Use the original recording as preview (no TTS generation)
    preview_path = voice_dir / "preview.webm"
    subprocess.run([FFMPEG, "-y", "-i", str(wav_path),
                    "-c:a", "libopus", "-b:a", "64k", str(preview_path)],
                   check=True, capture_output=True)

    # Save metadata
    meta = {"id": voice_id, "name": voice_name,
            "desc": "用户保存的音色", "custom": True,
            "prompt_text": prompt_text,
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
    (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return jsonify({"ok": True, "voice": meta})

@app.route("/api/custom-voices/<voice_id>/rename", methods=["POST"])
def api_rename_custom_voice(voice_id):
    data = request.json or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "名称不能为空"}), 400
    voice_dir = CUSTOM_VOICES_DIR / voice_id
    meta_path = voice_dir / "meta.json"
    if not voice_dir.exists() or not meta_path.exists():
        return jsonify({"error": "未找到该音色"}), 404
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["name"] = new_name
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return jsonify({"ok": True})

@app.route("/api/custom-voices/<voice_id>", methods=["DELETE"])
def api_delete_custom_voice(voice_id):
    voice_dir = CUSTOM_VOICES_DIR / voice_id
    if not voice_dir.exists() or not (voice_dir / "meta.json").exists():
        return jsonify({"error": "未找到该音色"}), 404
    # Check if this voice is currently being used for cloning in any project
    for pdir in PROJECTS.iterdir():
        if not pdir.is_dir():
            continue
        sp = pdir / "state.json"
        if not sp.exists():
            continue
        try:
            st = json.loads(sp.read_text("utf-8"))
            if st.get("stage") == "cloning" and st.get("voice_id") == voice_id:
                return jsonify({"error": "该音色正在克隆中使用，无法删除"}), 409
        except Exception:
            pass
    shutil.rmtree(str(voice_dir))
    return jsonify({"ok": True})

@app.route("/api/custom-voices/<voice_id>/sample")
def api_custom_voice_sample(voice_id):
    """Return the sample audio of a custom voice (for cloning)."""
    sample = CUSTOM_VOICES_DIR / voice_id / "sample.wav"
    if not sample.exists():
        return ("", 404)
    return send_file(str(sample), mimetype="audio/wav")

@app.route("/api/custom-voices/<voice_id>/preview")
def api_custom_voice_preview(voice_id):
    """Return the preview audio of a custom voice."""
    preview = CUSTOM_VOICES_DIR / voice_id / "preview.webm"
    if not preview.exists():
        return ("", 404)
    return send_file(str(preview), mimetype="audio/webm")

def _pipeline_transcribe(name):
    d = pd(name)
    inp = _input_video(name)
    try:
        save_state(name, stage="processing", msg="提取音频...")
        subprocess.run([FFMPEG, "-y", "-i", str(inp),
                        "-vn", "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(d/"audio_16k.wav")],
                       check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # Get duration
        probe = subprocess.run(
            [FFMPEG, "-i", str(inp)],
            capture_output=True, encoding='utf-8', errors='ignore')
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
        dur = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 600
        (d / "_duration.txt").write_text(str(dur))

        save_state(name, stage="processing", msg="语音转写中...", duration=dur, transcribe_progress=[0, int(dur)])

        # Run Whisper in subprocess to avoid blocking Flask/WebSocket
        import time
        worker = subprocess.Popen(
            [sys.executable, str(BASE / "transcribe_worker.py"), name, str(d), HF_CACHE],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Poll state.json and push updates via WebSocket
        last_progress = None
        while worker.poll() is None:
            time.sleep(2)
            try:
                st = json.loads((d / "state.json").read_text("utf-8"))
                progress = st.get("transcribe_progress")
                if progress != last_progress:
                    last_progress = progress
                    socketio.emit('project_update', {'project': name, 'state': st}, namespace='/')
            except Exception:
                pass

        # Final state push
        if d.exists():
            st = load_state(name)
            socketio.emit('project_update', {'project': name, 'state': st}, namespace='/')

        # Cleanup
        (d / "_duration.txt").unlink(missing_ok=True)
    except Exception as e:
        save_state(name, stage="error", msg=f"{e}\n{traceback.format_exc()}")
    except Exception as e:
        save_state(name, stage="error", msg=f"{e}\n{traceback.format_exc()}")

def _pipeline_compose(name):
    d = pd(name)
    inp = _input_video(name)
    try:
        import numpy as np, soundfile as sf
        sentences = json.loads(_active_sentences_path(name).read_text("utf-8"))
        dur = load_state(name).get("duration", 600)
        SR = 24000
        total = int(dur * SR)
        out = np.zeros(total, dtype=np.float32)

        for i, seg in enumerate(sentences, 1):
            rec = d / "recordings" / f"s_{i:03d}.webm"
            # Fallback to clone if no manual recording
            if not rec.exists():
                rec = d / "recordings" / f"s_{i:03d}_clone.webm"
            if not rec.exists(): continue
            wav_tmp = d / f"_t{i}.wav"
            subprocess.run([FFMPEG, "-y", "-i", str(rec), "-ar", str(SR),
                            "-ac", "1", "-c:a", "pcm_s16le", str(wav_tmp)],
                           check=True, capture_output=True)
            audio, _ = sf.read(str(wav_tmp)); wav_tmp.unlink(missing_ok=True)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            si = int(seg["start"] * SR)
            ei = min(si + len(audio), total)
            if si < total:
                out[si:ei] = audio[:ei - si]

        mx = float(np.max(np.abs(out)))
        if mx > 0: out = out / mx * 0.95
        sf.write(str(d / "final_audio.wav"), out, SR)

        save_state(name, stage="composing", msg="生成字幕...")
        srt = d / "final.srt"
        with open(str(srt), "w", encoding="utf-8") as f:
            for i, seg in enumerate(sentences, 1):
                f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text']}\n\n")

        save_state(name, stage="composing", msg="合成视频（编码中）...")
        srt_p = str(srt).replace("\\", "/").replace(":", "\\:")
        subprocess.run([
            FFMPEG, "-y",
            "-i", str(inp), "-i", str(d / "final_audio.wav"),
            "-vf", (f"subtitles='{srt_p}':force_style='FontName=Microsoft YaHei,"
                    "FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                    "Outline=1,Shadow=0,Alignment=2,MarginV=30'"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(d / "final.mp4")
        ], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        save_state(name, stage="done", msg="完成！点击下载")
    except Exception as e:
        save_state(name, stage="error", msg=f"{e}\n{traceback.format_exc()}")

def _get_cosyvoice_model():
    """Find best available CosyVoice model."""
    for name in ["Fun-CosyVoice3-0.5B", "CosyVoice2-0.5B", "CosyVoice-300M"]:
        if (COSYVOICE_DIR / "pretrained_models" / name).exists():
            return str(COSYVOICE_DIR / "pretrained_models" / name), name
    return None, None

def _pipeline_voice_clone(name, prompt_text, voice_id=""):
    d = pd(name)
    try:
        sys.path.insert(0, str(COSYVOICE_DIR))
        sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio

        model_dir, model_name = _get_cosyvoice_model()
        if not model_dir:
            raise FileNotFoundError("未找到 CosyVoice 模型，请先下载模型到 E:/视频处理/CosyVoice/pretrained_models/")

        model = AutoModel(model_dir=model_dir)
        sentences = json.loads(_active_sentences_path(name).read_text("utf-8"))
        total = len(sentences)

        # Check which sentences already have clone files (resume support)
        rec_dir = d / "recordings"
        done = sum(1 for i in range(total) if (rec_dir / f"s_{i+1:03d}_clone.webm").exists())
        if done >= total:
            save_state(name, stage="recording", msg="音色克隆完成，可试听或直接生成视频")
            return
        save_state(name, stage="cloning",
                   msg=f"音色克隆中 {done}/{total}..." + ("（恢复中）" if done > 0 else ""),
                   clone_progress=[done, total])

        # Determine inference mode
        voice = next((v for v in VOICE_LIBRARY if v["id"] == voice_id), None)
        is_custom = voice_id.startswith("custom_") and (CUSTOM_VOICES_DIR / voice_id / "sample.wav").exists()
        use_instruct = voice is not None and "CosyVoice" in model_name and "300M" not in model_name

        # For zero-shot mode, auto-transcribe the sample so prompt_text matches the audio content.
        # This prevents reference-audio content from leaking into synthesized output.
        if not use_instruct:
            sample_wav = d / "voice_sample.wav"
            if sample_wav.exists():
                transcribed = _transcribe_sample(sample_wav)
                if transcribed:
                    prompt_text = transcribed
                    # Save back to state so resume-clone uses the correct prompt
                    save_state(name, clone_prompt_text=prompt_text)

        if use_instruct:
            # Voice library mode: use inference_instruct2
            ref_wav = str(COSYVOICE_DIR / "asset" / voice.get("ref", "zero_shot_prompt.wav"))
            for i, seg in enumerate(sentences):
                if name in _cancel_clone:
                    _cancel_clone.discard(name)
                    return
                if (rec_dir / f"s_{i+1:03d}_clone.webm").exists():
                    continue  # Skip already cloned
                for j, result in enumerate(model.inference_instruct2(
                        seg["text"], voice["instruct"], ref_wav, stream=False)):
                    wav_out = d / "recordings" / f"s_{i+1:03d}_clone.wav"
                    torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                    webm_out = d / "recordings" / f"s_{i+1:03d}_clone.webm"
                    subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                                    "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                                   check=True, capture_output=True)
                    wav_out.unlink(missing_ok=True)
                save_state(name, stage="cloning",
                           msg=f"音色克隆中 {i+1}/{total}...",
                           clone_progress=[i + 1, total])
        else:
            # Zero-shot mode: use user-uploaded sample
            if "CosyVoice3" in model_name:
                full_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
            else:
                full_prompt = prompt_text
            sample_path = str(d / "voice_sample.wav")

            for i, seg in enumerate(sentences):
                if name in _cancel_clone:
                    _cancel_clone.discard(name)
                    return
                if (rec_dir / f"s_{i+1:03d}_clone.webm").exists():
                    continue  # Skip already cloned
                for j, result in enumerate(model.inference_zero_shot(
                        seg["text"], full_prompt, sample_path, stream=False)):
                    wav_out = d / "recordings" / f"s_{i+1:03d}_clone.wav"
                    torchaudio.save(str(wav_out), result["tts_speech"], model.sample_rate)
                    webm_out = d / "recordings" / f"s_{i+1:03d}_clone.webm"
                    subprocess.run([FFMPEG, "-y", "-i", str(wav_out),
                                    "-c:a", "libopus", "-b:a", "64k", str(webm_out)],
                                   check=True, capture_output=True)
                    wav_out.unlink(missing_ok=True)
                save_state(name, stage="cloning",
                           msg=f"音色克隆中 {i+1}/{total}...",
                           clone_progress=[i + 1, total])

        save_state(name, stage="recording", msg="音色克隆完成，可试听或直接生成视频")
    except Exception as e:
        save_state(name, stage="error", msg=f"音色克隆失败: {e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# Video Editor module (delete / insert segments)
# ═══════════════════════════════════════════════════════════════════
def _parse_ranges(text):
    """Parse '10-11, 20-21, 30-32' into [(10.0, 11.0), (20.0, 21.0), ...]"""
    ranges = []
    for part in text.replace(';', ',').split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            a, b = part.split('-', 1)
            ranges.append((float(a.strip()), float(b.strip())))
        else:
            t = float(part)
            ranges.append((t, t))
    return sorted(ranges, key=lambda x: x[0])

@app.route("/api/project/<name>/video-edit/delete", methods=["POST"])
def api_ve_delete(name):
    d = pd(name)
    ranges_text = request.json.get("ranges", "")
    if not ranges_text.strip():
        return jsonify({"error": "请输入要删除的时间段"}), 400
    try:
        ranges = _parse_ranges(ranges_text)
    except Exception:
        return jsonify({"error": "格式错误，请用 10-11, 20-21 格式"}), 400

    save_state(name, stage="editing", msg="视频裁剪中...", sub="video_edit")
    threading.Thread(target=_pipeline_ve_delete, args=(name, ranges), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/video-edit/insert-video", methods=["POST"])
def api_ve_insert_video(name):
    d = pd(name)
    f = request.files.get("video")
    if not f: return jsonify({"error": "请上传视频"}), 400
    try:
        pos = float(request.form.get("position", "0"))
    except: return jsonify({"error": "位置格式错误"}), 400
    tmp = d / "_ins_tmp"
    f.save(str(tmp))
    _normalize_video(str(tmp), str(d / "insert_segment.mp4"))
    tmp.unlink(missing_ok=True)
    save_state(name, stage="editing", msg="插入片段中...", sub="video_edit")
    threading.Thread(target=_pipeline_ve_insert, args=(name, pos, "video"), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/video-edit/insert-images", methods=["POST"])
def api_ve_insert_images(name):
    d = pd(name)
    try:
        pos = float(request.form.get("position", "0"))
    except: return jsonify({"error": "位置格式错误"}), 400
    subtitles_raw = request.form.get("subtitles", "[]")
    subtitles = json.loads(subtitles_raw)

    img_dir = d / "insert_images"
    img_dir.mkdir(exist_ok=True)
    idx = 0
    for key in sorted(request.files):
        f = request.files[key]
        if not f or not f.filename: continue
        ext = Path(f.filename).suffix or ".jpg"
        f.save(str(img_dir / f"img_{idx:03d}{ext}"))
        idx += 1
    if idx == 0: return jsonify({"error": "请上传图片"}), 400

    # voice sample from project if exists
    has_voice = (d / "voice_sample.wav").exists()

    save_state(name, stage="editing", msg="生成图片视频并插入中...", sub="video_edit")
    threading.Thread(target=_pipeline_ve_insert_images,
                     args=(name, pos, subtitles, has_voice), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/video-edit/speedup", methods=["POST"])
def api_ve_speedup(name):
    data = request.json or {}
    try:
        start = float(data.get("start", 0))
        end = float(data.get("end", 0))
        rate = float(data.get("rate", 2))
    except (ValueError, TypeError):
        return jsonify({"error": "参数格式错误"}), 400
    if end <= start or rate <= 1:
        return jsonify({"error": "结束时间需大于起始时间，倍速需大于1"}), 400
    threading.Thread(target=_pipeline_ve_speedup, args=(name, start, end, rate), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/video-edit/speedup-merge", methods=["POST"])
def api_ve_speedup_merge(name):
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "请上传录音"}), 400
    d = pd(name)
    if not (d / "speedup_meta.json").exists():
        return jsonify({"error": "请先执行变速操作"}), 400
    ext = Path(audio.filename).suffix or ".webm"
    audio_path = d / f"speedup_audio{ext}"
    audio.save(str(audio_path))
    threading.Thread(target=_pipeline_ve_speedup_merge, args=(name, str(audio_path)), daemon=True).start()
    return jsonify({"ok": True})

def _pipeline_ve_delete(name, ranges):
    d = pd(name)
    inp = _input_video(name)
    try:
        # Get total duration
        probe = subprocess.run([FFMPEG, "-i", str(inp)], capture_output=True, encoding='utf-8', errors='ignore')
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
        total = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 600

        # Build keep segments (inverse of delete ranges)
        keep = []
        prev_end = 0.0
        for start, end in ranges:
            if start > prev_end:
                keep.append((prev_end, start))
            prev_end = max(prev_end, end)
        if prev_end < total:
            keep.append((prev_end, total))

        if not keep:
            raise ValueError("删除后没有剩余内容")

        # Extract each segment
        seg_files = []
        for i, (s, e) in enumerate(keep):
            seg = d / f"_seg_{i:03d}.mp4"
            subprocess.run([FFMPEG, "-y", "-i", str(inp),
                            "-ss", str(s), "-to", str(e),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k",
                            str(seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
            seg_files.append(seg)

        out = d / "input_edited.mp4"
        _ffmpeg_concat(seg_files, str(out))

        # Replace original (always output as .mp4 after editing)
        inp.rename(d / "input_backup")
        out.rename(d / "input.mp4")

        # Cleanup
        for f in seg_files: f.unlink(missing_ok=True)

        save_state(name, stage="editing", msg="视频裁剪完成！可下载或继续编辑",
                   sub="video_edit_done")
    except Exception as e:
        save_state(name, stage="error", msg=f"视频裁剪失败: {e}\n{traceback.format_exc()}")

def _get_video_info(path):
    """Return (width, height, duration) of a video file."""
    probe = subprocess.run([FFMPEG, "-i", str(path)],
                           capture_output=True, encoding='utf-8', errors='ignore')
    # Resolution always follows ", " and precedes " [" in ffmpeg output
    m = re.search(r",\s*(\d+)x(\d+)\s", probe.stderr or "")
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)
    m2 = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
    dur = int(m2.group(1))*3600 + int(m2.group(2))*60 + float(m2.group(3)) if m2 else 600
    return w, h, dur

def _reencode_to(path_in, path_out, width, height, extra_args=None):
    """Re-encode video to exactly width x height (scale + pad with black bars)."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")
    cmd = [FFMPEG, "-y", "-i", str(path_in),
           "-vf", vf,
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k",
           "-r", "25", "-pix_fmt", "yuv420p"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(path_out))
    subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='ignore')

def _ffmpeg_concat_images(image_files_with_dur, output_path, vf, extra_args=None):
    """Concat images with durations into a video. Handles non-ASCII paths."""
    tmpdir = tempfile.mkdtemp(prefix="vs_")
    try:
        lines = []
        for i, (img_path, dur) in enumerate(image_files_with_dur):
            ext = Path(img_path).suffix or ".jpg"
            tmp_img = os.path.join(tmpdir, f"img_{i:04d}{ext}")
            shutil.copy2(str(img_path), tmp_img)
            lines.append(f"file 'img_{i:04d}{ext}'")
            lines.append(f"duration {dur:.3f}")
        # Repeat last image to avoid ffmpeg cutting the last frame
        if image_files_with_dur:
            last_ext = Path(image_files_with_dur[-1][0]).suffix or ".jpg"
            lines.append(f"file 'img_{len(image_files_with_dur)-1:04d}{last_ext}'")
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
               "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(str(output_path))
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='ignore')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _ffmpeg_concat(segment_paths, output_path):
    """Concatenate video segments into one file. Handles non-ASCII paths by using a temp dir."""
    tmpdir = tempfile.mkdtemp(prefix="vs_")
    try:
        lines = []
        for i, seg in enumerate(segment_paths):
            tmp_seg = os.path.join(tmpdir, f"s{i:04d}.mp4")
            shutil.copy2(str(seg), tmp_seg)
            lines.append(f"file 's{i:04d}.mp4'")
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        tmp_out = os.path.join(tmpdir, "out.mp4")
        subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_file, "-c", "copy", tmp_out],
                       check=True, capture_output=True, encoding='utf-8', errors='ignore')
        shutil.copy2(tmp_out, str(output_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _normalize_video(path_in, path_out):
    """Normalize any video format to MP4 (H264 + AAC, yuv420p)."""
    subprocess.run([FFMPEG, "-y", "-i", str(path_in),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", "25",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(path_out)],
                   check=True, capture_output=True, encoding='utf-8', errors='ignore')

def _pipeline_ve_insert(name, pos, mode):
    d = pd(name)
    try:
        inp = _input_video(name)
        seg = str(d / "insert_segment.mp4")
        part1 = d / "_part1.mp4"
        part2 = d / "_part2.mp4"
        out = d / "input_edited.mp4"

        # Detect original resolution
        ow, oh, _ = _get_video_info(str(inp))

        # Split original at pos — re-encode to normalized resolution
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(pos),
                        "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(pos),
                        "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Re-encode insert segment to match original resolution (handles AVI/MKV/MOV etc.)
        seg_enc = d / "_seg_enc.mp4"
        subprocess.run([FFMPEG, "-y", "-i", seg,
                        "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(seg_enc)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Concat: part1 + segment + part2 (all same resolution, codec, fps)
        out = d / "input_edited.mp4"
        _ffmpeg_concat([part1, seg_enc, part2], str(out))

        # Replace original (always output as .mp4 after editing)
        inp.rename(d / "input_backup2")
        out.rename(d / "input.mp4")

        # Cleanup
        for f in [part1, seg_enc, d/"insert_segment.mp4"]:
            f.unlink(missing_ok=True)

        save_state(name, stage="editing", msg="视频插入完成！",
                   sub="video_edit_done")
    except Exception as e:
        save_state(name, stage="error", msg=f"插入失败: {e}\n{traceback.format_exc()}")

def _pipeline_ve_speedup(name, start, end, rate):
    d = pd(name)
    try:
        inp = _input_video(name)
        ow, oh, total_dur = _get_video_info(str(inp))

        part1 = d / "_sp_part1.mp4"
        part2 = d / "_sp_part2.mp4"
        speed_raw = d / "_sp_raw.mp4"
        speed_up = d / "_sp_fast.mp4"
        out = d / "input_edited.mp4"

        # Extract the three parts
        # part1: 0 → start
        if start > 0:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(start),
                            "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                    f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # speed segment: start → end
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(start), "-to", str(end),
                        "-an",
                        "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-r", "25", "-pix_fmt", "yuv420p",
                        str(speed_raw)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # Speed up (remove audio with -an, use setpts)
        subprocess.run([FFMPEG, "-y", "-i", str(speed_raw),
                        "-vf", f"setpts=PTS/{rate}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(speed_up)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # part2: end → total
        if end < total_dur - 0.1:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(end),
                            "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                    f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Get speed_up segment duration
        _, _, new_seg_dur = _get_video_info(str(speed_up))

        # Concat parts
        segments = []
        if start > 0:
            segments.append(part1)
        segments.append(speed_up)
        if end < total_dur - 0.1:
            segments.append(part2)
        _ffmpeg_concat(segments, str(out))

        # Save metadata for later merge
        new_start = start
        new_end = start + new_seg_dur
        meta = {"start": start, "end": end, "rate": rate,
                "new_start": new_start, "new_end": new_end,
                "new_seg_duration": new_seg_dur}
        (d / "speedup_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        # Replace original
        backup_name = "input_backup_sp"
        if (d / "input.mp4").exists():
            (d / "input.mp4").rename(d / backup_name)
        out.rename(d / "input.mp4")

        # Cleanup
        for f in [part1, part2, speed_raw, speed_up]:
            f.unlink(missing_ok=True)

        save_state(name, stage="editing", msg="变速完成！请录制新音频",
                   sub="speedup_done")
    except Exception as e:
        save_state(name, stage="error", msg=f"变速失败: {e}\n{traceback.format_exc()}")

def _pipeline_ve_speedup_merge(name, audio_path):
    d = pd(name)
    try:
        inp = _input_video(name)
        meta = json.loads((d / "speedup_meta.json").read_text("utf-8"))
        new_start = meta["new_start"]
        new_end = meta["new_end"]

        ow, oh, total_dur = _get_video_info(str(inp))

        part1 = d / "_sm_part1.mp4"
        part2 = d / "_sm_part2.mp4"
        speed_seg = d / "_sm_speed.mp4"
        merged_seg = d / "_sm_merged.mp4"
        out = d / "input_edited.mp4"

        # part1: 0 → new_start
        if new_start > 0:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(new_start),
                            "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                    f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # speed segment: new_start → new_end (no audio)
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(new_start), "-to", str(new_end),
                        "-an",
                        "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(speed_seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # part2: new_end → end
        if new_end < total_dur - 0.1:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(new_end),
                            "-vf", (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
                                    f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Merge audio with speed segment
        subprocess.run([FFMPEG, "-y",
                        "-i", str(speed_seg), "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest",
                        str(merged_seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Concat all parts
        segments = []
        if new_start > 0:
            segments.append(part1)
        segments.append(merged_seg)
        if new_end < total_dur - 0.1:
            segments.append(part2)
        _ffmpeg_concat(segments, str(out))

        # Replace original
        backup_name = "input_backup_sm"
        if (d / "input.mp4").exists():
            (d / "input.mp4").rename(d / backup_name)
        out.rename(d / "input.mp4")

        # Cleanup
        for f in [part1, part2, speed_seg, merged_seg, Path(audio_path)]:
            f.unlink(missing_ok=True)
        (d / "speedup_meta.json").unlink(missing_ok=True)

        save_state(name, stage="editing", msg="视频合成完成！")
    except Exception as e:
        save_state(name, stage="error", msg=f"合并失败: {e}\n{traceback.format_exc()}")

def _pipeline_ve_insert_images(name, pos, subtitles, has_voice):
    d = pd(name)
    try:
        # Generate a video segment from images (reuse CosyVoice TTS)
        sys.path.insert(0, str(COSYVOICE_DIR))
        sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio, numpy as np, soundfile as sf

        model_dir, model_name = _get_cosyvoice_model()
        if not model_dir: raise FileNotFoundError("未找到 CosyVoice 模型")
        model = AutoModel(model_dir=model_dir)

        voice_sample = d / "voice_sample.wav" if has_voice else COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav"
        prompt_text = "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题，希望你们能够喜欢这个内容，也希望大家能够多多支持，谢谢你们的关注和鼓励。"
        full_prompt = ("You are a helpful assistant.<|endofprompt|>" + prompt_text) if "CosyVoice3" in (model_name or "") else prompt_text

        img_dir = d / "insert_images"
        img_files = sorted(img_dir.glob("img_*"))

        audio_segments = []
        for i, sub in enumerate(subtitles):
            text = sub if isinstance(sub, str) else sub.get("text", sub.get("narration", ""))
            for j, result in enumerate(model.inference_zero_shot(text, full_prompt, str(voice_sample), stream=False)):
                audio_segments.append(result["tts_speech"].cpu().numpy().squeeze())

        # Build video from images with matching durations
        SR = model.sample_rate
        total_audio = np.concatenate(audio_segments) if audio_segments else np.zeros(SR, dtype=np.float32)

        img_durs = [(img_files[min(i, len(img_files)-1)], len(seg)/SR) for i, seg in enumerate(audio_segments)]

        mx = float(np.max(np.abs(total_audio)))
        if mx > 0: total_audio = total_audio / mx * 0.95
        sf.write(str(d / "insert_images" / "audio.wav"), total_audio.astype(np.float32), SR)

        # Detect original video resolution to match
        ow, oh, _ = _get_video_info(str(_input_video(name)))

        # Create image video segment at original resolution
        img_durs = [(img_files[min(i, len(img_files)-1)], len(seg)/SR) for i, seg in enumerate(audio_segments)]
        _ffmpeg_concat_images(img_durs, str(d / "insert_images" / "video_only.mp4"),
                              f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black")

        # Combine video + audio
        subprocess.run([FFMPEG, "-y",
                        "-i", str(d / "insert_images" / "video_only.mp4"),
                        "-i", str(d / "insert_images" / "audio.wav"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-shortest",
                        str(d / "insert_segment.mp4")
                       ], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Now insert into original
        (d / "insert_images").mkdir(exist_ok=True)  # already exists
        _pipeline_ve_insert(name, pos, "images")
    except Exception as e:
        save_state(name, stage="error", msg=f"图片插入失败: {e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# Image-to-Video module
# ═══════════════════════════════════════════════════════════════════
def i2v_path(name):
    return IMG2VID / name

def i2v_state(name):
    p = i2v_path(name) / "state.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {"stage": "new"}

def i2v_save(name, **kw):
    s = i2v_state(name); s.update(kw)
    (i2v_path(name) / "state.json").write_text(json.dumps(s, ensure_ascii=False), "utf-8")

@app.route("/api/img2vid")
def api_i2v_list():
    out = []
    for d in sorted(IMG2VID.iterdir()):
        if d.is_dir() and (d / "state.json").exists():
            out.append({"name": d.name, **i2v_state(d.name)})
    return jsonify(out)

@app.route("/api/img2vid", methods=["POST"])
def api_i2v_create():
    name = request.form.get("name", "").strip()
    if not name: return jsonify({"error": "项目名必填"}), 400
    d = IMG2VID / name
    if d.exists(): return jsonify({"error": "项目已存在"}), 400
    theme = request.form.get("theme", "")
    d.mkdir(parents=True); (d / "images").mkdir(); (d / "recordings").mkdir()
    # save uploaded images in order
    idx = 0
    for key in sorted(request.files):
        f = request.files[key]
        if not f or not f.filename: continue
        ext = Path(f.filename).suffix or ".jpg"
        f.save(str(d / "images" / f"img_{idx:03d}{ext}"))
        idx += 1
    if idx == 0:
        return jsonify({"error": "请上传至少一张图片"}), 400
    i2v_save(name, stage="uploading", theme=theme, image_count=idx)
    return jsonify({"name": name, "image_count": idx})

@app.route("/api/img2vid/<name>")
def api_i2v_status(name):
    return jsonify(i2v_state(name))

@app.route("/api/img2vid/<name>/images")
def api_i2v_images(name):
    d = i2v_path(name) / "images"
    files = sorted(d.glob("img_*"))
    return jsonify([{"file": f.name, "url": f"/api/img2vid/{name}/image/{f.name}"} for f in files])

@app.route("/api/img2vid/<name>/image/<fname>")
def api_i2v_image(name, fname):
    p = i2v_path(name) / "images" / fname
    if not p.exists(): return ("", 404)
    mt = "image/jpeg" if fname.endswith(".jpg") else "image/png"
    return send_file(str(p), mimetype=mt)

@app.route("/api/img2vid/<name>/reorder", methods=["POST"])
def api_i2v_reorder(name):
    order = request.json.get("order", [])
    d = i2v_path(name) / "images"
    files = {f.name: f for f in d.glob("img_*")}
    # rename to temp then to final order
    for i, fname in enumerate(order):
        src = d / fname
        if src.exists():
            ext = Path(fname).suffix
            tmp = d / f"_tmp_{i:03d}{ext}"
            src.rename(tmp)
    for f in list(d.glob("_tmp_*")):
        idx = int(f.stem.split("_")[-1])
        ext = f.suffix
        f.rename(d / f"img_{idx:03d}{ext}")
    return jsonify({"ok": True})

@app.route("/api/img2vid/<name>/analyze", methods=["POST"])
def api_i2v_analyze(name):
    i2v_save(name, stage="analyzing", msg="AI 正在识别图片并生成旁白...")
    theme = i2v_state(name).get("theme", "")
    threading.Thread(target=_pipeline_i2v_analyze, args=(name, theme), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/img2vid/<name>/narration")
def api_i2v_narration(name):
    p = i2v_path(name) / "narration.json"
    return jsonify(json.loads(p.read_text("utf-8")) if p.exists() else [])

@app.route("/api/img2vid/<name>/narration", methods=["PUT"])
def api_i2v_save_narration(name):
    data = request.json
    (i2v_path(name) / "narration.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    i2v_save(name, stage="narration_ready", msg="旁白已保存")
    return jsonify({"ok": True})

@app.route("/api/img2vid/<name>/voice-sample", methods=["POST"])
def api_i2v_voice(name):
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "请上传音频"}), 400
    d = i2v_path(name)
    raw = d / "_voice_raw.webm"
    f.save(str(raw))
    # Convert to standard 24kHz mono WAV for CosyVoice
    subprocess.run([FFMPEG, "-y", "-i", str(raw),
                    "-ar", "24000", "-ac", "1", "-f", "wav",
                    str(d / "voice_sample.wav")],
                   check=True, capture_output=True)
    raw.unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/api/img2vid/<name>/generate", methods=["POST"])
def api_i2v_generate(name):
    i2v_save(name, stage="generating", msg="开始生成视频...")
    threading.Thread(target=_pipeline_i2v_generate, args=(name,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/img2vid/<name>/download")
def api_i2v_download(name):
    p = i2v_path(name) / "final.mp4"
    if not p.exists(): return ("", 404)
    return send_file(str(p), mimetype="video/mp4",
                     as_attachment=True, attachment_filename=f"{name}.mp4")

@app.route("/api/img2vid/<name>", methods=["DELETE"])
def api_i2v_delete(name):
    d = i2v_path(name)
    if d.exists(): shutil.rmtree(d)
    return jsonify({"ok": True})

# ── Image-to-Video pipelines ───────────────────────────────────────
def _pipeline_i2v_analyze(name, theme):
    d = i2v_path(name)
    try:
        import PIL.Image
        images_dir = d / "images"
        img_files = sorted(images_dir.glob("img_*"))
        images_b64 = []
        for f in img_files:
            img = PIL.Image.open(f)
            img.thumbnail((1024, 1024))
            buf = __import__("io").BytesIO()
            img.save(buf, format="JPEG", quality=85)
            images_b64.append(base64.b64encode(buf.getvalue()).decode())

        prompt = (
            f"你是一个视频制作专家。以下是按顺序排列的 {len(images_b64)} 张图片。"
            + (f"用户提供的主题是：{theme}\n" if theme else "")
            + "请为每张图片生成一段旁白（2-3句话，适合朗读配音）。\n"
            "要求：旁白要连贯，前后图片之间要有自然过渡；语言口语化。\n"
            "输出JSON数组，每个元素：{\"image_idx\": 序号(从0开始), \"narration\": \"旁白文字\"}"
        )
        resp = call_llm_vision(prompt, images_b64,
                               system="你是一个JSON输出专家，只输出纯JSON数组，不要任何解释文字。")
        cleaned = re.sub(r'```json\s*|\s*```', '', resp).strip()
        narration = json.loads(cleaned)
        if not isinstance(narration, list):
            raise ValueError("LLM 返回格式错误")
        (d / "narration.json").write_text(json.dumps(narration, ensure_ascii=False, indent=2), "utf-8")
        i2v_save(name, stage="narration_ready", msg="旁白生成完成，请查看并编辑")
    except Exception as e:
        i2v_save(name, stage="error", msg=f"分析失败: {e}\n{traceback.format_exc()}")

def _pipeline_i2v_generate(name):
    d = i2v_path(name)
    try:
        narration = json.loads((d / "narration.json").read_text("utf-8"))
        n_items = len(narration)

        # ── Step 1: TTS for each narration segment ──
        i2v_save(name, stage="generating", msg="生成语音 0/{}...".format(n_items))
        sys.path.insert(0, str(COSYVOICE_DIR))
        sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        import torchaudio, numpy as np, soundfile as sf

        model_dir, model_name = _get_cosyvoice_model()
        if not model_dir:
            raise FileNotFoundError("未找到 CosyVoice 模型")
        model = AutoModel(model_dir=model_dir)

        # Check for user voiceprint
        voice_sample = d / "voice_sample.wav"
        has_voiceprint = voice_sample.exists()
        if not has_voiceprint:
            voice_sample = COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav"

        prompt_text = "大家好，欢迎来到我的频道。今天我要和大家分享一个非常有趣的话题，希望你们能够喜欢这个内容，也希望大家能够多多支持，谢谢你们的关注和鼓励。"
        if "CosyVoice3" in (model_name or ""):
            full_prompt = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        else:
            full_prompt = prompt_text

        SR = model.sample_rate
        audio_segments = []  # (numpy_array, sample_rate)

        for i, item in enumerate(narration):
            text = item.get("narration", "")
            i2v_save(name, stage="generating", msg=f"生成语音 {i+1}/{n_items}...")
            for j, result in enumerate(model.inference_zero_shot(
                text, full_prompt, str(voice_sample), stream=False)):
                wav = result["tts_speech"].cpu().numpy().squeeze()
                audio_segments.append((wav, SR))

        # ── Step 2: Create video from images with matching durations ──
        i2v_save(name, stage="generating", msg="合成视频中...")
        images_dir = d / "images"
        img_files = sorted(images_dir.glob("img_*"))
        total_audio = np.concatenate([seg for seg, _ in audio_segments])
        total_dur = len(total_audio) / SR

        # Build image-duration pairs and create video
        img_durs = []
        for i, (seg, seg_sr) in enumerate(audio_segments):
            dur = len(seg) / seg_sr
            img_durs.append((img_files[min(i, len(img_files)-1)], dur))

        # Save combined audio
        mx = float(np.max(np.abs(total_audio)))
        if mx > 0: total_audio = total_audio / mx * 0.95
        sf.write(str(d / "audio.wav"), total_audio.astype(np.float32), SR)

        # Create silent video from images at 1920x1080
        _ffmpeg_concat_images(img_durs, str(d / "video_only.mp4"),
                              "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black")

        # Build SRT subtitles from narration
        srt = d / "final.srt"
        time_offset = 0.0
        with open(str(srt), "w", encoding="utf-8") as f:
            for i, (seg, seg_sr) in enumerate(audio_segments):
                dur = len(seg) / seg_sr
                text = narration[i].get("narration", "") if i < len(narration) else ""
                f.write(f"{i+1}\n{ts(time_offset)} --> {ts(time_offset+dur)}\n{text}\n\n")
                time_offset += dur

        # Combine video + audio + subtitles
        srt_p = str(srt).replace("\\", "/").replace(":", "\\:")
        subprocess.run([
            FFMPEG, "-y",
            "-i", str(d / "video_only.mp4"), "-i", str(d / "audio.wav"),
            "-vf", (f"subtitles='{srt_p}':force_style='FontName=Microsoft YaHei,"
                    "FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                    "Outline=1,Shadow=0,Alignment=2,MarginV=30'"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-shortest",
            str(d / "final.mp4")
        ], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Cleanup temp files
        for f in [d/"video_only.mp4", d/"audio.wav"]:
            f.unlink(missing_ok=True)

        i2v_save(name, stage="done", msg="视频生成完成！")
    except Exception as e:
        i2v_save(name, stage="error", msg=f"生成失败: {e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# Video Conversion module
# ═══════════════════════════════════════════════════════════════════
FORMATS = {
    "mp4":  {"ext": ".mp4",  "codec": "libx264", "audio": "aac",     "mime": "video/mp4"},
    "avi":  {"ext": ".avi",  "codec": "mpeg4",  "audio": "mp3",     "mime": "video/x-msvideo"},
    "mkv":  {"ext": ".mkv",  "codec": "libx264", "audio": "aac",     "mime": "video/x-matroska"},
    "mov":  {"ext": ".mov",  "codec": "libx264", "audio": "aac",     "mime": "video/quicktime"},
    "webm": {"ext": ".webm", "codec": "libvpx-vp9", "audio": "libopus", "mime": "video/webm"},
}

PRESETS = {
    "1080p":  {"w": 1920, "h": 1080},
    "720p":   {"w": 1280, "h": 720},
    "480p":   {"w": 854,  "h": 480},
    "360p":   {"w": 640,  "h": 360},
    "original": None,
}

@app.route("/api/project/<name>/convert", methods=["POST"])
def api_convert(name):
    fmt = request.json.get("format", "mp4")
    res = request.json.get("resolution", "original")
    if fmt not in FORMATS:
        return jsonify({"error": f"不支持的格式: {fmt}"}), 400
    if res not in PRESETS:
        return jsonify({"error": f"不支持的分辨率: {res}"}), 400
    save_state(name, stage="converting", msg=f"转换为 {fmt} {res}...")
    threading.Thread(target=_pipeline_convert, args=(name, fmt, res), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/project/<name>/converted")
def api_converted(name):
    p = pd(name)
    for fmt, cfg in FORMATS.items():
        f = p / f"converted_{fmt}.mp4" if fmt == "mp4" else p / f"converted{cfg['ext']}"
        if f.exists():
            return jsonify({"ready": True, "format": fmt,
                            "url": f"/api/project/{name}/converted-download/{fmt}"})
    return jsonify({"ready": False})

@app.route("/api/project/<name>/converted-download/<fmt>")
def api_converted_download(name, fmt):
    cfg = FORMATS.get(fmt, FORMATS["mp4"])
    p = pd(name) / f"converted{cfg['ext']}"
    if not p.exists(): return ("", 404)
    return send_file(str(p), mimetype=cfg["mime"],
                     as_attachment=True, attachment_filename=f"{name}{cfg['ext']}")

def _pipeline_convert(name, fmt, res):
    d = pd(name)
    try:
        inp = _input_video(name)
        cfg = FORMATS[fmt]
        out = d / f"converted{cfg['ext']}"

        vf_parts = []
        if PRESETS[res]:
            w, h = PRESETS[res]["w"], PRESETS[res]["h"]
            vf_parts.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")

        cmd = [FFMPEG, "-y", "-i", str(inp)]
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        cmd += ["-c:v", cfg["codec"], "-preset", "medium", "-crf", "20",
                "-c:a", cfg["audio"], "-b:a", "192k",
                "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='ignore')

        save_state(name, stage="done", msg=f"转换完成 ({fmt} {res})")
    except Exception as e:
        save_state(name, stage="error", msg=f"转换失败: {e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# Standalone Tool Workspace
# ═══════════════════════════════════════════════════════════════════
TOOL_DIR = BASE / "tool_workspace"
TOOL_DIR.mkdir(exist_ok=True)

def _tool_session(sid):
    d = TOOL_DIR / sid
    if not d.exists():
        raise FileNotFoundError("会话不存在")
    return d

def _tool_input_video(sid):
    d = _tool_session(sid)
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
        p = d / f"input{ext}"
        if p.exists(): return p
    for f in d.glob("input.*"):
        if f.suffix in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".wmv"):
            return f
    raise FileNotFoundError("未找到视频文件")

def _tool_state_path(sid):
    return TOOL_DIR / sid / "state.json"

def _tool_load_state(sid):
    p = _tool_state_path(sid)
    return json.loads(p.read_text("utf-8")) if p.exists() else {}

def _tool_save_state(sid, **kw):
    p = _tool_state_path(sid)
    st = _tool_load_state(sid)
    st.update(kw)
    p.write_text(json.dumps(st, ensure_ascii=False), "utf-8")

@app.route("/api/tool/upload", methods=["POST"])
def api_tool_upload():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "请上传视频文件"}), 400
    sid = uuid.uuid4().hex[:8]
    d = TOOL_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".mp4"
    video_path = d / f"input{ext}"
    file.save(str(video_path))
    w, h, dur = _get_video_info(str(video_path))
    _tool_save_state(sid, filename=file.filename, width=w, height=h, duration=dur, stage="ready")
    return jsonify({"session_id": sid, "filename": file.filename,
                     "width": w, "height": h, "duration": round(dur, 2)})

@app.route("/api/tool/<sid>/video")
def api_tool_video(sid):
    try:
        p = _tool_input_video(sid)
    except FileNotFoundError:
        return ("", 404)
    return send_file(str(p), mimetype="video/mp4")

@app.route("/api/tool/<sid>/state")
def api_tool_state(sid):
    return jsonify(_tool_load_state(sid))

@app.route("/api/tool/<sid>/edit/delete", methods=["POST"])
def api_tool_edit_delete(sid):
    ranges_text = request.json.get("ranges", "")
    ranges = []
    for part in ranges_text.replace("；", ";").split(";"):
        part = part.strip()
        if not part: continue
        m = re.match(r"([\d:.]+)\s*[-~]\s*([\d:.]+)", part)
        if not m:
            return jsonify({"error": f"格式错误: {part}，请用 00:01:20-00:02:30 格式"}), 400
        ranges.append((_parse_ts(m.group(1)), _parse_ts(m.group(2))))
    if not ranges:
        return jsonify({"error": "请输入要删除的时间段"}), 400
    _tool_save_state(sid, stage="processing", msg="视频裁剪中...")
    threading.Thread(target=_pipeline_tool_ve_delete, args=(sid, ranges), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/edit/insert-video", methods=["POST"])
def api_tool_edit_insert_video(sid):
    pos = request.json.get("position", 0)
    file = request.files.get("segment")
    if not file:
        return jsonify({"error": "请上传插入的视频"}), 400
    d = _tool_session(sid)
    file.save(str(d / "insert_segment.mp4"))
    _tool_save_state(sid, stage="processing", msg="视频插入中...")
    threading.Thread(target=_pipeline_tool_ve_insert, args=(sid, pos, "video"), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/edit/insert-images", methods=["POST"])
def api_tool_edit_insert_images(sid):
    pos = float(request.form.get("position", 0))
    subtitles = json.loads(request.form.get("subtitles", "[]"))
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "请上传图片"}), 400
    d = _tool_session(sid)
    img_dir = d / "insert_images"
    img_dir.mkdir(exist_ok=True)
    for i, f in enumerate(files):
        f.save(str(img_dir / f"img_{i:03d}{Path(f.filename).suffix}"))
    _tool_save_state(sid, stage="processing", msg="图片视频生成中...")
    threading.Thread(target=_pipeline_ve_insert_images,
                     args=(None, pos, subtitles, False, sid), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/convert", methods=["POST"])
def api_tool_convert(sid):
    fmt = request.json.get("format", "mp4")
    res = request.json.get("resolution", "original")
    if fmt not in FORMATS:
        return jsonify({"error": f"不支持的格式: {fmt}"}), 400
    _tool_save_state(sid, stage="processing", msg=f"转换为 {fmt}...")
    threading.Thread(target=_pipeline_tool_convert, args=(sid, fmt, res), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/edit/speedup", methods=["POST"])
def api_tool_edit_speedup(sid):
    data = request.json or {}
    try:
        start = float(data.get("start", 0))
        end = float(data.get("end", 0))
        rate = float(data.get("rate", 2))
    except (ValueError, TypeError):
        return jsonify({"error": "参数格式错误"}), 400
    if end <= start or rate <= 1:
        return jsonify({"error": "结束时间需大于起始时间，倍速需大于1"}), 400
    _tool_save_state(sid, stage="processing", msg="视频变速中...")
    threading.Thread(target=_pipeline_tool_speedup, args=(sid, start, end, rate), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/edit/speedup-merge", methods=["POST"])
def api_tool_edit_speedup_merge(sid):
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "请上传录音"}), 400
    d = _tool_session(sid)
    if not (d / "speedup_meta.json").exists():
        return jsonify({"error": "请先执行变速操作"}), 400
    ext = Path(audio.filename).suffix or ".webm"
    audio_path = d / f"speedup_audio{ext}"
    audio.save(str(audio_path))
    _tool_save_state(sid, stage="processing", msg="合成视频中...")
    threading.Thread(target=_pipeline_tool_speedup_merge, args=(sid, str(audio_path)), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/edit/replace-audio", methods=["POST"])
def api_tool_edit_replace_audio(sid):
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "请上传音频"}), 400
    try:
        start = float(request.form.get("start", "0"))
        end = float(request.form.get("end", "0"))
    except (ValueError, TypeError):
        return jsonify({"error": "时间段格式错误"}), 400
    if end <= start:
        return jsonify({"error": "结束时间需大于起始时间"}), 400
    d = _tool_session(sid)
    ext = Path(audio.filename).suffix or ".webm"
    audio_path = d / f"replace_audio{ext}"
    audio.save(str(audio_path))
    _tool_save_state(sid, stage="processing", msg="替换音频中...")
    threading.Thread(target=_pipeline_tool_replace_audio,
                     args=(sid, str(audio_path), start, end), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/tool/<sid>/result")
def api_tool_result(sid):
    d = _tool_session(sid)
    cfg = _tool_load_state(sid)
    result_name = cfg.get("result_file")
    if result_name:
        p = d / result_name
        if p.exists():
            mime = "video/mp4"
            for fmt, fc in FORMATS.items():
                if p.suffix == fc["ext"]:
                    mime = fc["mime"]
            return send_file(str(p), mimetype=mime)
    return ("", 404)

@app.route("/api/tool/<sid>/download")
def api_tool_download(sid):
    d = _tool_session(sid)
    cfg = _tool_load_state(sid)
    result_name = cfg.get("result_file")
    if not result_name:
        return ("", 404)
    p = d / result_name
    if not p.exists():
        return ("", 404)
    mime = "video/mp4"
    for fmt, fc in FORMATS.items():
        if p.suffix == fc["ext"]:
            mime = fc["mime"]
    fname = cfg.get("filename", "video")
    base = Path(fname).stem
    return send_file(str(p), mimetype=mime, as_attachment=True,
                     attachment_filename=f"{base}_output{p.suffix}")

@app.route("/api/tool/<sid>", methods=["DELETE"])
def api_tool_delete(sid):
    d = _tool_session(sid)
    if d.exists():
        shutil.rmtree(str(d), ignore_errors=True)
    return jsonify({"ok": True})

def _parse_ts(s):
    """Parse timestamp like 01:20.5 or 00:01:20.5 to seconds."""
    parts = s.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)

def _pipeline_tool_ve_delete(sid, ranges):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        probe = subprocess.run([FFMPEG, "-i", str(inp)], capture_output=True, encoding='utf-8', errors='ignore')
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", probe.stderr or "")
        total = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 600

        keep = []
        prev_end = 0.0
        for start, end in ranges:
            if start > prev_end: keep.append((prev_end, start))
            prev_end = max(prev_end, end)
        if prev_end < total: keep.append((prev_end, total))
        if not keep: raise ValueError("删除后没有剩余内容")

        seg_files = []
        for i, (s, e) in enumerate(keep):
            seg = d / f"_seg_{i:03d}.mp4"
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(s), "-to", str(e),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", str(seg)],
                           check=True, capture_output=True, encoding='utf-8', errors='ignore')
            seg_files.append(seg)

        out = d / "result.mp4"
        _ffmpeg_concat(seg_files, str(out))
        for f in seg_files: f.unlink(missing_ok=True)

        # Replace input with result for chaining edits
        old = _tool_input_video(sid)
        old.unlink(missing_ok=True)
        out.rename(d / "input.mp4")

        _tool_save_state(sid, stage="done", msg="裁剪完成", result_file="input.mp4")
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"裁剪失败: {e}\n{traceback.format_exc()}")

def _pipeline_tool_ve_insert(sid, pos, mode):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        seg = str(d / "insert_segment.mp4")
        part1 = d / "_part1.mp4"
        part2 = d / "_part2.mp4"

        ow, oh, _ = _get_video_info(str(inp))
        vf = (f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
              f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black")
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(pos),
                        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(pos),
                        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        seg_enc = d / "_seg_enc.mp4"
        subprocess.run([FFMPEG, "-y", "-i", seg,
                        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                        str(seg_enc)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        _ffmpeg_concat([part1, seg_enc, part2], str(d / "_result.mp4"))
        for f in [part1, part2, seg_enc, d / "insert_segment.mp4"]: f.unlink(missing_ok=True)

        old = _tool_input_video(sid)
        old.unlink(missing_ok=True)
        (d / "_result.mp4").rename(d / "input.mp4")

        _tool_save_state(sid, stage="done", msg="插入完成", result_file="input.mp4")
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"插入失败: {e}\n{traceback.format_exc()}")

def _pipeline_tool_convert(sid, fmt, res):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        cfg = FORMATS[fmt]
        out_name = f"converted{cfg['ext']}"
        out = d / out_name

        vf_parts = []
        if PRESETS.get(res):
            w, h = PRESETS[res]["w"], PRESETS[res]["h"]
            vf_parts.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")

        cmd = [FFMPEG, "-y", "-i", str(inp)]
        if vf_parts: cmd += ["-vf", ",".join(vf_parts)]
        cmd += ["-c:v", cfg["codec"], "-preset", "medium", "-crf", "20",
                "-c:a", cfg["audio"], "-b:a", "192k", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='ignore')

        _tool_save_state(sid, stage="done", msg=f"转换完成 ({fmt})", result_file=out_name)
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"转换失败: {e}\n{traceback.format_exc()}")

def _pipeline_tool_speedup(sid, start, end, rate):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        ow, oh, total_dur = _get_video_info(str(inp))

        part1 = d / "_sp_part1.mp4"
        part2 = d / "_sp_part2.mp4"
        speed_raw = d / "_sp_raw.mp4"
        speed_up = d / "_sp_fast.mp4"
        out = d / "result.mp4"

        vf = f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"

        if start > 0:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(start),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(start), "-to", str(end),
                        "-an", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-r", "25", "-pix_fmt", "yuv420p",
                        str(speed_raw)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run([FFMPEG, "-y", "-i", str(speed_raw),
                        "-vf", f"setpts=PTS/{rate}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(speed_up)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        if end < total_dur - 0.1:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(end),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        _, _, new_seg_dur = _get_video_info(str(speed_up))

        segments = []
        if start > 0:
            segments.append(part1)
        segments.append(speed_up)
        if end < total_dur - 0.1:
            segments.append(part2)
        _ffmpeg_concat(segments, str(out))

        new_start = start
        new_end = start + new_seg_dur
        meta = {"start": start, "end": end, "rate": rate,
                "new_start": new_start, "new_end": new_end,
                "new_seg_duration": new_seg_dur}
        (d / "speedup_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        old = _tool_input_video(sid)
        old.unlink(missing_ok=True)
        out.rename(d / "input.mp4")

        for f in [part1, part2, speed_raw, speed_up]:
            f.unlink(missing_ok=True)

        _tool_save_state(sid, stage="done", msg="变速完成！请录制新音频",
                         result_file="input.mp4")
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"变速失败: {e}\n{traceback.format_exc()}")

def _pipeline_tool_speedup_merge(sid, audio_path):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        meta = json.loads((d / "speedup_meta.json").read_text("utf-8"))
        new_start = meta["new_start"]
        new_end = meta["new_end"]

        ow, oh, total_dur = _get_video_info(str(inp))

        part1 = d / "_sm_part1.mp4"
        part2 = d / "_sm_part2.mp4"
        speed_seg = d / "_sm_speed.mp4"
        merged_seg = d / "_sm_merged.mp4"
        out = d / "result.mp4"

        vf = f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"

        if new_start > 0:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(new_start),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(new_start), "-to", str(new_end),
                        "-an", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(speed_seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        if new_end < total_dur - 0.1:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(new_end),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        subprocess.run([FFMPEG, "-y", "-i", str(speed_seg), "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        str(merged_seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        segments = []
        if new_start > 0:
            segments.append(part1)
        segments.append(merged_seg)
        if new_end < total_dur - 0.1:
            segments.append(part2)
        _ffmpeg_concat(segments, str(out))

        old = _tool_input_video(sid)
        old.unlink(missing_ok=True)
        out.rename(d / "input.mp4")

        for f in [part1, part2, speed_seg, merged_seg, d / "speedup_meta.json"]:
            f.unlink(missing_ok=True)
        Path(audio_path).unlink(missing_ok=True)

        _tool_save_state(sid, stage="done", msg="合成完成",
                         result_file="input.mp4")
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"合成失败: {e}\n{traceback.format_exc()}")

def _pipeline_tool_replace_audio(sid, audio_path, start, end):
    d = _tool_session(sid)
    try:
        inp = _tool_input_video(sid)
        ow, oh, total_dur = _get_video_info(str(inp))

        part1 = d / "_ra_part1.mp4"
        part2 = d / "_ra_part2.mp4"
        seg_video = d / "_ra_seg.mp4"
        merged_seg = d / "_ra_merged.mp4"
        out = d / "result.mp4"

        vf = f"scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"

        # part1: before start
        if start > 0:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-t", str(start),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part1)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # segment video: start → end (no audio)
        subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(start), "-to", str(end),
                        "-an", "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-r", "25", "-pix_fmt", "yuv420p",
                        str(seg_video)], check=True, capture_output=True, encoding='utf-8', errors='ignore')
        # part2: after end
        if end < total_dur - 0.1:
            subprocess.run([FFMPEG, "-y", "-i", str(inp), "-ss", str(end),
                            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-r", "25", "-pix_fmt", "yuv420p",
                            str(part2)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Merge new audio with segment video
        subprocess.run([FFMPEG, "-y", "-i", str(seg_video), "-i", audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        str(merged_seg)], check=True, capture_output=True, encoding='utf-8', errors='ignore')

        # Concatenate
        segments = []
        if start > 0:
            segments.append(part1)
        segments.append(merged_seg)
        if end < total_dur - 0.1:
            segments.append(part2)
        _ffmpeg_concat(segments, str(out))

        # Replace input
        old = _tool_input_video(sid)
        old.unlink(missing_ok=True)
        out.rename(d / "input.mp4")

        # Cleanup
        for f in [part1, part2, seg_video, merged_seg, Path(audio_path)]:
            f.unlink(missing_ok=True)

        _tool_save_state(sid, stage="done", msg="音频替换完成",
                         result_file="input.mp4")
    except Exception as e:
        _tool_save_state(sid, stage="error", msg=f"替换失败: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import sys
    dev_mode = "--dev" in sys.argv
    print("Voice Studio -> http://127.0.0.1:5050")
    if dev_mode:
        print("开发模式：代码改动自动重载")
    socketio.run(app, host="127.0.0.1", port=5050, debug=dev_mode, use_reloader=dev_mode, allow_unsafe_werkzeug=True)
