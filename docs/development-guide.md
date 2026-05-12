# Voice Studio — Development Guide

## Prerequisites

- **OS**: Windows 11 (hardcoded paths assume Windows)
- **Python**: 3.11 (in `.venv`)
- **GPU**: NVIDIA with CUDA support (cuBLAS, cuDNN)
- **FFmpeg**: v7.1 binary at `D:/Tech/program/python/Lib/site-packages/imageio_ffmpeg/binaries/`
- **CosyVoice**: Cloned to `E:/视频处理/CosyVoice/` with pretrained models downloaded
- **Disk**: ~10GB for models (Whisper large-v3 + CosyVoice3-0.5B)

## Environment Setup

```bash
cd E:/视频处理
python -m venv .venv
.venv/Scripts/activate

# Core dependencies (inferred from imports)
pip install flask flask-socketio httpx faster-whisper
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy soundfile Pillow

# CosyVoice dependencies
cd CosyVoice
pip install -r requirements.txt
```

## Running

```bash
# Production mode
python voice_studio/app.py
# → http://127.0.0.1:5050

# Development mode (auto-reload on code changes)
python voice_studio/app.py --dev
```

## Configuration

### LLM Config (`voice_studio/llm_config.json`)
```json
{
  "configs": [
    {
      "name": "配置名称",
      "provider": "openai",
      "api_key": "sk-...",
      "model": "gpt-4o-mini",
      "base_url": "https://api.openai.com/v1"
    }
  ],
  "active_idx": 0
}
```

Supports: OpenAI, Anthropic, Kimi, and any OpenAI-compatible endpoint.

### Key Paths (hardcoded in `app.py`)
| Variable | Path | Purpose |
|----------|------|---------|
| `FFMPEG` | `D:/Tech/.../ffmpeg-win-x86_64-v7.1.exe` | FFmpeg binary |
| `HF_CACHE` | `E:/视频处理/.cache/huggingface` | Whisper model cache |
| `COSYVOICE_DIR` | `E:/视频处理/CosyVoice` | CosyVoice repo root |

## Architecture Notes for Developers

### Adding a New API Endpoint
All routes are in `app.py`. Pattern:
1. Add `@app.route(...)` handler
2. For async work: spawn `threading.Thread(target=_pipeline_..., daemon=True)`
3. Push progress via `socketio.emit('event_name', data, namespace='/')`
4. Update state via `save_state(name, stage=..., msg=...)`

### Adding a New Voice Style
Add entry to `VOICE_LIBRARY` list in `app.py`:
```python
{"id": "unique_id", "name": "显示名", "desc": "描述",
 "instruct": "CosyVoice instruct prompt"}
```

### Frontend Changes
Edit `voice_studio/templates/index.html`. The frontend is a single file with:
- CSS at the top
- HTML structure in the middle
- JavaScript at the bottom
- WebSocket event handlers for real-time updates

### State Machine Transitions
Projects follow: `new` → `processing` → `editing` → `recording` → `cloning` → `composing` → `done`

Any stage can transition to `error`. The frontend polls `/api/project/<name>` and listens to WebSocket `project_update` events.

## Testing

No automated test suite exists. Manual testing workflow:
1. Start the server in dev mode
2. Create a project with a short video (~30s)
3. Verify transcription completes
4. Test subtitle editing and optimization
5. Test voice cloning with a preset voice
6. Verify final video composition

## Common Issues

- **First clone sentence is silent**: Model warmup should handle this. If it persists, check `_load_cosyvoice_model_cached()`.
- **FFmpeg path not found**: Update `FFMPEG` constant in `app.py`.
- **CUDA out of memory**: Whisper large-v3 + CosyVoice together need ~8GB VRAM.
- **WebSocket disconnects**: `ping_timeout=600` is set high; check network/proxy settings.
