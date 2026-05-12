# Voice Studio — Architecture

## System Architecture

Voice Studio is a **monolithic Flask application** running as a single process on localhost. It combines:
- An HTTP/WebSocket server (Flask + SocketIO)
- Background processing threads for long-running pipelines
- Direct GPU model inference (CosyVoice, Whisper) in-process
- FFmpeg subprocess calls for all video/audio encoding

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser (SPA)                           │
│  Single HTML page with vanilla JS + WebSocket client        │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST + WebSocket (SocketIO)
┌──────────────────────▼──────────────────────────────────────┐
│                   Flask + SocketIO                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Project API │  │ Img2Vid API  │  │ Tool Workspace API│  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                │                    │             │
│  ┌──────▼────────────────▼────────────────────▼──────────┐  │
│  │              Background Thread Pipelines               │  │
│  │  _pipeline_transcribe  _pipeline_voice_clone           │  │
│  │  _pipeline_compose     _pipeline_optimize              │  │
│  │  _pipeline_i2v_*       _pipeline_tool_*                │  │
│  └──────┬────────────────┬────────────────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
┌─────────▼───┐  ┌────────▼────────┐  ┌───────▼──────┐
│   FFmpeg    │  │  CosyVoice GPU  │  │ Whisper GPU  │
│ (subprocess)│  │  (in-process)   │  │ (subprocess) │
└─────────────┘  └─────────────────┘  └──────────────┘
```

## Key Design Decisions

### 1. Single-Process Monolith
All functionality lives in one Flask process. GPU models are loaded once and cached globally (`_COSYVOICE_MODEL_INSTANCE`). This avoids IPC complexity but means the app is single-tenant.

### 2. Thread-Based Concurrency
Long-running operations (transcription, cloning, composition) run in daemon threads. Progress is pushed to the browser via SocketIO events. No task queue (Celery/RQ) — simplicity over scalability.

### 3. File-Based State
Each project stores state in `state.json` alongside its media files. No database. State transitions drive the UI (stages: `new` → `processing` → `editing` → `recording` → `cloning` → `composing` → `done`).

### 4. Whisper in Subprocess
Transcription runs in a separate Python process (`transcribe_worker.py`) to avoid blocking the Flask event loop. The main process polls `state.json` and relays progress via WebSocket.

### 5. CosyVoice Model Warmup
The first inference after model load produces poor audio (near-silent). A warmup inference is run on load to prime the model before any user request.

### 6. LLM Abstraction Layer
`llm.py` provides a unified interface for OpenAI-compatible and Anthropic APIs. Supports multiple saved configurations, API key masking, and automatic retry with exponential backoff.

## API Surface

### Project Lifecycle
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create project (upload video) |
| DELETE | `/api/project/<name>` | Delete project |
| GET | `/api/project/<name>` | Get project status |
| PUT | `/api/project/<name>/stage` | Transition stage |

### Subtitle Management
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/project/<name>/sentences` | Get sentences (with version info) |
| PUT | `/api/project/<name>/sentences` | Save edited sentences |
| POST | `/api/project/<name>/optimize` | LLM text optimization |
| POST | `/api/project/<name>/match-subtitles` | Match uploaded subtitles to timestamps |
| GET | `/api/project/<name>/export` | Export as SRT/TXT/JSON |

### Voice Cloning
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/project/<name>/voice-clone` | Start cloning pipeline |
| POST | `/api/project/<name>/cancel-clone` | Cancel in-progress clone |
| POST | `/api/project/<name>/resume-clone` | Resume stalled clone |
| POST | `/api/project/<name>/regenerate-clone/<idx>` | Regenerate single sentence |
| POST | `/api/project/<name>/accept-clone/<idx>` | Accept clone for sentence |
| POST | `/api/project/<name>/accept-all-clones` | Accept all clones |
| GET | `/api/voices` | List all voices (preset + custom) |

### Image-to-Video
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/img2vid` | Create i2v project |
| POST | `/api/img2vid/<name>/analyze` | AI image analysis + narration |
| POST | `/api/img2vid/<name>/preview-audio` | Generate TTS per segment |
| POST | `/api/img2vid/<name>/generate` | Compose final video |

### Standalone Tools
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/tool/upload` | Upload video for editing |
| POST | `/api/tool/<sid>/edit/delete` | Delete time ranges |
| POST | `/api/tool/<sid>/edit/insert-video` | Insert video segment |
| POST | `/api/tool/<sid>/edit/concat` | Concatenate videos |
| POST | `/api/tool/<sid>/edit/replace-audio` | Replace audio in range |
| POST | `/api/tool/<sid>/convert` | Format/resolution conversion |

## State Machine

### Main Project Stages
```
new → processing → editing → recording → cloning → composing → done
                                                              ↗
                                          (manual record) ───┘
```

### Image-to-Video Stages
```
uploading → analyzing → narration_ready → audio_preview → generating → done
```

## External Dependencies

| Dependency | Location | Purpose |
|-----------|----------|---------|
| FFmpeg 7.1 | `D:/Tech/program/python/Lib/site-packages/imageio_ffmpeg/binaries/` | Video processing |
| CosyVoice | `E:/视频处理/CosyVoice/` | Voice synthesis models |
| HuggingFace cache | `E:/视频处理/.cache/huggingface/` | Whisper model weights |
| CUDA DLLs | `.venv/Lib/site-packages/nvidia/` | GPU acceleration |
