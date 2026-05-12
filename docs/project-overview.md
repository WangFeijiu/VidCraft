# Voice Studio — Project Overview

## Purpose

Voice Studio is a local-first web application for video dubbing and voice cloning. Users upload a video, edit the auto-transcribed subtitles, select or clone a voice, and produce a final video with AI-generated narration burned in as subtitles.

## Executive Summary

| Attribute | Value |
|-----------|-------|
| Project Name | 视频处理 (Voice Studio) |
| Primary Language | Python (backend), HTML/JS (frontend) |
| Framework | Flask + Flask-SocketIO |
| Architecture | Monolithic single-process web app |
| AI Models | CosyVoice (TTS/voice cloning), faster-whisper (ASR) |
| Video Engine | FFmpeg (external binary) |
| LLM Integration | OpenAI-compatible + Anthropic APIs (text optimization, image analysis) |
| Platform | Windows 11 + NVIDIA CUDA GPU |
| Entry Point | `voice_studio/app.py` → http://127.0.0.1:5050 |

## Core Modules

### 1. Video Dubbing (Main Workflow)
Upload video → Whisper transcription → subtitle editing → LLM text optimization → voice cloning (CosyVoice) → final video composition with burned subtitles.

### 2. Image-to-Video
Upload image sequence → AI narration generation (vision LLM) → TTS audio → Ken Burns animation → final video with subtitles.

### 3. Video Editing Tools (Standalone)
Session-based video manipulation: delete segments, insert clips/images, speed up sections, replace audio, format conversion. Operations chain — each edit builds on the previous result.

### 4. Voice Library
8 preset voice styles (instruct mode) + user-saved custom voices (zero-shot cloning from uploaded samples).

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Web Framework | Flask 2.x + Flask-SocketIO | HTTP API + real-time progress updates |
| ASR | faster-whisper (large-v3) | Video transcription (CUDA accelerated) |
| TTS | CosyVoice (Fun-CosyVoice3-0.5B / CosyVoice2-0.5B) | Voice cloning and synthesis |
| Video | FFmpeg 7.1 (external binary) | All video/audio encoding, muxing, filtering |
| LLM | httpx → OpenAI/Anthropic APIs | Text optimization, image analysis, subtitle matching |
| Frontend | Single-page HTML (~4000 lines, vanilla JS) | Browser UI with WebSocket for live updates |
| GPU | NVIDIA CUDA (cuBLAS, cuDNN) | Model inference acceleration |
| Environment | Python 3.11, Windows venv | Runtime |

## Repository Structure

```
E:/视频处理/
├── voice_studio/           # Main application
│   ├── app.py              # Flask server (3614 lines) — all routes + pipelines
│   ├── llm.py              # LLM abstraction layer (OpenAI + Anthropic)
│   ├── transcribe_worker.py # Whisper subprocess worker
│   ├── gen_voice_previews.py # Batch voice preview generator
│   ├── llm_config.json     # LLM API credentials (gitignored)
│   ├── templates/index.html # Single-page frontend (~4000 lines)
│   ├── voice_cache/        # Cached voice preview audio
│   ├── custom_voices/      # User-saved voice samples
│   ├── projects/           # Per-project data (video, audio, state)
│   ├── img2vid/            # Image-to-video project data
│   └── tool_workspace/     # Standalone tool session data
├── CosyVoice/             # CosyVoice model repo (submodule/clone)
├── .venv/                 # Python virtual environment
├── .cache/huggingface/    # Model cache
├── _bmad/                 # BMad Method configuration
└── (legacy scripts)       # transcribe.py, compose_video.py, etc.
```

## Data Flow

1. **Upload** → video saved to `projects/{name}/input.mp4`
2. **Transcribe** → FFmpeg extracts audio → Whisper produces `sentences.json`
3. **Edit** → user edits text, LLM optimizes → `sentences_optimized.json`
4. **Clone** → CosyVoice generates per-sentence audio → `recordings/s_NNN_clone.webm`
5. **Compose** → FFmpeg cuts video segments, overlays audio, burns SRT → `final.mp4`

## Running the Application

```bash
cd E:/视频处理
.venv/Scripts/activate
python voice_studio/app.py        # Production
python voice_studio/app.py --dev  # Dev mode with auto-reload
```

Requires: NVIDIA GPU with CUDA, FFmpeg binary at configured path, CosyVoice model downloaded.
