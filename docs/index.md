# Voice Studio — Documentation Index

## Project Overview

- **Type:** Monolith (single cohesive codebase)
- **Primary Language:** Python 3.11
- **Architecture:** Flask web app with GPU-accelerated AI pipelines
- **Entry Point:** `voice_studio/app.py` → http://127.0.0.1:5050

## Quick Reference

- **Tech Stack:** Flask + SocketIO, CosyVoice TTS, faster-whisper ASR, FFmpeg
- **Platform:** Windows 11 + NVIDIA CUDA
- **Frontend:** Single-page HTML/JS (~4000 lines)
- **Backend:** Single Python file (~3600 lines) + LLM abstraction

## Generated Documentation

- [Project Overview](./project-overview.md) — Executive summary, tech stack, data flow
- [Architecture](./architecture.md) — System design, API surface, state machines, dependencies
- [Source Tree Analysis](./source-tree-analysis.md) — Annotated directory structure, key files
- [Development Guide](./development-guide.md) — Setup, running, configuration, common issues
- [API Contracts](./api-contracts.md) — Complete REST + WebSocket API reference

## Getting Started

1. Ensure prerequisites: Python 3.11, NVIDIA GPU + CUDA, FFmpeg, CosyVoice models
2. Activate venv: `.venv/Scripts/activate`
3. Run: `python voice_studio/app.py`
4. Open: http://127.0.0.1:5050
5. Create a project by uploading a video
