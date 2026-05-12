# Voice Studio — Source Tree Analysis

## Critical Directories

```
E:/视频处理/
├── voice_studio/                    # [MAIN APP] All application code
│   ├── app.py                       # [ENTRY] Flask server, all routes & pipelines (3614 LOC)
│   ├── llm.py                       # [CORE] LLM abstraction (OpenAI + Anthropic) (302 LOC)
│   ├── transcribe_worker.py         # [WORKER] Whisper subprocess (46 LOC)
│   ├── gen_voice_previews.py        # [UTIL] Batch voice preview generation
│   ├── llm_config.json              # [CONFIG] API keys (gitignored content)
│   ├── templates/
│   │   └── index.html               # [FRONTEND] Single-page app (4003 LOC)
│   ├── voice_cache/                 # [CACHE] Pre-generated voice style previews (.webm)
│   ├── custom_voices/               # [DATA] User-saved voice samples + metadata
│   │   └── custom_<hash>/
│   │       ├── sample.wav           # 24kHz mono WAV for CosyVoice
│   │       ├── preview.webm         # Playback preview
│   │       └── meta.json            # Voice name, creation date
│   ├── projects/                    # [DATA] Video dubbing projects
│   │   └── <project-name>/
│   │       ├── input.mp4            # Original uploaded video
│   │       ├── audio_16k.wav        # Extracted audio for Whisper
│   │       ├── state.json           # Project state machine
│   │       ├── sentences.json       # Original transcription
│   │       ├── sentences_optimized.json  # LLM-polished version
│   │       ├── sentences_uploaded.json   # User-uploaded subtitles
│   │       ├── voice_sample.wav     # Voice sample for zero-shot cloning
│   │       ├── recordings/          # Per-sentence audio
│   │       │   ├── s_001.webm       # Manual recording or accepted clone
│   │       │   └── s_001_clone.webm # AI-generated clone
│   │       ├── clips/               # Cached video clips per sentence
│   │       ├── final.srt            # Generated subtitle file
│   │       └── final.mp4            # Composed output video
│   ├── img2vid/                     # [DATA] Image-to-video projects
│   │   └── <project-name>/
│   │       ├── images/              # Uploaded images (img_000.png, ...)
│   │       ├── recordings/          # Generated TTS audio per segment
│   │       ├── narration.json       # AI-generated narration text
│   │       ├── voice_sample.wav     # Optional voice sample
│   │       ├── state.json           # Project state
│   │       ├── final.srt            # Generated subtitles
│   │       └── final.mp4            # Output video
│   └── tool_workspace/              # [DATA] Standalone tool sessions
│       └── <session-id>/
│           ├── input.mp4            # Uploaded source video
│           ├── result.mp4           # Latest edit result (chains)
│           ├── state.json           # Session state
│           └── _frames/             # Extracted keyframes (for UI)
├── CosyVoice/                       # [EXTERNAL] CosyVoice model repository
│   ├── pretrained_models/           # Downloaded model weights
│   │   └── Fun-CosyVoice3-0.5B/    # Primary model
│   ├── asset/                       # Reference audio files
│   │   └── zero_shot_prompt.wav     # Default voice reference
│   └── cosyvoice/cli/cosyvoice.py  # AutoModel entry point
├── .venv/                           # [ENV] Python virtual environment
├── .cache/huggingface/              # [CACHE] Whisper model weights
├── _bmad/                           # [TOOLING] BMad Method config
├── docs/                            # [DOCS] Project documentation (this folder)
└── (legacy scripts)
    ├── transcribe.py                # Standalone transcription script
    ├── compose_video.py             # Standalone video composition
    ├── tts_synthesize.py            # Standalone TTS
    ├── merge_srt.py                 # SRT manipulation
    └── server.py                    # Legacy server (superseded by voice_studio/)
```

## Key Files by Function

### Application Logic
- `voice_studio/app.py` — The entire backend: routes, pipelines, state management, FFmpeg orchestration
- `voice_studio/llm.py` — Multi-provider LLM client with retry, masking, and config management

### Frontend
- `voice_studio/templates/index.html` — Complete SPA: project list, editor, recorder, player, tools

### Data Formats
- `state.json` — JSON state machine per project (stage, progress, settings)
- `sentences.json` — Array of `{text, start, end}` objects
- `narration.json` — Array of `{image_idx, narration, analysis}` objects
- `meta.json` — Custom voice metadata `{id, name, desc, prompt_text, created_at}`
