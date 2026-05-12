# Voice Studio — API Contracts

## Base URL
`http://127.0.0.1:5050`

## Real-Time Events (WebSocket / SocketIO)

| Event | Direction | Payload | Purpose |
|-------|-----------|---------|---------|
| `project_update` | Server→Client | `{project, state}` | State change notification |
| `optimize_progress` | Server→Client | `{project, step, msg, progress}` | LLM optimization progress |
| `match_progress` | Server→Client | `{project, msg, progress}` | Subtitle matching progress |
| `clone_progress` | Server→Client | `{project, progress}` | Voice cloning progress |

## REST API Endpoints

### Projects

#### `GET /api/projects`
List all dubbing projects.
```json
[{"name": "flowbrain", "stage": "done", "msg": "完成！"}]
```

#### `POST /api/projects`
Create project. `multipart/form-data`: `name` (string), `video` (file).
```json
{"name": "flowbrain"}
```

#### `DELETE /api/project/<name>`
Delete project and all data.

#### `GET /api/project/<name>`
Get project status including `stage`, `msg`, `recorded` count.

#### `PUT /api/project/<name>/stage`
Set stage. Body: `{"stage": "editing"|"recording", "version": "original"|"optimized"|"uploaded"}`

---

### Sentences

#### `GET /api/project/<name>/sentences`
Returns active sentences with version metadata.
```json
{
  "active": "optimized",
  "versions": {"original": [...], "optimized": [...]},
  "sentences": [{"text": "...", "start": 0.0, "end": 2.5}]
}
```

#### `PUT /api/project/<name>/sentences`
Save sentences. Body: `{"version": "original", "sentences": [...], "clear_after": false}`

#### `POST /api/project/<name>/optimize`
Start async LLM optimization. Body: `{"version": "original", "description": ""}`

#### `POST /api/project/<name>/match-subtitles`
Match uploaded text to original timestamps. Body: `{"subtitles": ["line1", "line2", ...]}`

#### `GET /api/project/<name>/export?format=srt&version=optimized`
Export subtitles as SRT, TXT, or JSON.

---

### Recordings

#### `POST /api/project/<name>/record/<idx>`
Upload manual recording. `multipart/form-data`: `audio` (file).

#### `GET /api/project/<name>/record/<idx>`
Stream recording audio (webm).

#### `DELETE /api/project/<name>/record/<idx>`
Delete recording for sentence.

#### `GET /api/project/<name>/recorded`
List indices with manual recordings.

#### `GET /api/project/<name>/cloned`
List indices with clone audio.

---

### Voice Cloning

#### `POST /api/project/<name>/voice-clone`
Start cloning. `multipart/form-data`: `voice_id`, `prompt_text`, optional `sample` (file).

#### `POST /api/project/<name>/cancel-clone`
Cancel in-progress clone job.

#### `POST /api/project/<name>/resume-clone`
Resume stalled clone from last completed sentence.

#### `POST /api/project/<name>/regenerate-clone/<idx>`
Regenerate single sentence clone. Body: `{"prompt_text": "..."}`

#### `POST /api/project/<name>/accept-clone/<idx>`
Copy clone audio as the accepted recording.

#### `POST /api/project/<name>/accept-all-clones`
Accept all clone recordings at once.

---

### Voice Library

#### `GET /api/voices`
List all voices (8 presets + custom).

#### `POST /api/custom-voices`
Save custom voice. `multipart/form-data`: `name`, `prompt_text`, `sample` (file).

#### `DELETE /api/custom-voices/<voice_id>`
Delete custom voice.

---

### Video Playback & Download

#### `GET /api/project/<name>/video`
Stream original input video.

#### `GET /api/project/<name>/final-video`
Stream composed final video.

#### `GET /api/project/<name>/download`
Download final video (or input if no final).

#### `GET /api/project/<name>/sentence-clip/<idx>`
Stream video clip for a single sentence.

---

### Image-to-Video

#### `POST /api/img2vid`
Create project. `multipart/form-data`: `name`, `theme`, image files.

#### `POST /api/img2vid/<name>/analyze`
AI image analysis + narration generation. Body: `{"style": "documentary"}`

#### `GET /api/img2vid/<name>/narration`
Get narration array.

#### `PUT /api/img2vid/<name>/narration`
Save edited narration.

#### `POST /api/img2vid/<name>/voice-sample`
Upload voice sample for TTS.

#### `POST /api/img2vid/<name>/preview-audio`
Generate TTS audio per segment.

#### `POST /api/img2vid/<name>/generate`
Compose final video. Body: `{"animate": true}`

---

### Standalone Tools

#### `POST /api/tool/upload`
Upload video, get session ID. Returns `{session_id, filename, width, height, duration}`.

#### `POST /api/tool/<sid>/edit/delete`
Delete time ranges. Body: `{"ranges": "00:10-00:15, 01:20-01:25"}`

#### `POST /api/tool/<sid>/edit/insert-video`
Insert video at position. `multipart/form-data`: `position`, `segment` (file).

#### `POST /api/tool/<sid>/edit/concat`
Concatenate videos. `multipart/form-data`: `segments` (files).

#### `POST /api/tool/<sid>/edit/replace-audio`
Replace audio in range. `multipart/form-data`: `audio`, `start`, `end`.

#### `POST /api/tool/<sid>/edit/speedup`
Speed up segment. Body: `{"start": 10, "end": 20, "rate": 2}`

#### `POST /api/tool/<sid>/convert`
Convert format. Body: `{"format": "mp4", "resolution": "720p"}`

#### `GET /api/tool/<sid>/result`
Stream latest edit result.

#### `GET /api/tool/<sid>/download`
Download result as attachment.

---

### LLM Configuration

#### `GET /api/llm-config`
Get LLM configs (keys masked).

#### `PUT /api/llm-config`
Save LLM configs.

#### `POST /api/llm-test`
Test LLM connection. Body: single config object.
