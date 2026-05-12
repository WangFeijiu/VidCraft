"""Tests for voice-related endpoints."""
import io


def test_list_voices(client):
    resp = client.get("/api/voices")
    assert resp.status_code == 200
    voices = resp.json()
    assert len(voices) >= 8
    preset_ids = {v["id"] for v in voices}
    assert "standard" in preset_ids
    assert "deep" in preset_ids
    assert "warm" in preset_ids


def test_create_custom_voice(client):
    sample_bytes = b"FAKE_AUDIO_SAMPLE"
    files = {"sample": ("voice.wav", io.BytesIO(sample_bytes), "audio/wav")}
    data = {"name": "TestVoice", "desc": "Test custom voice"}

    resp = client.post("/api/custom-voices", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    voice_id = resp.json()["voice_id"]
    assert voice_id.startswith("custom_")


def test_upload_recording(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "rec_test"})

    audio_bytes = b"FAKE_AUDIO"
    files = {"audio": ("rec.webm", io.BytesIO(audio_bytes), "audio/webm")}
    resp = client.post("/api/project/rec_test/record/1", files=files)
    assert resp.status_code == 200


def test_get_recorded_list(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "list_rec_test"})

    resp = client.get("/api/project/list_rec_test/recorded")
    assert resp.status_code == 200
    assert "recorded" in resp.json()
