"""Tests for sentence management endpoints."""
import io
import json


def _create_project(client, name: str, sample_bytes: bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": name})


def test_save_and_get_sentences(client, sample_video_bytes):
    _create_project(client, "sent_test", sample_video_bytes)

    sentences = [
        {"text": "第一句", "start": 0.0, "end": 2.5},
        {"text": "第二句", "start": 2.5, "end": 5.0},
    ]
    resp = client.put("/api/project/sent_test/sentences",
                      json={"version": "original", "sentences": sentences})
    assert resp.status_code == 200

    resp = client.get("/api/project/sent_test/sentences")
    assert resp.status_code == 200
    data = resp.json()
    assert "sentences" in data
    assert len(data["sentences"]) == 2


def test_delete_and_restore_sentence(client, sample_video_bytes):
    _create_project(client, "del_test", sample_video_bytes)

    resp = client.post("/api/project/del_test/delete-sentence/3")
    assert resp.status_code == 200

    resp = client.get("/api/project/del_test/deleted-sentences")
    assert 3 in resp.json()["deleted"]

    resp = client.post("/api/project/del_test/restore-sentence/3")
    assert resp.status_code == 200

    resp = client.get("/api/project/del_test/deleted-sentences")
    assert 3 not in resp.json()["deleted"]


def test_export_srt(client, sample_video_bytes):
    _create_project(client, "export_test", sample_video_bytes)
    sentences = [{"text": "测试句子", "start": 0.0, "end": 1.5}]
    client.put("/api/project/export_test/sentences",
              json={"version": "original", "sentences": sentences})

    resp = client.get("/api/project/export_test/export?format=srt")
    assert resp.status_code == 200
    content = resp.text
    assert "测试句子" in content
    assert "-->" in content
