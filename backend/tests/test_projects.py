"""Tests for project management endpoints."""
import io


def test_list_projects_empty(client):
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_project(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    data = {"name": "test_project"}
    resp = client.post("/api/projects", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["name"] == "test_project"


def test_create_project_duplicate(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    data = {"name": "dup_project"}
    resp1 = client.post("/api/projects", files=files, data=data)
    assert resp1.status_code == 200

    files2 = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    resp2 = client.post("/api/projects", files=files2, data=data)
    assert resp2.status_code == 400


def test_create_project_no_name(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    data = {"name": "   "}
    resp = client.post("/api/projects", files=files, data=data)
    assert resp.status_code == 400


def test_project_status(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "status_test"})

    resp = client.get("/api/project/status_test")
    assert resp.status_code == 200
    data = resp.json()
    assert "stage" in data
    assert "recorded" in data


def test_delete_project(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "to_delete"})

    resp = client.delete("/api/project/to_delete")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp2 = client.get("/api/projects")
    assert all(p["name"] != "to_delete" for p in resp2.json())


def test_set_stage(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "stage_test"})

    resp = client.put("/api/project/stage_test/stage", json={"stage": "editing"})
    assert resp.status_code == 200

    status = client.get("/api/project/stage_test").json()
    assert status["stage"] == "editing"


def test_has_video(client, sample_video_bytes):
    files = {"video": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}
    client.post("/api/projects", files=files, data={"name": "has_video_test"})

    resp = client.get("/api/project/has_video_test/has-video")
    assert resp.status_code == 200
    assert resp.json()["has"] is True
