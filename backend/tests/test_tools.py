"""Tests for tool workspace endpoints."""
import io


def test_list_tools_empty(client):
    resp = client.get("/api/tool/list")
    assert resp.status_code == 200
    assert resp.json() == []


def test_tool_upload(client):
    files = {"video": ("test.mp4", io.BytesIO(b"FAKE"), "video/mp4")}
    resp = client.post("/api/tool/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "sid" in data
    assert data["filename"] == "test.mp4"


def test_tool_state(client):
    files = {"video": ("test.mp4", io.BytesIO(b"FAKE"), "video/mp4")}
    sid = client.post("/api/tool/upload", files=files).json()["sid"]

    resp = client.get(f"/api/tool/{sid}/state")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "ready"


def test_tool_delete(client):
    files = {"video": ("test.mp4", io.BytesIO(b"FAKE"), "video/mp4")}
    sid = client.post("/api/tool/upload", files=files).json()["sid"]

    resp = client.delete(f"/api/tool/{sid}")
    assert resp.status_code == 200

    listing = client.get("/api/tool/list").json()
    assert all(s["sid"] != sid for s in listing)
