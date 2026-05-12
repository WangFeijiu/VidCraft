"""Tests for image-to-video endpoints."""
import io


def test_list_i2v_empty(client):
    resp = client.get("/api/img2vid")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_i2v_project(client):
    files = [
        ("image0", ("img1.jpg", io.BytesIO(b"FAKE_IMG_1"), "image/jpeg")),
        ("image1", ("img2.jpg", io.BytesIO(b"FAKE_IMG_2"), "image/jpeg")),
    ]
    data = {"name": "i2v_test", "theme": "科技未来"}
    resp = client.post("/api/img2vid", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["name"] == "i2v_test"
    assert resp.json()["image_count"] == 2


def test_i2v_theme_update(client):
    files = [("image0", ("img.jpg", io.BytesIO(b"FAKE"), "image/jpeg"))]
    client.post("/api/img2vid", files=files, data={"name": "theme_test", "theme": "旧主题"})

    resp = client.post("/api/img2vid/theme_test/theme", json={"theme": "新主题"})
    assert resp.status_code == 200

    status = client.get("/api/img2vid/theme_test").json()
    assert status["theme"] == "新主题"


def test_i2v_delete(client):
    files = [("image0", ("img.jpg", io.BytesIO(b"FAKE"), "image/jpeg"))]
    client.post("/api/img2vid", files=files, data={"name": "to_del", "theme": ""})

    resp = client.delete("/api/img2vid/to_del")
    assert resp.status_code == 200

    resp = client.get("/api/img2vid")
    assert all(p["name"] != "to_del" for p in resp.json())
