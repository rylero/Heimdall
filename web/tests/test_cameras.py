def test_list_cameras_empty(client):
    r = client.get("/cameras/")
    assert r.status_code == 200
    assert r.json() == []

def test_create_camera(client):
    r = client.post("/cameras/", json={"name": "Front", "device": "/dev/video0"})
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == 1
    assert data["name"] == "Front"
    assert data["device"] == "/dev/video0"
    assert data["camera_type"] == "usb"
    assert data["fx"] == 500.0

def test_create_sets_all_fields(client):
    payload = {
        "name": "Side", "device": "/dev/video1", "camera_type": "csi",
        "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
        "tx": 0.3, "ty": 0.0, "tz": 0.6,
        "yaw_deg": 0.0, "pitch_deg": 28.0, "roll_deg": 0.0,
        "exposure_us": 5000, "gain": 1.5,
    }
    r = client.post("/cameras/", json=payload)
    assert r.status_code == 201
    d = r.json()
    assert d["fx"] == 600.0
    assert d["pitch_deg"] == 28.0
    assert d["exposure_us"] == 5000

def test_update_camera(client):
    client.post("/cameras/", json={"name": "Front"})
    r = client.put("/cameras/1", json={"name": "Rear", "device": "/dev/video1"})
    assert r.status_code == 200
    assert r.json()["name"] == "Rear"
    assert r.json()["device"] == "/dev/video1"

def test_update_nonexistent_returns_404(client):
    r = client.put("/cameras/999", json={"name": "Ghost"})
    assert r.status_code == 404

def test_delete_camera(client):
    client.post("/cameras/", json={"name": "Front"})
    r = client.delete("/cameras/1")
    assert r.status_code == 204
    assert client.get("/cameras/").json() == []

def test_list_multiple_cameras(client):
    client.post("/cameras/", json={"name": "A"})
    client.post("/cameras/", json={"name": "B"})
    r = client.get("/cameras/")
    assert len(r.json()) == 2
    assert r.json()[0]["name"] == "A"
