def test_get_tracker_defaults(client):
    r = client.get("/tracker/")
    assert r.status_code == 200
    d = r.json()
    assert d["confirmation_frames"] == 3
    assert d["loss_frames"] == 5
    assert d["gate_distance"] == 1.0
    assert d["clutter_density"] == 1.0
    assert d["p_detection"] == 0.9

def test_update_tracker(client):
    payload = {
        "confirmation_frames": 5,
        "loss_frames": 10,
        "gate_distance": 2.5,
        "clutter_density": 0.5,
        "p_detection": 0.85,
    }
    r = client.put("/tracker/", json=payload)
    assert r.status_code == 200
    assert r.json()["confirmation_frames"] == 5
    assert r.json()["gate_distance"] == 2.5

def test_update_tracker_persists(client):
    client.put("/tracker/", json={
        "confirmation_frames": 7, "loss_frames": 5,
        "gate_distance": 1.0, "clutter_density": 1.0, "p_detection": 0.9,
    })
    r = client.get("/tracker/")
    assert r.json()["confirmation_frames"] == 7
