def test_list_thresholds_empty(client):
    r = client.get("/thresholds/")
    assert r.status_code == 200
    assert r.json() == []

def test_set_thresholds(client):
    payload = [
        {"class_id": 0, "class_name": "robot", "threshold": 0.4},
        {"class_id": 1, "class_name": "note",  "threshold": 0.55},
    ]
    r = client.put("/thresholds/", json=payload)
    assert r.status_code == 200
    assert len(r.json()) == 2

def test_get_thresholds_after_set(client):
    client.put("/thresholds/", json=[
        {"class_id": 0, "class_name": "robot", "threshold": 0.4},
    ])
    r = client.get("/thresholds/")
    assert r.json()[0]["class_name"] == "robot"
    assert r.json()[0]["threshold"] == 0.4

def test_set_thresholds_replaces_all(client):
    client.put("/thresholds/", json=[{"class_id": 0, "class_name": "a", "threshold": 0.3}])
    client.put("/thresholds/", json=[{"class_id": 0, "class_name": "b", "threshold": 0.6}])
    r = client.get("/thresholds/")
    assert len(r.json()) == 1
    assert r.json()[0]["class_name"] == "b"
