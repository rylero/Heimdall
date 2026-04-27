from io import BytesIO


def test_list_models_empty(client):
    r = client.get("/models/")
    assert r.status_code == 200
    assert r.json() == []


def test_upload_engine(client):
    r = client.post(
        "/models/upload",
        files={"file": ("rfdetr.engine", b"fake engine bytes", "application/octet-stream")},
    )
    assert r.status_code == 201
    assert r.json()["filename"] == "rfdetr.engine"


def test_upload_wrong_extension_rejected(client):
    r = client.post(
        "/models/upload",
        files={"file": ("model.pt", b"data", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_list_models_after_upload(client):
    client.post("/models/upload",
        files={"file": ("a.engine", b"x", "application/octet-stream")})
    client.post("/models/upload",
        files={"file": ("b.engine", b"x", "application/octet-stream")})
    r = client.get("/models/")
    assert sorted(r.json()) == ["a.engine", "b.engine"]


def test_get_active_model_defaults(client):
    r = client.get("/models/active")
    assert r.status_code == 200
    assert r.json()["engine_filename"] is None
    assert r.json()["labels"] == []


def test_set_active_model(client):
    client.post("/models/upload",
        files={"file": ("model.engine", b"data", "application/octet-stream")})
    r = client.put("/models/active", json={
        "engine_filename": "model.engine",
        "num_classes": 3,
        "labels": ["robot", "note", "amp"],
    })
    assert r.status_code == 200
    r2 = client.get("/models/active")
    assert r2.json()["engine_filename"] == "model.engine"
    assert r2.json()["labels"] == ["robot", "note", "amp"]
    assert r2.json()["num_classes"] == 3


def test_set_active_nonexistent_engine_rejected(client):
    r = client.put("/models/active", json={
        "engine_filename": "ghost.engine", "num_classes": 1, "labels": [],
    })
    assert r.status_code == 404


def test_delete_model(client):
    client.post("/models/upload",
        files={"file": ("del.engine", b"x", "application/octet-stream")})
    r = client.delete("/models/del.engine")
    assert r.status_code == 204
    assert client.get("/models/").json() == []
