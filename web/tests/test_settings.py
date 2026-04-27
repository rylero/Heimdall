import zipfile
from io import BytesIO


def test_export_returns_zip(client):
    r = client.get("/settings/export")
    assert r.status_code == 200
    assert "application/zip" in r.headers["content-type"]


def test_export_contains_db(client):
    r = client.get("/settings/export")
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        assert "heimdall.db" in zf.namelist()


def test_export_contains_engine_files(client):
    client.post("/models/upload",
        files={"file": ("x.engine", b"engine", "application/octet-stream")})
    r = client.get("/settings/export")
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        assert "models/x.engine" in zf.namelist()


def test_import_restores_tracker_config(client):
    client.put("/tracker/", json={
        "confirmation_frames": 9, "loss_frames": 5,
        "gate_distance": 1.0, "clutter_density": 1.0, "p_detection": 0.9,
    })
    export = client.get("/settings/export").content
    # Reset tracker
    client.put("/tracker/", json={
        "confirmation_frames": 3, "loss_frames": 5,
        "gate_distance": 1.0, "clutter_density": 1.0, "p_detection": 0.9,
    })
    # Import
    r = client.post("/settings/import",
        files={"file": ("settings.zip", export, "application/zip")})
    assert r.status_code == 204
    assert client.get("/tracker/").json()["confirmation_frames"] == 9


def test_import_wrong_type_rejected(client):
    r = client.post("/settings/import",
        files={"file": ("bad.txt", b"not a zip", "text/plain")})
    assert r.status_code == 400
