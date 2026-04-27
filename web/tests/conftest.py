import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import app.database as db_mod
    import app.routers.model_mgmt as mm_mod
    import app.routers.settings as st_mod

    test_db     = tmp_path / "test.db"
    test_models = tmp_path / "models"
    test_models.mkdir()

    monkeypatch.setattr(db_mod,  "DB_PATH",    test_db)
    monkeypatch.setattr(mm_mod,  "MODELS_DIR", test_models)
    monkeypatch.setattr(st_mod,  "DB_PATH",    test_db)
    monkeypatch.setattr(st_mod,  "MODELS_DIR", test_models)

    db_mod.init_db(test_db)

    from app.main import app
    return TestClient(app)
