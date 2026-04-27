import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile
from ..database import get_db
from ..models import ModelConfig

router = APIRouter()
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


@router.get("/", response_model=list[str])
def list_models():
    MODELS_DIR.mkdir(exist_ok=True)
    return sorted(f.name for f in MODELS_DIR.glob("*.engine"))


@router.post("/upload", status_code=201)
def upload_model(file: UploadFile):
    if not (file.filename or "").endswith(".engine"):
        raise HTTPException(400, "File must have .engine extension")
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / Path(file.filename).name  # basename only — prevent path traversal
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"filename": file.filename}


@router.get("/active", response_model=ModelConfig)
def get_active():
    with get_db() as db:
        row = db.execute("SELECT * FROM model_config WHERE id = 1").fetchone()
        r = dict(row)
        return ModelConfig(
            engine_filename=r["engine_filename"],
            num_classes=r["num_classes"],
            labels=r["labels"].splitlines() if r["labels"] else [],
        )


@router.put("/active", response_model=ModelConfig)
def set_active(cfg: ModelConfig):
    if cfg.engine_filename is not None:
        if not (MODELS_DIR / cfg.engine_filename).exists():
            raise HTTPException(404, f"Engine not found: {cfg.engine_filename}")
    with get_db() as db:
        db.execute(
            "UPDATE model_config SET engine_filename=?, num_classes=?, labels=? WHERE id=1",
            (cfg.engine_filename, cfg.num_classes, "\n".join(cfg.labels)),
        )
    return cfg


@router.delete("/{filename}", status_code=204)
def delete_model(filename: str):
    if not filename.endswith(".engine"):
        raise HTTPException(400, "Only .engine files can be deleted")
    path = MODELS_DIR / Path(filename).name  # basename only
    if not path.exists():
        raise HTTPException(404, f"Model not found: {filename}")
    path.unlink()
