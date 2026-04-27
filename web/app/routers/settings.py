import io
import zipfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from ..database import DB_PATH, init_db

router = APIRouter()
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


@router.get("/export")
def export_settings():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if DB_PATH.exists():
            zf.write(DB_PATH, "heimdall.db")
        MODELS_DIR.mkdir(exist_ok=True)
        for engine in MODELS_DIR.glob("*.engine"):
            zf.write(engine, f"models/{engine.name}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=heimdall_settings.zip"},
    )


@router.post("/import", status_code=204)
def import_settings(file: UploadFile):
    if not (file.filename or "").endswith(".zip"):
        raise HTTPException(400, "File must be a .zip archive")
    data = file.file.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if "heimdall.db" in names:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            DB_PATH.write_bytes(zf.read("heimdall.db"))
            init_db(DB_PATH)
        MODELS_DIR.mkdir(exist_ok=True)
        for name in names:
            if name.startswith("models/") and name.endswith(".engine"):
                (MODELS_DIR / Path(name).name).write_bytes(zf.read(name))
