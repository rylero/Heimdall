from fastapi import APIRouter, HTTPException
from ..database import get_db
from ..models import CameraConfig

router = APIRouter()

_COLS = ("name", "device", "camera_type", "fx", "fy", "cx", "cy",
         "tx", "ty", "tz", "yaw_deg", "pitch_deg", "roll_deg",
         "exposure_us", "gain")


@router.get("/", response_model=list[CameraConfig])
def list_cameras():
    with get_db() as db:
        rows = db.execute("SELECT * FROM cameras ORDER BY id").fetchall()
        return [CameraConfig(**dict(r)) for r in rows]


@router.post("/", response_model=CameraConfig, status_code=201)
def create_camera(cam: CameraConfig):
    with get_db() as db:
        placeholders = ", ".join("?" * len(_COLS))
        col_list = ", ".join(_COLS)
        values = tuple(getattr(cam, c) for c in _COLS)
        cur = db.execute(
            f"INSERT INTO cameras ({col_list}) VALUES ({placeholders})", values
        )
        row = db.execute("SELECT * FROM cameras WHERE id = ?", (cur.lastrowid,)).fetchone()
        return CameraConfig(**dict(row))


@router.put("/{camera_id}", response_model=CameraConfig)
def update_camera(camera_id: int, cam: CameraConfig):
    with get_db() as db:
        set_clause = ", ".join(f"{c} = ?" for c in _COLS)
        values = tuple(getattr(cam, c) for c in _COLS) + (camera_id,)
        db.execute(f"UPDATE cameras SET {set_clause} WHERE id = ?", values)
        row = db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"Camera {camera_id} not found")
        return CameraConfig(**dict(row))


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int):
    with get_db() as db:
        db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
