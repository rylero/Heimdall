import cv2
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


@router.get("/available")
def list_available_cameras():
    """Scan /dev/video0..9 for present devices. Returns empty list on non-Linux."""
    devices = []
    for i in range(10):
        p = Path(f"/dev/video{i}")
        if p.exists():
            devices.append({"index": i, "device": str(p), "name": f"USB Camera {i}"})
    return devices


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
        if not db.execute("SELECT 1 FROM cameras WHERE id = ?", (camera_id,)).fetchone():
            raise HTTPException(404, f"Camera {camera_id} not found")
        set_clause = ", ".join(f"{c} = ?" for c in _COLS)
        values = tuple(getattr(cam, c) for c in _COLS) + (camera_id,)
        db.execute(f"UPDATE cameras SET {set_clause} WHERE id = ?", values)
        row = db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return CameraConfig(**dict(row))


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int):
    with get_db() as db:
        db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))


def _mjpeg_frames(device: str):
    src = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@router.get("/{camera_id}/stream")
def stream_camera(camera_id: int):
    with get_db() as db:
        row = db.execute(
            "SELECT device FROM cameras WHERE id = ?", (camera_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "Camera not found")

    device = row["device"]
    # Pass the device directly to the generator — it handles unavailability by yielding nothing.
    # Checking isOpened() here first to give a clean 503 before streaming starts.
    src = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(src)
    opened = cap.isOpened()
    cap.release()  # Release probe; generator opens its own handle

    if not opened:
        raise HTTPException(503, "Camera device unavailable (may be in use by pipeline)")

    return StreamingResponse(
        _mjpeg_frames(device),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
