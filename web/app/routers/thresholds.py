from fastapi import APIRouter
from ..database import get_db
from ..models import ClassThreshold

router = APIRouter()


@router.get("/", response_model=list[ClassThreshold])
def list_thresholds():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM class_thresholds ORDER BY class_id"
        ).fetchall()
        return [ClassThreshold(**dict(r)) for r in rows]


@router.put("/", response_model=list[ClassThreshold])
def set_thresholds(thresholds: list[ClassThreshold]):
    with get_db() as db:
        db.execute("DELETE FROM class_thresholds")
        for t in thresholds:
            db.execute(
                "INSERT INTO class_thresholds (class_id, class_name, threshold) "
                "VALUES (?, ?, ?)",
                (t.class_id, t.class_name, t.threshold),
            )
        rows = db.execute(
            "SELECT * FROM class_thresholds ORDER BY class_id"
        ).fetchall()
        return [ClassThreshold(**dict(r)) for r in rows]
