from fastapi import APIRouter
from ..database import get_db
from ..models import TrackerConfig

router = APIRouter()


@router.get("/", response_model=TrackerConfig)
def get_tracker():
    with get_db() as db:
        row = db.execute("SELECT * FROM tracker_config WHERE id = 1").fetchone()
        return TrackerConfig(**dict(row))


@router.put("/", response_model=TrackerConfig)
def update_tracker(cfg: TrackerConfig):
    with get_db() as db:
        db.execute(
            "UPDATE tracker_config SET "
            "confirmation_frames=?, loss_frames=?, gate_distance=?, "
            "clutter_density=?, p_detection=? WHERE id=1",
            (cfg.confirmation_frames, cfg.loss_frames, cfg.gate_distance,
             cfg.clutter_density, cfg.p_detection),
        )
        row = db.execute("SELECT * FROM tracker_config WHERE id = 1").fetchone()
        return TrackerConfig(**dict(row))
