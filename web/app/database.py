import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "heimdall.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL DEFAULT '',
    device      TEXT    NOT NULL DEFAULT '/dev/video0',
    camera_type TEXT    NOT NULL DEFAULT 'usb',
    fx REAL NOT NULL DEFAULT 500.0,
    fy REAL NOT NULL DEFAULT 500.0,
    cx REAL NOT NULL DEFAULT 320.0,
    cy REAL NOT NULL DEFAULT 240.0,
    tx REAL NOT NULL DEFAULT 0.0,
    ty REAL NOT NULL DEFAULT 0.0,
    tz REAL NOT NULL DEFAULT 0.5,
    yaw_deg   REAL NOT NULL DEFAULT 0.0,
    pitch_deg REAL NOT NULL DEFAULT 30.0,
    roll_deg  REAL NOT NULL DEFAULT 0.0,
    exposure_us INTEGER NOT NULL DEFAULT 10000,
    gain        REAL    NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS tracker_config (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    confirmation_frames INTEGER NOT NULL DEFAULT 3,
    loss_frames         INTEGER NOT NULL DEFAULT 5,
    gate_distance       REAL    NOT NULL DEFAULT 1.0,
    clutter_density     REAL    NOT NULL DEFAULT 1.0,
    p_detection         REAL    NOT NULL DEFAULT 0.9
);

CREATE TABLE IF NOT EXISTS model_config (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    engine_filename  TEXT,
    num_classes      INTEGER NOT NULL DEFAULT 1,
    labels           TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS class_thresholds (
    class_id   INTEGER PRIMARY KEY,
    class_name TEXT    NOT NULL DEFAULT '',
    threshold  REAL    NOT NULL DEFAULT 0.3
);
"""

def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(_SCHEMA)
        con.execute("INSERT OR IGNORE INTO tracker_config (id) VALUES (1)")
        con.execute("INSERT OR IGNORE INTO model_config   (id) VALUES (1)")

@contextmanager
def get_db():
    # Read DB_PATH from module namespace at call time so monkeypatching works in tests
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
