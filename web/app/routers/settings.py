from pathlib import Path
from fastapi import APIRouter

DB_PATH = None
MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
router = APIRouter()
