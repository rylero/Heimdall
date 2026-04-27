from pathlib import Path
from fastapi import APIRouter

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
router = APIRouter()
