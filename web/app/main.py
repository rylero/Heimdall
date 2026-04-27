from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .database import init_db
from .routers import cameras, tracker, model_mgmt, thresholds, settings, detections


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Heimdall Config", version="1.0", lifespan=lifespan)

app.include_router(cameras.router,    prefix="/cameras",    tags=["cameras"])
app.include_router(tracker.router,    prefix="/tracker",    tags=["tracker"])
app.include_router(model_mgmt.router, prefix="/models",     tags=["models"])
app.include_router(thresholds.router, prefix="/thresholds", tags=["thresholds"])
app.include_router(settings.router,   prefix="/settings",   tags=["settings"])
app.include_router(detections.router, prefix="/detections", tags=["detections"])

_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
