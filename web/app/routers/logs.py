import asyncio
import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import docker
from app import config

router = APIRouter()


async def _tail_file(path: str):
    """Async generator: yield new lines appended to a file (SSE format)."""
    try:
        with open(path) as f:
            f.seek(0, 2)  # seek to end
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    await asyncio.sleep(0.2)
    except FileNotFoundError:
        yield f"data: [file not found: {path}]\n\n"


async def _container_logs():
    """Async generator: stream heimdall container stdout/stderr via Docker SDK."""
    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"name": config.HEIMDALL_CONTAINER_NAME})
        if not containers:
            yield f"data: [container '{config.HEIMDALL_CONTAINER_NAME}' not running]\n\n"
            return
        container = containers[0]
        for chunk in container.logs(stream=True, follow=True, tail=200):
            line = chunk.decode("utf-8", errors="replace").rstrip()
            if line:
                yield f"data: {line}\n\n"
    except Exception as e:
        yield f"data: [docker error: {e}]\n\n"


@router.get("/tracker")
def stream_tracker():
    return StreamingResponse(
        _tail_file(config.TRACKER_LOG),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/container")
def stream_container():
    return StreamingResponse(
        _container_logs(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
