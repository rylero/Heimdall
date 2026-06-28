from fastapi import APIRouter, HTTPException
import docker
from app import config

router = APIRouter()


def _get_container():
    client = docker.from_env()
    containers = client.containers.list(
        all=True, filters={"name": config.HEIMDALL_CONTAINER_NAME}
    )
    if not containers:
        raise HTTPException(404, f"Container '{config.HEIMDALL_CONTAINER_NAME}' not found")
    return containers[0]


@router.get("/status")
def get_status():
    try:
        c = _get_container()
        return {"status": c.status, "name": c.name}
    except HTTPException:
        return {"status": "not_found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/restart")
def restart_heimdall():
    try:
        c = _get_container()
        c.restart(timeout=5)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
