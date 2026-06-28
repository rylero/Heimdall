from fastapi import APIRouter, HTTPException
from app import config, jsonc

router = APIRouter()


@router.get("")
def get_settings():
    try:
        with open(config.HEIMDALL_CONFIG) as f:
            return jsonc.loads(f.read())
    except FileNotFoundError:
        raise HTTPException(404, "heimdall.jsonc not found")


@router.put("")
def update_settings(body: dict):
    try:
        with open(config.HEIMDALL_CONFIG, "w") as f:
            f.write(jsonc.dumps(body))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))
