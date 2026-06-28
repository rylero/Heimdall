import os
import glob
import subprocess
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app import config, jsonc

router = APIRouter()


def _list_camera_files() -> list[str]:
    pattern = os.path.join(config.CAMERAS_DIR, "*.jsonc")
    return sorted(glob.glob(pattern))


def _read_camera(path: str) -> dict:
    with open(path) as f:
        return jsonc.loads(f.read())


def _write_camera(path: str, data: dict):
    with open(path, "w") as f:
        f.write(jsonc.dumps(data))


@router.get("")
def list_cameras():
    result = []
    for path in _list_camera_files():
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            data = _read_camera(path)
            data["_name"] = name
            result.append(data)
        except Exception as e:
            result.append({"_name": name, "_error": str(e)})
    # Also include apriltag layout as a pseudo-camera
    if os.path.exists(config.APRILTAG_CONFIG):
        try:
            data = jsonc.loads(open(config.APRILTAG_CONFIG).read())
            data["_name"] = "apriltag"
            data["_is_apriltag"] = True
            result.append(data)
        except Exception:
            pass
    return result


@router.get("/{name}")
def get_camera(name: str):
    if name == "apriltag":
        if not os.path.exists(config.APRILTAG_CONFIG):
            raise HTTPException(404, "apriltag config not found")
        data = jsonc.loads(open(config.APRILTAG_CONFIG).read())
        data["_name"] = "apriltag"
        data["_is_apriltag"] = True
        return data
    path = os.path.join(config.CAMERAS_DIR, f"{name}.jsonc")
    if not os.path.exists(path):
        raise HTTPException(404, f"Camera {name} not found")
    return _read_camera(path)


@router.put("/{name}")
def update_camera(name: str, body: dict):
    body.pop("_name", None)
    body.pop("_is_apriltag", None)
    body.pop("_error", None)
    if name == "apriltag":
        _write_camera(config.APRILTAG_CONFIG, body)
        return {"ok": True}
    path = os.path.join(config.CAMERAS_DIR, f"{name}.jsonc")
    if not os.path.exists(path):
        raise HTTPException(404, f"Camera {name} not found")
    _write_camera(path, body)
    return {"ok": True}


@router.get("/{name}/v4l2")
def get_v4l2_controls(name: str):
    """Read current v4l2 control values for a camera device."""
    cam = get_camera(name)
    device = cam.get("device")
    if not device:
        raise HTTPException(400, "Camera has no device path")
    try:
        result = subprocess.run(
            ["v4l2-ctl", f"--device={device}", "--list-ctrls-menus"],
            capture_output=True, text=True, timeout=5
        )
        return {"raw": result.stdout, "device": device}
    except FileNotFoundError:
        raise HTTPException(503, "v4l2-ctl not found")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.put("/{name}/v4l2")
def set_v4l2_controls(name: str, controls: dict):
    """Apply v4l2 controls live without restarting pipeline."""
    cam = get_camera(name)
    device = cam.get("device")
    if not device:
        raise HTTPException(400, "Camera has no device path")
    ctrl_str = ",".join(f"{k}={v}" for k, v in controls.items())
    try:
        result = subprocess.run(
            ["v4l2-ctl", f"--device={device}", f"--set-ctrl={ctrl_str}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise HTTPException(400, result.stderr.strip())
        return {"ok": True, "device": device, "controls": controls}
    except FileNotFoundError:
        raise HTTPException(503, "v4l2-ctl not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
