import asyncio
import json
import os
import threading
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Address of the C++ pipeline's raw detection PUB socket.
# Override via RAW_DET_ADDR env var when Jetson is on a different host.
_RAW_DET_ADDR = os.getenv("RAW_DET_ADDR", "tcp://localhost:5557")

_latest_frame: dict | None = None
_frame_lock = threading.Lock()
_subscriber_started = False
_subscriber_lock = threading.Lock()


def _start_subscriber() -> None:
    global _subscriber_started
    with _subscriber_lock:
        if _subscriber_started:
            return
        _subscriber_started = True
    threading.Thread(target=_zmq_loop, daemon=True, name="zmq-raw-sub").start()


def _zmq_loop() -> None:
    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(_RAW_DET_ADDR)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 200)

    while True:
        try:
            data = sock.recv()
        except zmq.Again:
            continue
        except zmq.ZMQError:
            break
        try:
            from app.proto import heimdall_pb2
            msg = heimdall_pb2.RawDetectionFrameMsg()
            msg.ParseFromString(data)
            payload = {
                "ts": msg.timestamp_ns,
                "dets": [
                    {
                        "cam":  d.camera_id,
                        "cls":  d.class_id,
                        "conf": round(d.confidence, 3),
                        "x":    d.left,
                        "y":    d.top,
                        "w":    d.width,
                        "h":    d.height,
                    }
                    for d in msg.detections
                ],
            }
            with _frame_lock:
                global _latest_frame
                _latest_frame = payload
        except Exception:
            pass


@router.websocket("/ws")
async def ws_detections(ws: WebSocket) -> None:
    """Browser connects here to receive raw detection frames at ~30 Hz."""
    _start_subscriber()
    await ws.accept()
    try:
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is not None:
                try:
                    await ws.send_text(json.dumps(frame))
                except Exception:
                    break
            await asyncio.sleep(0.033)  # ~30 Hz
    except WebSocketDisconnect:
        pass
