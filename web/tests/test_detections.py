import time
import pytest
from fastapi.testclient import TestClient


def test_proto_roundtrip_raw_detection():
    """RawDetectionFrameMsg serializes and deserializes correctly."""
    from app.proto import heimdall_pb2

    frame = heimdall_pb2.RawDetectionFrameMsg()
    frame.timestamp_ns = 12345
    d = frame.detections.add()
    d.camera_id = 0
    d.class_id  = 2
    d.confidence = 0.87
    d.left, d.top, d.width, d.height = 100.0, 50.0, 80.0, 60.0

    data = frame.SerializeToString()
    parsed = heimdall_pb2.RawDetectionFrameMsg()
    parsed.ParseFromString(data)

    assert parsed.timestamp_ns == 12345
    assert len(parsed.detections) == 1
    assert abs(parsed.detections[0].confidence - 0.87) < 1e-4
    assert parsed.detections[0].left == 100.0


def test_ws_connects_without_error(client):
    """WebSocket endpoint accepts connections without crashing."""
    with client.websocket_connect("/detections/ws") as ws:
        # Give the ZMQ subscriber thread time to start
        time.sleep(0.1)
        # No data expected (no ZMQ publisher running), but connection stays open
