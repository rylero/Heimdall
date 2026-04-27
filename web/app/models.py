from typing import Literal
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    id: int | None = None
    name: str = ""
    device: str = "/dev/video0"
    camera_type: Literal["usb", "csi"] = "usb"
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.5
    yaw_deg: float = 0.0
    pitch_deg: float = Field(default=30.0, description="Downward tilt degrees")
    roll_deg: float = 0.0
    exposure_us: int = 10000
    gain: float = 1.0


class TrackerConfig(BaseModel):
    confirmation_frames: int = Field(default=3, ge=1)
    loss_frames: int = Field(default=5, ge=1)
    gate_distance: float = Field(default=1.0, gt=0)
    clutter_density: float = Field(default=1.0, gt=0)
    p_detection: float = Field(default=0.9, gt=0, le=1)


class ModelConfig(BaseModel):
    engine_filename: str | None = None
    num_classes: int = Field(default=1, ge=1)
    labels: list[str] = []


class ClassThreshold(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str = ""
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
