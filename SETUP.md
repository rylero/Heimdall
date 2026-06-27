# Heimdall Robot Setup Guide

Two object-detection cameras (floor-facing, DeepStream pipeline) + one AprilTag camera (wall-facing, pose correction). All parameters are file-configurable — no recompile needed for tuning.

---

## Status legend

- ✅ Done — already implemented, no action needed
- 🔧 You do this — hands-on steps on the robot / Jetson

---

## Architecture overview

```
Robot (RoboRIO / roboRIO 2)          Jetson Orin
────────────────────────────         ────────────────────────────────────────
HeimdallSubsystem.java               Heimdall Docker container
  │                                    │
  │ sendPose()  ──── ZMQ :5555 ──────► pose_recv_loop
  │                                    │
  │                                    ├─ DeepStream pipeline (cam0, cam1)
  │                                    │   nvinfer → JPDAF tracker → track events
  │                                    │
  │ getLatestFrame() ◄─ ZMQ :5556 ───  comm_layer PUSH
  │                                    │
  │                                    └─ AprilTag thread (cam2)
  │                                        V4L2 → tag36h11 → solvePnP → pose
  │
  │ getLatestVisionPose() ◄─ ZMQ :5558 ─ comm_layer PUB
  │
  └─ drive.addVisionMeasurement()
```

---

## Step 1 — Heimdall C++ (AprilTag detection) ✅

Already implemented:

- `src/apriltag/apriltag_detector.cpp` — V4L2 camera capture (YUYV → grayscale, no FFmpeg dep), `apriltag` C library detection, `cv::solvePnPGeneric(SOLVEPNP_IPPE_SQUARE)` for 3-D pose, ambiguity rejection
- `src/apriltag/tag_layout.h/.cpp` — loads `config/apriltag_layout.jsonc`
- `src/comm/comm_layer.cpp` — adds ZMQ PUB socket on `:5558`, `send_vision_pose()`
- `src/app/heimdall_app.cpp` — background `apriltag_loop()` thread
- `proto/heimdall.proto` — `VisionPoseMsg {x, y, heading, timestamp_ns}`

Filter models available in `heimdall.jsonc`:
| Value | State vector | Use case |
|-------|-------------|----------|
| `constant_position` | [x, y] | Very slow objects |
| `constant_velocity` | [x, y, vx, vy] | Default — most FRC use cases |
| `constant_acceleration` | [x, y, vx, vy, ax, ay] | Fast, predictable trajectories |

---

## Step 2 — Java vendordep (VisionPoseEstimate) ✅

Already implemented in `vendordep/src/main/java/com/heimdall/`:

- `VisionPoseEstimate.java` — fields: `x, y, headingRad, timestampNs`; helper `getTimestampSecs()` for WPILib
- `proto/ProtoReader.java` — `parseVisionPose(byte[])` — hand-rolled, no protobuf-java dep
- `HeimdallClient.java` — SUB socket on `:5558`, `getLatestVisionPose()` (consume-once semantics)

JAR rebuilt and checksums updated in `vendordep/maven/`.

---

## Step 3 — Config files ✅

All Heimdall parameters are in files under `config/`. No recompile needed to change them.

```
config/
├── heimdall.jsonc            ← master app config (tracker, ZMQ ports, logging)
├── cameras/
│   ├── cam0.jsonc            ← object-detection camera 0 (intrinsics + extrinsics)
│   └── cam1.jsonc            ← object-detection camera 1
├── apriltag_layout.jsonc     ← AprilTag camera intrinsics, field tag poses
└── infer_yolo26n.txt        ← DeepStream/TensorRT nvinfer config (existing)
```

Key parameters in `heimdall.jsonc`:

```jsonc
{
  "cameras_dir": "config/cameras",
  "infer_config": "config/infer_yolo26n.txt",
  "tracker": {
    "confirmation_frames": 2,    // frames before a new track is reported
    "loss_frames": 15,           // frames without detection before LOST
    "gate_distance": 2.0,        // Mahalanobis gate in metres
    "clutter_density": 1.0,      // false-positive density (per m²)
    "p_detection": 0.9,          // probability a real object is detected each frame
    "filter_model": "constant_velocity"
  },
  "comm": {
    "pose_bind_addr":        "tcp://*:5555",
    "output_bind_addr":      "tcp://*:5556",
    "raw_output_bind_addr":  "tcp://*:5557",
    "vision_pose_bind_addr": "tcp://*:5558"
  },
  "bypass_tracker": false,
  "log_tracking": true,
  "log_path": "logs/tracker_log.csv",
  "apriltag_layout": "config/apriltag_layout.jsonc"
}
```

---

## Step 4 — Build Docker image 🔧

Build on your Windows machine from the project root:

```powershell
cd C:\Users\ryan\Dev\Heimdall

docker buildx build `
  --platform linux/arm64 `
  --tag heimdall:latest `
  --output type=docker `
  --file docker/Dockerfile .
```

> **First build takes 30–60 minutes** under QEMU ARM64 emulation (compiling libzmq, protobuf, Catch2). Subsequent builds hit the layer cache and take < 5 minutes (only your changed source files recompile).

Export and copy to Jetson:

```powershell
docker save heimdall:latest | gzip > heimdall.tar.gz
scp heimdall.tar.gz jetson@<JETSON_IP>:~
```

On the Jetson:

```bash
docker load < ~/heimdall.tar.gz
```

---

## Step 5 — Identify cameras 🔧

On the Jetson with all cameras plugged in:

```bash
v4l2-ctl --list-devices
```

Example output:
```
USB Camera (usb-3610000.xhci-2.1):
    /dev/video0
    /dev/video1

USB Camera (usb-3610000.xhci-2.2):
    /dev/video2
    /dev/video3

USB Camera (usb-3610000.xhci-2.3):
    /dev/video4
    /dev/video5
```

Each USB camera gets two entries — use the **even** one (the actual video capture node, not the metadata node). To confirm which camera is which physical unit, unplug one at a time.

Quick preview to confirm:
```bash
ffplay -f v4l2 -video_size 640x480 /dev/video0
```

Update device paths in:
- `config/cameras/cam0.jsonc` → `"device"`
- `config/cameras/cam1.jsonc` → `"device"`
- `config/apriltag_layout.jsonc` → `"camera"."device"`

---

## Step 6 — Calibrate cameras 🔧

Print a checkerboard calibration target: go to https://calib.io → **Chess Board**, set inner corners to **9×6**, square size **25 mm**. Print on paper, mount flat on a rigid board (no flex).

### AprilTag camera

```bash
# Install OpenCV Python if not present
sudo apt install python3-opencv

python3 tools/calibrate_apriltag_camera.py \
    --device /dev/video4 \
    --squares 9x6 \
    --size 0.025 \
    --width 640 \
    --height 480
```

While running:
- Move the board to different **distances** (30 cm to 150 cm), **angles** (tilt ±30°), and **positions** within the frame (corners, centre, edges)
- Press **SPACE** to capture each view — aim for **25–30 captures**
- Press **q** to compute

The script prints values to paste into `config/apriltag_layout.jsonc`:

```jsonc
"camera": {
  "device": "/dev/video4",
  "width": 640, "height": 480, "fps": 10,
  "fx": 612.3456,   // ← from calibration output
  "fy": 614.1234,
  "cx": 318.7654,
  "cy": 241.3210,
  "k1": -0.123456,
  "k2":  0.089012,
  "p1":  0.000345,
  "p2": -0.000123,
  "k3":  0.0
}
```

**Reprojection error targets:**
- < 0.5 px — excellent
- 0.5–1.0 px — good
- 1.0–1.5 px — usable
- \> 1.5 px — re-do with more captures and better board hold

### Object-detection cameras

Run the same script for each:

```bash
python3 tools/calibrate_apriltag_camera.py --device /dev/video0 ...
python3 tools/calibrate_apriltag_camera.py --device /dev/video2 ...
```

Paste the `fx/fy/cx/cy/k1…` values into `config/cameras/cam0.jsonc` and `cam1.jsonc` under `"intrinsics"`.

---

## Step 7 — Measure camera extrinsics 🔧

Extrinsics describe where each camera is **physically mounted on the robot**.

**Coordinate convention:** +X = robot forward, +Y = robot left, +Z = up. Origin = robot centre of rotation (typically the centre of the drivebase).

### What to measure

Use a tape measure or CAD model. For each camera:

| Field | Meaning |
|-------|---------|
| `tx` | How far **forward** from robot centre (metres, positive = forward) |
| `ty` | How far **left** of robot centre (metres, positive = left, negative = right) |
| `tz` | Height above the **floor** (metres) |
| `yaw` | Which direction the camera faces (radians, 0 = robot forward, π/2 = left) |
| `pitch` | Tilt down/up (radians, negative = tilted down) |
| `roll` | Rotation around camera axis (radians, 0 = camera "up" is robot-up) |

### Floor-facing cameras (cam0, cam1)

```jsonc
// cam0.jsonc — floor-facing, left of robot
"extrinsics": {
  "tx":    0.20,      // 20 cm forward of centre
  "ty":    0.15,      // 15 cm left of centre
  "tz":    0.30,      // 30 cm above floor
  "yaw":   0.0,       // camera "forward" = robot forward
  "pitch": -1.5708,   // pointing straight down (-π/2)
  "roll":  0.0
}
```

If the camera is mounted rotated (e.g. 90° clockwise), set `roll` accordingly.

### AprilTag camera (apriltag_layout.jsonc)

```jsonc
"robot_to_camera": {
  "x":     0.35,    // 35 cm forward of centre
  "y":     0.00,    // centred left-right
  "z":     0.50,    // 50 cm above floor
  "roll":  0.0,
  "pitch": -0.15,   // ~8° downward tilt to see tags at various distances
  "yaw":   0.0      // facing straight forward
}
```

---

## Step 8 — Set field AprilTag positions 🔧

Download the official WPILib AprilTag layout for your game year:

```
https://github.com/wpilibsuite/allwpilib/tree/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag
```

The WPILib JSON uses quaternion rotations. Convert to yaw with this one-liner:

```python
import math
# Example: W=0.5, X=0, Y=0, Z=0.866 → yaw = π/3 (60°)
W, X, Y, Z = 0.5, 0, 0, 0.866
yaw = math.atan2(2*(W*Z + X*Y), 1 - 2*(Y*Y + Z*Z))
print(round(yaw, 4))
```

Wall-mounted FRC tags have `roll=0, pitch=0` (facing horizontally).

Add one entry per visible tag in `config/apriltag_layout.jsonc`. You only need tags your robot can reasonably see from the field — no need to include all 22:

```jsonc
"tags": [
  { "id": 7,  "x": 8.308, "y": 4.115, "z": 1.451, "roll": 0, "pitch": 0, "yaw": 3.1416 },
  { "id": 8,  "x": 8.308, "y": 2.748, "z": 1.451, "roll": 0, "pitch": 0, "yaw": 3.1416 },
  { "id": 4,  "x": 0.356, "y": 4.983, "z": 1.451, "roll": 0, "pitch": 0, "yaw": 0.0    }
]
```

Also confirm `"tag_size_meters"` matches your printed tags (measure with calipers — printed sizes are often slightly off).

---

## Step 9 — Robot code (HeimdallTest) 🔧

In `HeimdallSubsystem.java`, add this to `periodic()`:

```java
VisionPoseEstimate vp = heimdall.getLatestVisionPose();
if (vp != null) {
    drive.addVisionMeasurement(
        new Pose2d(vp.getX(), vp.getY(), new Rotation2d(vp.getHeadingRad())),
        vp.getTimestampSecs(),
        VecBuilder.fill(0.5, 0.5, 0.3)   // x stddev, y stddev, heading stddev (metres/rad)
    );
}
```

**Tuning the standard deviations** (`VecBuilder.fill(x, y, θ)`):
- Lower values = trust vision **more** (pose snaps to tags aggressively)
- Higher values = trust odometry **more** (vision is a gentle correction)
- Start at `(0.5, 0.5, 0.3)` and lower toward `(0.1, 0.1, 0.05)` as you gain confidence in your calibration

---

## Step 10 — Run on Jetson 🔧

Create a config directory and copy your filled-in configs to the Jetson:

```bash
# On Jetson
mkdir -p ~/heimdall/config/cameras ~/heimdall/logs
```

```powershell
# On Windows — copy config after editing
scp -r config/* jetson@<JETSON_IP>:~/heimdall/config/
```

Start the container:

```bash
docker run --runtime nvidia \
  --device /dev/video0 \
  --device /dev/video2 \
  --device /dev/video4 \
  -v ~/heimdall/config:/app/config \
  -v ~/heimdall/logs:/app/logs \
  heimdall:latest
```

> Every camera device must have its own `--device` flag. `--runtime nvidia` is required for DeepStream + TensorRT.

**To change config:** edit files in `~/heimdall/config/` on the Jetson (or re-scp from Windows), then restart the container. No rebuild needed.

---

## Step 11 — Verify 🔧

### Tracks flowing

In HeimdallTest, enable the subsystem and log:
```java
DetectionFrame frame = heimdall.getLatestFrame();
SmartDashboard.putBoolean("Heimdall/healthy", heimdall.isHealthy());
SmartDashboard.putNumber("Heimdall/trackCount",
    frame != null ? frame.getEvents().size() : 0);
```

You should see `healthy=true` and non-zero track count within a few seconds of placing an object in front of the cameras.

### Pose corrections

Drive the robot within view of a known AprilTag. Log:
```java
VisionPoseEstimate vp = heimdall.getLatestVisionPose();
if (vp != null) {
    SmartDashboard.putNumber("Vision/x",   vp.getX());
    SmartDashboard.putNumber("Vision/y",   vp.getY());
    SmartDashboard.putNumber("Vision/hdg", Math.toDegrees(vp.getHeadingRad()));
}
```

The pose should arrive at ~10 Hz and be within ~20 cm of your actual position.

### Tracker log

```bash
tail -f ~/heimdall/logs/tracker_log.csv
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `healthy=false`, no tracks | Wrong `/dev/videoX` for detect cams | Re-run `v4l2-ctl --list-devices`, update `config/cameras/` |
| Tracks drift off real positions | Wrong extrinsics (tx/ty/tz/pitch) | Re-measure camera mount position |
| Ghost tracks / track churn | `gate_distance` too large or `clutter_density` too low | Raise `clutter_density`, lower `gate_distance` in `heimdall.jsonc` |
| Tracks drop instantly | `loss_frames` too low | Raise to 20–30 in `heimdall.jsonc` |
| No pose corrections | AprilTag cam can't see tags | Verify device path, check lighting, confirm tag IDs in JSON |
| Pose jumps on tag detect | Bad calibration or wrong tag field position | Re-run calibration, double-check tag coordinates from WPILib JSON |
| Pose correction in wrong direction | `robot_to_camera.yaw` wrong | Add/subtract π from yaw in `apriltag_layout.jsonc` |
| Pose estimate far off even with correct tags | `tag_size_meters` wrong | Measure your printed tag and update |
| Container exits immediately | Missing `--device` flags | Add `--device /dev/videoX` for every camera |

---

## Config quick-reference

### Changing a parameter

1. Edit the file in `~/heimdall/config/` on the Jetson (or edit locally and `scp` again)
2. Restart the container — no rebuild needed

### What requires a rebuild

Only C++ source code changes require rebuilding the Docker image. Config, tag layouts, camera calibrations, and tracker tuning never do.
