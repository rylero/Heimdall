# Heimdall

Real-time object tracking and AprilTag pose estimation for FRC robots. Runs on a Jetson Orin as a Docker container, streams results to the RoboRIO over ZeroMQ.

## What it does

- Ingests two USB cameras through NVIDIA DeepStream
- Detects game pieces using a custom RF-DETR OR YOLOv26
- Tracks detections across frames with a JPDAF Kalman filterr
- Projects field-relative positions using robot pose received from the RoboRIO
- Publishes track events (CONFIRMED / UPDATED / LOST) to the robot over ZMQ
- Detects AprilTags on a third camera and sends pose corrections back to the robot

## Architecture

```
Robot (RoboRIO)                      Jetson Orin
────────────────────────             ────────────────────────────────────────
HeimdallSubsystem.java               Heimdall Docker container
  │                                    │
  │  sendPose()  ── ZMQ :5555 ───────► pose_recv_loop
  │                                    │
  │                                    ├─ DeepStream pipeline (cam0, cam1)
  │                                    │   nvinfer → JPDAF tracker → track events
  │                                    │
  │  getLatestFrame() ◄─ ZMQ :5556 ─  output_pub_sock
  │                                    │
  │                                    └─ AprilTag thread (cam2)
  │                                        V4L2 → tag36h11 → solvePnP → pose
  │
  │  getLatestVisionPose() ◄─ ZMQ :5558 ─ apriltag_pose_pub_sock
  │
  └─ drive.addVisionMeasurement()
```

## Repository layout

```
src/
├── app/          — HeimdallApp: top-level orchestration, detection worker, pose recv loop
├── comm/         — CommLayer: ZMQ sockets (pose in, track events out, AprilTag pose out)
├── pipeline/     — DeepStream GStreamer pipeline, nvinfer probe, camera source
├── tracker/      — JPDAF tracker, Kalman filter (CP / CV / CA models)
├── pose/         — PoseEstimator: projects camera detections to field coordinates
├── apriltag/     — AprilTagDetector: V4L2 capture, tag detection, solvePnP
├── models/       — Custom nvinfer bbox parsers (RF-DETR, YOLO)
└── config/       — JSON config loaders

config/           — Runtime config files (no recompile needed to tune)
proto/            — Protobuf message definitions (RobotPoseMsg, DetectionFrameMsg, VisionPoseMsg)
vendordep/        — Java client library published to a local maven repo
examples/
└── HeimdallTest/ — Complete FRC robot project showing Heimdall integration
docker/           — Dockerfile and docker-compose for Jetson deployment
training/         — RF-DETR training pipeline (dataset prep → train → export → TRT)
tests/            — C++ unit tests (Catch2)
```

## Getting started

See [SETUP.md](SETUP.md) for full hardware setup, calibration, and deployment instructions.

## Configuration

All tunable parameters live in `config/heimdall.jsonc` — tracker gate distances, Kalman model, ZMQ ports, logging. No recompile needed.

## Java integration

The `vendordep/` directory contains a Java client library (`HeimdallLib-java`) for the RoboRIO side. See `examples/HeimdallTest/` for a complete working robot project using it.

## Development

Build natively on the Jetson for fast iteration:

```bash
git pull
cmake --build build --parallel $(nproc)
```

For competition deployment, build the Docker image and push to a registry:

```bash
docker buildx build --platform linux/arm64 --push \
  --tag ghcr.io/<your-org>/heimdall:latest \
  -f docker/Dockerfile .
# on Jetson:
docker pull ghcr.io/<your-org>/heimdall:latest
```

The Jetson wifi fixer:
```bash
sudo nmcli connection down dhcp-enP8p1s0 && sudo nmcli connection up dhcp-enP8p1s0
sudo nmcli connection down smackdown && sudo nmcli connection up smackdown```
