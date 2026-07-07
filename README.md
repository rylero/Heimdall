# Heimdall
I have tried many times to build an Object Detection/Vision for FRC. My first attempt was in 2024, when I was on team 972, where we built a super simple system that was just a python script managing yolo and sending the position to the robot via network tables. It never saw competition. In 2025 I use a simple approach, by using a PID controller to turn towards the nearest detected algae from Photonvision. This system was actually used in competition, however I wanted better reliability, and the ability to track the actual position, not just relative angle. In Dec 2025, I built my first full Object Detection system, which used TensorRT with RfDetr, and managed to get within 5cm of a alage center repeatedly. However it wasn't used that year due to time restrictions. This is my latest attempt. Heimdall (named after the all seeing watcher of the Bifrost in Norse Mythology), is an attempt to build an all in one system that supports low latency inference, advanced object tracking and filtering, moving object tracking, path planning, and april tags with a gyro constrained coprocessor based method.

<table border="0">
  <tr>
    <td>
      <img width="500" alt="image" src="https://github.com/user-attachments/assets/ef7d6372-ad56-4af5-a699-2d5f1a5632c3" />
    </td>
    <td>
      <h3>What it does</h3>
      <ul>
        <li>Process camera feeds through a json configurable Deepstream Pipeline</li>
        <li>Detects game pieces using a custom YOLOv26 models</li>
        <li>Projects field-relative positions using robot pose received from the RoboRIO</li>
        <li>Tracks detections across frames with a JPDAF filtering system, modified with a object creation and destruction heursitic</li>
        <li>Publishes track events (CONFIRMED / UPDATED / LOST) to the robot over ZMQ</li>
        <li>Detects AprilTags on an additional camera and sends pose corrections back to the robot</li>
      </ul>
    </td>
  </tr>
</table>



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
└── HeimdalTest/  — Complete FRC robot project showing Heimdall integration
docker/           — Dockerfile and docker-compose for Jetson deployment
training/         — RF-DETR training pipeline (dataset prep → train → export → TRT)
tests/            — C++ unit tests (Catch2)
```

## Configuration

All tunable parameters live in `config/heimdall.jsonc` — tracker gate distances, Kalman model, ZMQ ports, logging. Recompiling is not needed, just rerun docker compose up.

## Java integration

The `vendordep/` directory contains a Java client library (`HeimdallLib-java`) for the RoboRIO side. See `examples/HeimdalTest/` for a complete working robot project using it.

## Development

Build natively on the Jetson for fast iteration:

```bash
git pull
docker compose build && docker compose up
```

For competition deployment, build the Docker image and push to a registry:

```bash
docker buildx build --platform linux/arm64 --push \
  --tag ghcr.io/<your-org>/heimdall:latest \
  -f docker/Dockerfile .
# on Jetson:
docker pull ghcr.io/<your-org>/heimdall:latest
```

The Jetson wifi fixer (shuts down ethernet and wifi and reboots, convinves jetson to allow requests outside robot DNS):
```bash
sudo nmcli connection down dhcp-enP8p1s0 && sudo nmcli connection up dhcp-enP8p1s0
sudo nmcli connection down smackdown && sudo nmcli connection up smackdown```
