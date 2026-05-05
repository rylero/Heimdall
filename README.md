# Heimdall Vision System Spec

## Overview

Universal, game-agnostic FRC vision coprocessor. Runs on Jetson Orin Nano Super. Detects and tracks field objects, computes field-relative positions, and streams tracked object data to the RoboRIO in real time.

---

## Hardware Platform

- **Coprocessor:** Jetson Orin Nano Super
- **Cameras:** 2–4 cameras; USB (V4L2) and/or CSI supported simultaneously
- **Default camera count:** 2
- **Resolution:** 640×480 (RFDetr standard input)
- **Target FPS:** ≥50 per camera
- **Deployment:** Docker container

The Orange Pi AprilTag system is a separate, independent subsystem — no integration with Heimdall.

---

## Object Detection

- **Framework:** NVIDIA DeepStream with RFDetr backbone
- **Optimization:** INT8 TensorRT `.engine` files
- **Classes:** Variable — adapts to however many classes the loaded model defines
- **Confidence threshold:** Configurable per class via web interface
- **Model loading:** `.engine` files uploaded and selected via web interface
- **Multi-camera:** All camera streams processed in a combined DeepStream pipeline for throughput efficiency
- **Latency target:** Minimum possible end-to-end (camera → robot pose output)

---

## Object Pose Estimation

- **Method:** Ground ray projection
- **Coordinate frame:** Field-relative (WPILib standard field coordinates)
- **Ground plane:** Fixed at robot ground level; defined by camera extrinsic transform from robot center
- **Camera mounts:** Fixed; extrinsics set manually via web interface
- **Accuracy target:** ±7 cm

### Multi-Camera Fusion

When multiple cameras observe the same object simultaneously, positions are fused (weighted average) into a single field-relative estimate before being handed to the tracker.

---

## Object Tracking

- **Algorithm:** Joint Probabilistic Data Association (JPDA)
- **Scope:** Global field map — camera-agnostic; tracks are not tied to any specific camera
- **Max simultaneous tracks:** 80–100 objects
- **Track ID:** Assigned per object, included in output
- **Object classes tracked:** All detected classes including robots

### Tunable Parameters (via web interface)

| Parameter | Description |
|---|---|
| Confirmation frames | Consecutive detections required before a track is published |
| Loss frames | Consecutive missed frames before a track is marked lost |
| Gate distance (m) | Max displacement frame-to-frame before treating as new object |

### Track Lifecycle Events

- **Confirmed:** Track created after confirmation threshold met
- **Updated:** New position published every frame
- **Lost:** Published once when a track expires; track ID retired

---

## Communication

- **Protocol:** ZeroMQ (push-pull both directions — 1:1 RoboRIO ↔ Jetson)
- **Encoding:** Protocol Buffers (`.proto` files distributed with Heimdall repo)

### RoboRIO → Jetson

- **Data:** Full robot pose (field-relative)
- **Rate:** Every robot control loop cycle

### Jetson → RoboRIO

- **Data per tracked object:** field position (x, y), object class ID, track ID
- **Rate:** Every detection frame
- **Connection loss:** Publishes error state on ZeroMQ disconnect
- **Heartbeat:** Jetson sends periodic heartbeat so robot can detect Heimdall is alive

---

## Web Interface

Accessed from driver station laptop on robot WiFi. No internet required. No authentication.

### Features

- **Camera config:** Per-camera extrinsic transform (position + rotation relative to robot center)
- **Camera config:** Per-camera exposure settings
- **Camera preview:** Live stream with detection + tracking overlay
- **Model management:** Upload `.engine` files, select active model
- **Detection tuning:** Per-class confidence thresholds
- **Tracker tuning:** JPDA parameters (confirmation frames, loss frames, gate distance)
- **Settings management:** Import / export all settings as a ZIP file

---

## Configuration Persistence

- All settings stored in a single SQLite database on disk
- Import / export via the web interface as a ZIP archive
- Settings include: camera transforms, exposure values, model files, confidence thresholds, JPDA parameters

---

## Out of Scope

- Path planning and automation (handled on RoboRIO with custom robot code)
- AprilTag pose estimation (handled by separate Orange Pi system)
- Post-match logging (positions logged on RoboRIO via WPILib)

## Future Considerations

- **Post-match logging:** Log raw detections and track lifecycle events to `.wpilog` format for AdvantageScope compatibility. Not needed at launch — RoboRIO logging covers initial needs.
