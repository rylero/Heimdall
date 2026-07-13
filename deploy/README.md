# Deployment / competition hardening

Boot the Heimdall stack automatically and survive crashes/brownouts without SSH (review §2D).

## 1. Auto-start at boot (systemd)

`heimdall.service` runs `docker compose up -d` at boot and keeps it up.

```sh
# Edit WorkingDirectory in heimdall.service to the repo path on the Jetson first.
sudo cp deploy/heimdall.service /etc/systemd/system/heimdall.service
sudo systemctl daemon-reload
sudo systemctl enable --now heimdall.service
sudo systemctl status heimdall.service     # verify
```

The `heimdall` container also has `restart: unless-stopped` in `docker-compose.yml`, so it
recovers from a crash even without the unit. The unit covers a full power cycle.

## 2. Stable camera identity (avoid enumeration swaps)

`/dev/video0`, `/dev/video2`, `/dev/video4` are assigned in USB enumeration order — a brownout
or re-plug can swap which physical camera is which. Bind by the stable by-id path instead:

```sh
ls -l /dev/v4l/by-id/
# usb-<vendor>_<model>_<serial>-video-index0  ->  ../../video0
```

Put the `/dev/v4l/by-id/usb-..._<serial>-video-index0` path in each `config/cameras/*.jsonc`
`device` field (and the AprilTag camera in `config/apriltag_layout.jsonc`). The loader passes
the string straight to V4L2, so no code change is needed — only the config path.

## 3. Pipeline stall recovery

Handled in-process: a watchdog detects a silent stall (frames stop with no bus error), flags
`healthy=false` on the tracking stream, and cycles the pipeline `NULL→PLAYING` (see
`heimdall_app.cpp` / review 5.16). No external supervision needed for the stall case.
