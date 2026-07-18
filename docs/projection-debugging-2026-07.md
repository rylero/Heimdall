# Projection Debugging Postmortem — July 2026

Detected balls were being projected to wildly wrong field positions (e.g. reported
`2.0 m fwd, 0.5 m left` for a ball physically at `~1.2 m fwd, ~0.9 m left`). This
document records every distinct bug we found, in the order we hit them, so the same
traps don't cost days again.

**TL;DR** — five independent problems stacked on top of each other. The projection
*math* was correct the whole time; every bug was in the **inputs** to it: the mount
angles, which config file the runtime actually reads, the intrinsics, the pixel
coordinate space, and the sensor-orientation convention.

---

## The camera under test

- USB camera, native **640×480**, mounted **upside-down** (`"rotation": 180`).
- Model input **640×640** (`infer-dims=3;640;640`), `maintain-aspect-ratio=1`.
- Robot held at `(0, 0, 0)`, heading 0, for all tests (mock robot).
- Ground-plane projection: unproject the bbox bottom-center pixel through the
  intrinsics + extrinsics, intersect the `z=0` plane. Code in
  `src/pose/pose_estimator.cpp`.

---

## Bug 1 — Onshape yaw measured against the wrong reference frame

**Symptom:** the mount FeatureScript reported **yaw = 115°** (camera facing left-and-back)
for a camera that physically points **forward, slightly left** (~25°).

**Root cause:** yaw is measured relative to the robot-origin frame's **+X axis**. The
origin input was left at CAD **world**, whose +X was *not* robot-forward — it was ~90°
off. So every yaw was rotated by that offset (`115° − 90° ≈ 25°`).

**Fix:** drop a mate connector at the robot origin oriented **WPILib** (+X forward,
+Y left, +Z up) and select it as the FeatureScript's origin input. Yaw collapsed
115° → **25°**.

**Lesson:** an azimuth is only meaningful against a stated reference. Always pin the
robot-origin frame explicitly; never trust the CAD default world orientation.
See the mount tool: `tools/onshape/camera_mount.fs` (`f8a50c9`).

---

## Bug 2 — Calibration written to a phantom config file

**Symptom:** "I fully calibrated the camera and nothing changed."

**Root cause:** `tools/calibrate.py` read/wrote `config/cameras/camera_{id}.json`, but
those files were **deleted** when the `.jsonc` configs became canonical. The runtime
loads `config/cameras/*.jsonc` and keys on the `"id"` field (`cam1.jsonc` holds id 0).
Every "Write to JSON" landed in a file **the runtime never reads** — the live config
kept its old numbers.

**Fix:** resolve the config by matching the `"id"` field across `*.jsonc` and update
values in place, preserving comments (`916e960`).

**Lesson:** when a tool "has no effect," verify it's writing the file the runtime
actually loads *before* suspecting the math.

---

## Bug 3 — Stale / off intrinsics

**Symptom:** principal point noticeably off.

**Root cause:** the old intrinsics predated the current lens/mount; recalibration
(RMS **0.77 px**) moved the principal point by ~10 px: `cx 353.8 → 364.4`,
`cy 195.5 → 201.0`. A ~10 px principal-point error biases the whole ground-plane cast.

**Fix:** recalibrate with `tools/calibrate.py`; commit `fc32582`. Printable target
generator added at `tools/make_calib_target.py` (`dab9b69`, PDF/centered `cb39217`).

**Lesson:** print the calibration target at **100% / actual size** (not fit-to-page)
and verify the 50 mm scale bar with a ruler, or the square size the solver trusts is
silently wrong.

---

## Bug 4 — Inference letterbox never undone (the big one)

**Symptom:** range badly inflated; the debug trace showed the ball's ground-contact
pixel *above* the image center (`py=187`, `cy=201`) — i.e. the code thought a ball on
the floor 4 ft away was **above the horizon**. Physically impossible.

**Root cause:** detections reach the pose estimator in the **network 640×640**
(letterboxed) space, **not** the native 640×480 calibration frame. Proven by the
`HEIMDALL_DEBUG_PROJ` trace — the bbox fed to pose was byte-identical to the raw
`yolo_parser` output:

```
[proj] bbox(l,t,w,h)=(152.9, 292.0, 60.6, 61.0)     <- fed to pose
parser [2]  l=152.8 t=292.1 w=61.0 h=59.8  net=640x640
```

`maintain-aspect-ratio=1` fits 640×480 into 640×640 with **scale 1.0, pad_x 0,
pad_y 80**. The code un-rotated using native dims but never removed that **80 px
vertical pad**, so every `py` was 80 px off.

**Fix:** `ground_contact_native` now undoes the letterbox (scale + pad, rotation-aware)
before the rotation un-map, using new per-camera `infer_width`/`infer_height`
(`97d6d15`). `0` = detections already native (sim/replay), a no-op.

Result: `py 187 → 269` (now below center, correct), forward `1.79 → 1.35 m`.

**Lesson:** confirm which coordinate space `NvDsObjectMeta.rect_params` are actually
in. Here they were network-input space, not frame space — the letterbox must be undone
before unprojecting.

---

## Bug 5 — Lateral axis inverted (upside-down sensor roll unmodeled)

**Symptom:** with range fixed, the lateral axis was **backwards** — moving the ball
physically left drove `field_y` toward 0; centering it gave a large positive `field_y`.

**Root cause:** the camera is physically **upside-down**. `"rotation": 180` corrects
the *image* for nvinfer, but the projection unprojects in the **raw sensor frame**
(where the intrinsics live and where `ground_contact_native` lands after un-rotating).
In that frame, sensor **+X points physical-left**, not right. The extrinsics came from
Onshape with **roll = 0**, which assumes an upright sensor (+X right). The sensor's
180° roll was never modeled → the lateral term of R had the wrong sign → Y inverted.

Verified against the live trace: adding roll flips R's lateral element
`−0.906 → +0.906`, so physical-left → **+Y** as it should.

**Fix:** `"roll": 3.14159` (π) in the cam1 extrinsics (`53dcc24`).

**Lesson:** the image `rotation` enum fixes the picture for the detector; it does
**not** rotate the frame the projection reasons in. An upside-down mount needs the
sensor roll expressed in the extrinsics (roll = π), independent of `rotation`.

---

## What was NOT a bug

- **"10 detections but 1 object."** The `yolo_parser` line logs raw **pre-cluster**
  candidates (all 10 boxes were the same ball, `l≈153 t≈292`). DeepStream's NMS runs
  after the parser and collapses them — confirmed by the probe's own `objs=1`, which
  matched the video. No action needed.
- **The projection math / coordinate conventions in `project_pixel`.** Hand-checking
  the full pixel→field chain against the code reproduced the code's output exactly;
  every error was an input, not the transform.

---

## The tool that cracked it: `HEIMDALL_DEBUG_PROJ`

The turning point was instrumenting the pixel→field chain (`dd4d9be`, enabled in
compose `e77083d`). Set `HEIMDALL_DEBUG_PROJ=1` and, for the first 40 detections, it
dumps:

```
[proj] cam=0 rot=180 native_WxH=640x480 bbox(l,t,w,h)=(..) -> ground_contact_native=(px,py) | pose=(x,y,h)
[proj]   pixel=(px,py) -> undist(u,v)=(..) d_rob=(..) d_field=(..)
[proj]   cam_origin_field=(..) t=.. -> field=(x,y)
```

Read it in this order — each line kills a class of bug:
1. `pose=` — is the robot pose really what you think? (heading ~0 ruled out a stale pose.)
2. `bbox` bottom — ≤ native height, or ~net size? (caught the letterbox.)
3. `ground_contact_native` — does the pixel match a hand forward-projection?
4. `field` — does it land near ground truth?

---

## Residual / follow-ups

- **yaw/pitch magnitude.** roll=π fixed lateral *direction*; eyeballed Onshape angles
  may still be a few degrees off in magnitude. If distances need tightening, finish
  with the tool's **Extrinsics tab** (solvePnP: ground board → pitch+tz, yaw board →
  yaw) — measured beats CAD-estimated.
- **Durability idea:** fold the `rotation` enum into the extrinsics roll automatically
  in the loader (180→π, 90→π/2, …) so upside-down cameras Just Work without a manual
  `roll`. Not yet implemented.
- **Cleanup:** remove `HEIMDALL_DEBUG_PROJ` from `docker/docker-compose.yml` once the
  fix is confirmed stable.

---

## Fix commit index

| Bug | Fix commit |
|-----|-----------|
| 1 — Onshape yaw frame | `f8a50c9` (mount FeatureScript), operator fix |
| 2 — phantom config file | `916e960` |
| 3 — intrinsics | `fc32582` (+ target tooling `dab9b69`, `cb39217`) |
| 4 — letterbox | `97d6d15` |
| 5 — upside-down roll | `53dcc24` |
| diagnostics | `dd4d9be`, `e77083d` |
