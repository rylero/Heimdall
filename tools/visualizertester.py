"""
Heimdall simulator — counterpart to visualizer.py.

Binds on the same ports Heimdall uses:
  - PULL  on 5555  (receives robot pose from visualizer — logged but ignored)
  - PUSH  on 5556  (sends DetectionFrameMsg with simulated moving objects)

Run this on the same machine as the visualizer (or any reachable host):
  python tools/visualizertester.py

Then run the visualizer pointing at this machine:
  python tools/visualizer.py 127.0.0.1

Dependencies: pip install pyzmq
"""

import math
import struct
import time
import threading
import zmq

POSE_PORT  = 5555
TRACK_PORT = 5556
FRAME_HZ   = 30

FIELD_W = 16.54
FIELD_H = 8.21

EVENT_CONFIRMED = 0
EVENT_UPDATED   = 1
EVENT_LOST      = 2


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format encoder
# ---------------------------------------------------------------------------

def _varint(v: int) -> bytes:
    v &= 0xFFFFFFFFFFFFFFFF
    out = b''
    while v > 0x7F:
        out += bytes([v & 0x7F | 0x80]); v >>= 7
    return out + bytes([v])


def _encode_tracked_object(obj: dict) -> bytes:
    d = b''
    d += bytes([(1 << 3) | 0]) + _varint(obj['track_id'])
    d += bytes([(2 << 3) | 0]) + _varint(obj['class_id'] & 0xFFFFFFFF)
    d += struct.pack('<Bf', (3 << 3) | 5, obj['x'])
    d += struct.pack('<Bf', (4 << 3) | 5, obj['y'])
    d += struct.pack('<Bf', (5 << 3) | 5, obj.get('vx', 0.0))
    d += struct.pack('<Bf', (6 << 3) | 5, obj.get('vy', 0.0))
    d += struct.pack('<Bf', (7 << 3) | 5, obj.get('confidence', 0.9))
    d += struct.pack('<Bf', (8 << 3) | 5, obj.get('ax', 0.0))
    d += struct.pack('<Bf', (9 << 3) | 5, obj.get('ay', 0.0))
    return d


def _encode_track_event(ev_type: int, obj: dict) -> bytes:
    d = b''
    d += bytes([(1 << 3) | 0]) + _varint(ev_type)
    obj_bytes = _encode_tracked_object(obj)
    d += bytes([(2 << 3) | 2]) + _varint(len(obj_bytes)) + obj_bytes
    return d


def encode_detection_frame(events: list[tuple[int, dict]],
                            healthy: bool = True) -> bytes:
    d = b''
    for ev_type, obj in events:
        ev_bytes = _encode_track_event(ev_type, obj)
        d += bytes([(1 << 3) | 2]) + _varint(len(ev_bytes)) + ev_bytes
    ts = int(time.monotonic_ns())
    d += bytes([(2 << 3) | 0]) + _varint(ts)
    d += bytes([(3 << 3) | 0]) + _varint(1 if healthy else 0)
    return d


# ---------------------------------------------------------------------------
# Simulated object trajectories
# ---------------------------------------------------------------------------

class SimObject:
    def __init__(self, track_id: int, class_id: int, kind: str, **kwargs):
        self.track_id = track_id
        self.class_id = class_id
        self.kind = kind
        self.kw = kwargs
        # disappear/reappear simulation
        self.visible = True
        self.hide_at: float | None = None
        self.show_at: float | None = None
        self.confirmed = False

    def state(self, t: float) -> dict:
        """Return (x, y, vx, vy) for time t."""
        if self.kind == 'circle':
            cx = self.kw['cx']; cy = self.kw['cy']
            r  = self.kw['r'];  w  = self.kw['w']
            phase = self.kw.get('phase', 0.0)
            x  = cx + r * math.cos(w * t + phase)
            y  = cy + r * math.sin(w * t + phase)
            vx = -r * w * math.sin(w * t + phase)
            vy =  r * w * math.cos(w * t + phase)

        elif self.kind == 'bounce':
            # linear bounce between x0↔x1, y0↔y1
            x0 = self.kw['x0']; x1 = self.kw['x1']
            y0 = self.kw['y0']; y1 = self.kw['y1']
            period = self.kw['period']
            frac = (math.sin(2 * math.pi * t / period) + 1) / 2
            x  = x0 + frac * (x1 - x0)
            y  = y0 + frac * (y1 - y0)
            dfrac = math.cos(2 * math.pi * t / period) * math.pi / period
            vx = dfrac * (x1 - x0)
            vy = dfrac * (y1 - y0)

        elif self.kind == 'figure8':
            a = self.kw['a']; b = self.kw['b']
            cx = self.kw['cx']; cy = self.kw['cy']
            w  = self.kw['w']
            x  = cx + a * math.sin(w * t)
            y  = cy + b * math.sin(2 * w * t)
            vx = a * w * math.cos(w * t)
            vy = 2 * b * w * math.cos(2 * w * t)

        elif self.kind == 'drift':
            # slow diagonal drift, wraps field
            sx = self.kw['sx']; sy = self.kw['sy']
            ox = self.kw.get('ox', 0.0); oy = self.kw.get('oy', 0.0)
            x  = (ox + sx * t) % FIELD_W
            y  = (oy + sy * t) % FIELD_H
            vx = sx; vy = sy

        else:
            x = y = vx = vy = 0.0

        return dict(
            track_id=self.track_id,
            class_id=self.class_id,
            x=float(x), y=float(y),
            vx=float(vx), vy=float(vy),
            confidence=0.85 + 0.1 * math.sin(t * 3.7 + self.track_id),
        )


# Define the simulated objects on the field.
OBJECTS = [
    # Circular orbit — game piece near blue side
    SimObject(0, 0, 'circle', cx=4.0, cy=4.1, r=1.8, w=0.6, phase=0.0),
    # Circular orbit — game piece near red side, opposite direction
    SimObject(1, 0, 'circle', cx=12.5, cy=4.1, r=1.5, w=-0.7, phase=1.2),
    # Bouncing across field — robot-like object
    SimObject(2, 1, 'bounce', x0=2.0, y0=2.0, x1=14.5, y1=6.2, period=8.0),
    # Figure-8 around centre
    SimObject(3, 0, 'figure8', cx=8.27, cy=4.1, a=3.0, b=1.6, w=0.4),
    # Slow drift — debris / stray note
    SimObject(4, 0, 'drift', sx=0.5, sy=0.3, ox=1.0, oy=1.0),
    # Another robot circling their side
    SimObject(5, 1, 'circle', cx=13.0, cy=2.0, r=1.0, w=1.1, phase=3.0),
    # Short-lived object that disappears and reappears
    SimObject(6, 0, 'bounce', x0=6.0, y0=5.5, x1=10.0, y1=5.5, period=3.0),
]

# Object 6 goes missing every 6 seconds for 2 seconds.
OBJECTS[6].hide_at  = 6.0
OBJECTS[6].show_at  = 8.0


# ---------------------------------------------------------------------------
# Receiver thread — just prints received pose so you can see it working
# ---------------------------------------------------------------------------

def _pose_receiver():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(f"tcp://*:{POSE_PORT}")
    while True:
        try:
            sock.recv()   # consume; ignore content for the simulator
        except zmq.Again:
            pass


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def main():
    threading.Thread(target=_pose_receiver, daemon=True).start()

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind(f"tcp://*:{TRACK_PORT}")

    print(f"Heimdall simulator listening — PULL:{POSE_PORT}  PUSH:{TRACK_PORT}")
    print("Run visualizer with:  python tools/visualizer.py 127.0.0.1")

    interval = 1.0 / FRAME_HZ
    t0 = time.monotonic()

    # per-object visibility state
    hide_phase: dict[int, float | None] = {o.track_id: None for o in OBJECTS}

    while True:
        t = time.monotonic() - t0
        events: list[tuple[int, dict]] = []

        for obj in OBJECTS:
            tid = obj.track_id

            # simple periodic hide/show for object 6
            if obj.hide_at is not None:
                cycle = t % obj.show_at
                was_hidden = hide_phase.get(tid) == 'hidden'
                is_hidden  = cycle < obj.hide_at

                if is_hidden and not was_hidden:
                    # emit LOST
                    state = obj.state(t)
                    events.append((EVENT_LOST, state))
                    hide_phase[tid] = 'hidden'
                    obj.confirmed = False
                    continue
                elif not is_hidden and was_hidden:
                    hide_phase[tid] = 'visible'
                    obj.confirmed = False
                elif is_hidden:
                    continue

            state = obj.state(t)

            if not obj.confirmed:
                events.append((EVENT_CONFIRMED, state))
                obj.confirmed = True
            else:
                events.append((EVENT_UPDATED, state))

        frame_bytes = encode_detection_frame(events, healthy=True)
        try:
            push.send(frame_bytes, zmq.NOBLOCK)
        except zmq.Again:
            pass

        time.sleep(interval)


if __name__ == "__main__":
    main()
