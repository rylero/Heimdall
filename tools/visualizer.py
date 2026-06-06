"""
Heimdall top-down field visualizer.

Connects to a running Heimdall instance over ZeroMQ:
  - PUSH robot pose (x=0, y=0) to Jetson on port 5555
  - PULL DetectionFrameMsg from Jetson on port 5556

Dependencies: pip install pyzmq pygame

Usage:
  python tools/visualizer.py [JETSON_IP]        (default: 127.0.0.1)
"""

import sys
import struct
import time
import threading
import collections
import zmq
import pygame

JETSON_IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
POSE_PORT  = 5555   # we PUSH robot pose here
TRACK_PORT = 5556   # we PULL detection frames from here

POSE_HZ = 50        # how often to send pose

# 2025 FRC field dimensions in metres (Reefscape)
FIELD_W = 16.54
FIELD_H = 8.21

MARGIN = 40
SCALE  = 80   # pixels per metre (mouse wheel to zoom)

WIN_W = 1400
WIN_H = 800

EVENT_CONFIRMED = 0
EVENT_UPDATED   = 1
EVENT_LOST      = 2

TRACK_COLORS = [
    (0, 200, 255), (255, 180, 0), (0, 255, 120), (255, 80, 80),
    (200, 0, 255), (255, 255, 0), (0, 255, 255), (255, 120, 200),
]


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format decoder (no protoc needed)
# ---------------------------------------------------------------------------

def _varint(data: bytes, pos: int):
    result = shift = 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _parse_tracked_object(data: bytes) -> dict:
    obj = dict(track_id=0, class_id=0, x=0.0, y=0.0,
               vx=0.0, vy=0.0, ax=0.0, ay=0.0, confidence=0.0)
    pos = 0
    while pos < len(data):
        tag, pos = _varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(data, pos)
            if field == 1: obj['track_id'] = val
            elif field == 2: obj['class_id'] = val & 0xFFFFFFFF
        elif wire == 5:
            val = struct.unpack_from('<f', data, pos)[0]; pos += 4
            if field == 3: obj['x'] = val
            elif field == 4: obj['y'] = val
            elif field == 5: obj['vx'] = val
            elif field == 6: obj['vy'] = val
            elif field == 7: obj['confidence'] = val
            elif field == 8: obj['ax'] = val
            elif field == 9: obj['ay'] = val
        elif wire == 2:
            ln, pos = _varint(data, pos); pos += ln
        elif wire == 1:
            pos += 8
    return obj


def _parse_track_event(data: bytes) -> dict:
    ev = dict(type=EVENT_UPDATED, object=None)
    pos = 0
    while pos < len(data):
        tag, pos = _varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(data, pos)
            if field == 1: ev['type'] = val
        elif wire == 2:
            ln, pos = _varint(data, pos)
            sub = data[pos:pos + ln]; pos += ln
            if field == 2: ev['object'] = _parse_tracked_object(sub)
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
    return ev


def parse_detection_frame(data: bytes) -> dict:
    frame = dict(events=[], timestamp_ns=0, healthy=True)
    pos = 0
    while pos < len(data):
        tag, pos = _varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(data, pos)
            if field == 2: frame['timestamp_ns'] = val
            elif field == 3: frame['healthy'] = bool(val)
        elif wire == 2:
            ln, pos = _varint(data, pos)
            sub = data[pos:pos + ln]; pos += ln
            if field == 1: frame['events'].append(_parse_track_event(sub))
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
    return frame


def encode_robot_pose(x: float, y: float, heading: float) -> bytes:
    def varint(v: int) -> bytes:
        out = b''
        v = v & 0xFFFFFFFFFFFFFFFF
        while v > 0x7F:
            out += bytes([v & 0x7F | 0x80]); v >>= 7
        return out + bytes([v])

    ts = int(time.monotonic_ns())
    return (
        struct.pack('<Bf', (1 << 3) | 5, x) +
        struct.pack('<Bf', (2 << 3) | 5, y) +
        struct.pack('<Bf', (3 << 3) | 5, heading) +
        bytes([(4 << 3) | 0]) + varint(ts)
    )


# ---------------------------------------------------------------------------
# ZeroMQ threads
# ---------------------------------------------------------------------------

_latest_frame: dict | None = None
_frame_lock = threading.Lock()
_healthy = False


def _pose_sender(ip: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{ip}:{POSE_PORT}")
    interval = 1.0 / POSE_HZ
    while True:
        msg = encode_robot_pose(0.0, 0.0, 0.0)
        try:
            sock.send(msg, zmq.NOBLOCK)
        except zmq.Again:
            pass
        time.sleep(interval)


def _frame_receiver(ip: str):
    global _latest_frame, _healthy
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{ip}:{TRACK_PORT}")
    while True:
        try:
            data = sock.recv()
            frame = parse_detection_frame(data)
            with _frame_lock:
                _latest_frame = frame
                _healthy = frame.get('healthy', False)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[receiver] {e}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_scale = SCALE   # mutable — changed by mouse wheel

def field_to_screen(fx: float, fy: float) -> tuple[int, int]:
    """Field coords → screen pixels. Origin (0,0) is at screen centre."""
    sx = int(WIN_W // 2 + fx * _scale)
    sy = int(WIN_H // 2 - fy * _scale)
    return sx, sy


def draw_field(surf: pygame.Surface, font: pygame.font.Font):
    surf.fill((50, 80, 50))

    # Centre line
    pygame.draw.line(surf,
                     (80, 120, 80),
                     field_to_screen(FIELD_W / 2, 0),
                     field_to_screen(FIELD_W / 2, FIELD_H), 1)

    # Origin crosshair
    ox, oy = field_to_screen(0, 0)
    pygame.draw.line(surf, (80, 80, 160), (ox - 12, oy), (ox + 12, oy), 1)
    pygame.draw.line(surf, (80, 80, 160), (ox, oy - 12), (ox, oy + 12), 1)


def draw_robot(surf: pygame.Surface):
    sx, sy = field_to_screen(0.0, 0.0)
    pygame.draw.rect(surf, (0, 180, 255),
                     pygame.Rect(sx - 10, sy - 10, 20, 20), 0)
    pygame.draw.rect(surf, (255, 255, 255),
                     pygame.Rect(sx - 10, sy - 10, 20, 20), 2)


def draw_tracks(surf: pygame.Surface, font: pygame.font.Font,
                tracks: dict, now: float):
    FADE_S = 0.4   # seconds to fade after last update before removal
    to_delete = [tid for tid, info in tracks.items()
                 if now - info['last_seen'] > FADE_S or info['type'] == EVENT_LOST]
    for tid in to_delete:
        del tracks[tid]

    for tid, info in list(tracks.items()):
        age = now - info['last_seen']
        obj = info['obj']
        ev_type = info['type']

        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        if age > 0.1:
            alpha = max(60, int(255 * (1.0 - age / FADE_S)))
            color = tuple(int(c * alpha / 255) for c in color)

        sx, sy = field_to_screen(obj['x'], obj['y'])

        # velocity arrow
        vx, vy = obj['vx'], obj['vy']
        spd = (vx**2 + vy**2) ** 0.5
        if spd > 0.05:
            ex, ey = field_to_screen(obj['x'] + vx * 0.5,
                                     obj['y'] + vy * 0.5)
            pygame.draw.line(surf, color, (sx, sy), (ex, ey), 2)

        # object dot
        pygame.draw.circle(surf, color, (sx, sy), 10)
        pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 10, 1)

        # label: id + confidence
        label = f"#{tid} c{obj['class_id']} {obj['confidence']:.2f}"
        lbl_surf = font.render(label, True, color)
        surf.blit(lbl_surf, (sx + 12, sy - 8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    threading.Thread(target=_pose_sender, args=(JETSON_IP,), daemon=True).start()
    threading.Thread(target=_frame_receiver, args=(JETSON_IP,), daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Heimdall — {JETSON_IP}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 12)
    big_font = pygame.font.SysFont("monospace", 16, bold=True)

    # track_id → {obj, type, last_seen}
    tracks: dict = {}
    frames_rx = 0
    last_frame_time = 0.0
    last_frame_ts = -1   # timestamp_ns of last processed frame

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit(); return
            if ev.type == pygame.MOUSEWHEEL:
                global _scale
                _scale = max(10, min(400, int(_scale * (1.1 if ev.y > 0 else 0.9))))

        with _frame_lock:
            frame = _latest_frame

        now = time.monotonic()

        if frame is not None and frame['timestamp_ns'] != last_frame_ts:
            last_frame_ts = frame['timestamp_ns']
            for event in frame['events']:
                obj = event.get('object')
                if obj is None:
                    continue
                tid = obj['track_id']
                tracks[tid] = dict(obj=obj, type=event['type'], last_seen=now)
            frames_rx += 1
            last_frame_time = now

        draw_field(screen, font)
        draw_robot(screen)
        draw_tracks(screen, font, tracks, now)

        # HUD
        age = now - last_frame_time if last_frame_time else 999
        healthy_str = "OK" if _healthy else "UNHEALTHY"
        hud_color = (0, 255, 120) if _healthy and age < 0.5 else (255, 80, 80)
        active = sum(1 for t in tracks.values()
                     if now - t['last_seen'] < 0.5 and t['type'] != EVENT_LOST)
        hud = big_font.render(
            f"Heimdall {JETSON_IP}  {healthy_str}  frames={frames_rx}"
            f"  age={age:.2f}s  active={active}",
            True, hud_color)
        screen.blit(hud, (MARGIN, 8))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
