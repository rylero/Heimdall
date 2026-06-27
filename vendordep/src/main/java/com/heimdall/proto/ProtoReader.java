package com.heimdall.proto;

import com.heimdall.DetectionFrame;
import com.heimdall.TrackEvent;
import com.heimdall.TrackEventType;
import com.heimdall.TrackedObject;
import com.heimdall.VisionPoseEstimate;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-rolled protobuf parser for DetectionFrameMsg — no protobuf-java dependency.
 *
 * Wire layout (proto3):
 *   DetectionFrameMsg {
 *     repeated TrackEventMsg events       = 1;  tag 0x0A (wire 2 = length-delimited)
 *     uint64                 timestamp_ns = 2;  tag 0x10 (wire 0 = varint)
 *     bool                   healthy      = 3;  tag 0x18 (wire 0 = varint)
 *   }
 *   TrackEventMsg {
 *     TrackEventTypeMsg type   = 1;  tag 0x08 (wire 0)
 *     TrackedObjectMsg  object = 2;  tag 0x12 (wire 2)
 *   }
 *   TrackedObjectMsg {
 *     uint32 track_id   = 1;  tag 0x08 (wire 0)
 *     int32  class_id   = 2;  tag 0x10 (wire 0)
 *     float  x          = 3;  tag 0x1D (wire 5)
 *     float  y          = 4;  tag 0x25 (wire 5)
 *     float  vx         = 5;  tag 0x2D (wire 5)
 *     float  vy         = 6;  tag 0x35 (wire 5)
 *     float  confidence = 7;  tag 0x3D (wire 5)
 *     float  ax         = 8;  tag 0x45 (wire 5)
 *     float  ay         = 9;  tag 0x4D (wire 5)
 *   }
 */
public final class ProtoReader {
    private ProtoReader() {}

    /**
     * Parses a VisionPoseMsg (wire-identical to RobotPoseMsg: x/y/heading floats + timestamp_ns varint).
     *
     *   VisionPoseMsg {
     *     float  x            = 1;  tag 0x0D (field 1, wire 5)
     *     float  y            = 2;  tag 0x15
     *     float  heading      = 3;  tag 0x1D
     *     uint64 timestamp_ns = 4;  tag 0x20 (varint)
     *   }
     */
    public static VisionPoseEstimate parseVisionPose(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        float x = 0, y = 0, heading = 0;
        long  timestampNs = 0;

        while (buf.hasRemaining()) {
            long tag     = readVarint(buf);
            int  field   = (int)(tag >>> 3);
            int  wireType = (int)(tag & 0x7);

            switch (field) {
                case 1: x           = buf.getFloat();   break;
                case 2: y           = buf.getFloat();   break;
                case 3: heading     = buf.getFloat();   break;
                case 4: timestampNs = readVarint(buf);  break;
                default: skipField(buf, wireType);       break;
            }
        }
        return new VisionPoseEstimate(x, y, heading, timestampNs);
    }

    public static DetectionFrame parseDetectionFrame(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        List<TrackEvent> events = new ArrayList<>();
        long timestampNs = 0;
        boolean healthy = false; // proto3 default for bool

        while (buf.hasRemaining()) {
            long tag = readVarint(buf);
            int field    = (int)(tag >>> 3);
            int wireType = (int)(tag & 0x7);

            switch (field) {
                case 1: { // repeated TrackEventMsg
                    int len = (int) readVarint(buf);
                    byte[] embedded = new byte[len];
                    buf.get(embedded);
                    events.add(parseTrackEvent(embedded));
                    break;
                }
                case 2: // timestamp_ns
                    timestampNs = readVarint(buf);
                    break;
                case 3: // healthy
                    healthy = readVarint(buf) != 0;
                    break;
                default:
                    skipField(buf, wireType);
                    break;
            }
        }
        return new DetectionFrame(events, timestampNs, healthy);
    }

    private static TrackEvent parseTrackEvent(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        TrackEventType type = TrackEventType.CONFIRMED;
        TrackedObject obj = new TrackedObject(0, 0, 0, 0, 0, 0, 0, 0, 0);

        while (buf.hasRemaining()) {
            long tag = readVarint(buf);
            int field    = (int)(tag >>> 3);
            int wireType = (int)(tag & 0x7);

            switch (field) {
                case 1: { // TrackEventTypeMsg enum: 0=CONFIRMED 1=UPDATED 2=LOST
                    int t = (int) readVarint(buf);
                    type = t == 2 ? TrackEventType.LOST
                         : t == 1 ? TrackEventType.UPDATED
                         : TrackEventType.CONFIRMED;
                    break;
                }
                case 2: { // TrackedObjectMsg
                    int len = (int) readVarint(buf);
                    byte[] embedded = new byte[len];
                    buf.get(embedded);
                    obj = parseTrackedObject(embedded);
                    break;
                }
                default:
                    skipField(buf, wireType);
                    break;
            }
        }
        return new TrackEvent(type, obj);
    }

    private static TrackedObject parseTrackedObject(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int trackId = 0, classId = 0;
        float x = 0, y = 0, vx = 0, vy = 0, ax = 0, ay = 0, confidence = 0;

        while (buf.hasRemaining()) {
            long tag = readVarint(buf);
            int field    = (int)(tag >>> 3);
            int wireType = (int)(tag & 0x7);

            switch (field) {
                case 1: trackId    = (int) readVarint(buf); break;
                case 2: classId    = (int) readVarint(buf); break;
                case 3: x          = buf.getFloat();         break;
                case 4: y          = buf.getFloat();         break;
                case 5: vx         = buf.getFloat();         break;
                case 6: vy         = buf.getFloat();         break;
                case 7: confidence = buf.getFloat();         break;
                case 8: ax         = buf.getFloat();         break;
                case 9: ay         = buf.getFloat();         break;
                default: skipField(buf, wireType);           break;
            }
        }
        return new TrackedObject(trackId, classId, x, y, vx, vy, ax, ay, confidence);
    }

    private static long readVarint(ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    private static void skipField(ByteBuffer buf, int wireType) {
        switch (wireType) {
            case 0: readVarint(buf);                                              break;
            case 1: if (buf.remaining() >= 8) buf.getLong();                     break;
            case 2: { int n = (int) readVarint(buf);
                      buf.position(Math.min(buf.position() + n, buf.limit())); } break;
            case 5: if (buf.remaining() >= 4) buf.getInt();                      break;
            default: break;
        }
    }
}
