package com.heimdall.proto;

import java.io.ByteArrayOutputStream;

/**
 * Hand-rolled protobuf serializer for RobotPoseMsg — no protobuf-java dependency.
 *
 * Wire format reference (proto3):
 *   RobotPoseMsg {
 *     float  x            = 1;   tag 0x0D (field 1, wire 5 = 32-bit)
 *     float  y            = 2;   tag 0x15
 *     float  heading      = 3;   tag 0x1D
 *     uint64 timestamp_ns = 4;   tag 0x20 (field 4, wire 0 = varint)
 *   }
 */
public final class ProtoWriter {
    private ProtoWriter() {}

    public static byte[] serializeRobotPose(float x, float y, float heading, long timestampNs) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(28);
        writeFloat(out, 1, x);
        writeFloat(out, 2, y);
        writeFloat(out, 3, heading);
        if (timestampNs != 0) {
            writeTag(out, 4, 0); // varint
            writeVarint(out, timestampNs);
        }
        return out.toByteArray();
    }

    private static void writeFloat(ByteArrayOutputStream out, int field, float value) {
        writeTag(out, field, 5); // wire type 5 = 32-bit little-endian
        int bits = Float.floatToRawIntBits(value);
        out.write(bits & 0xFF);
        out.write((bits >> 8) & 0xFF);
        out.write((bits >> 16) & 0xFF);
        out.write((bits >> 24) & 0xFF);
    }

    private static void writeTag(ByteArrayOutputStream out, int fieldNum, int wireType) {
        writeVarint(out, (long)((fieldNum << 3) | wireType));
    }

    private static void writeVarint(ByteArrayOutputStream out, long value) {
        while ((value & ~0x7FL) != 0) {
            out.write((int)((value & 0x7F) | 0x80));
            value >>>= 7;
        }
        out.write((int) value);
    }
}
