package com.heimdall;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.heimdall.proto.ProtoReader;
import com.heimdall.proto.ProtoVersion;
import com.heimdall.proto.ProtoWriter;
import java.io.ByteArrayOutputStream;
import org.junit.jupiter.api.Test;

/** Wire-format tests for the hand-rolled proto codec (§2H, mirrors the C++ VisionPoseMsg layout). */
class ProtoCodecTest {

    // ── minimal proto3 encoder, matching the C++ generated wire format ──────────
    private static void varint(ByteArrayOutputStream o, long v) {
        while ((v & ~0x7FL) != 0) { o.write((int)((v & 0x7F) | 0x80)); v >>>= 7; }
        o.write((int) v);
    }
    private static void tag(ByteArrayOutputStream o, int field, int wire) { varint(o, (field << 3) | wire); }
    private static void f32(ByteArrayOutputStream o, int field, float val) {
        tag(o, field, 5);
        int b = Float.floatToRawIntBits(val);
        o.write(b & 0xFF); o.write((b >> 8) & 0xFF); o.write((b >> 16) & 0xFF); o.write((b >> 24) & 0xFF);
    }
    private static void vfield(ByteArrayOutputStream o, int field, long val) { tag(o, field, 0); varint(o, val); }

    @Test
    void visionPoseParsesLatencyAndQualityFields() {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        f32(o, 1, 1.5f);                 // x
        f32(o, 2, -2.5f);                // y
        f32(o, 3, 0.25f);                // heading
        vfield(o, 4, 123456789L);        // timestamp_ns
        vfield(o, 5, 42_000_000L);       // latency_ns = 42 ms
        vfield(o, 6, 3);                 // tag_count
        f32(o, 7, 2.0f);                 // avg_tag_distance
        f32(o, 8, 0.7f);                 // reproj_error
        f32(o, 9, 0.1f);                 // ambiguity
        vfield(o, 10, 1);                // solve_mode = IPPE
        vfield(o, 15, ProtoVersion.PROTO_VERSION);

        VisionPoseEstimate v = ProtoReader.parseVisionPose(o.toByteArray());

        assertEquals(1.5, v.getX(), 1e-6);
        assertEquals(-2.5, v.getY(), 1e-6);
        assertEquals(0.25, v.getHeadingRad(), 1e-6);
        assertEquals(3, v.getTagCount());
        assertEquals(1, v.getSolveMode());
        assertEquals(0.042, v.getLatencySecs(), 1e-9);
        // FPGA timestamp = fpgaNow − latency (§2A.1).
        assertEquals(100.0 - 0.042, v.getTimestampSecs(100.0), 1e-9);
    }

    @Test
    void protoReaderRejectsOutOfRangeLengthPrefix() {
        // DetectionFrame field 1 (events) is length-delimited; claim 1000 bytes with none following.
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        tag(o, 1, 2);
        varint(o, 1000);
        assertThrows(IllegalArgumentException.class,
                () -> ProtoReader.parseDetectionFrame(o.toByteArray()));
    }

    @Test
    void robotPoseCarriesProtoVersion() {
        byte[] b = ProtoWriter.serializeRobotPose(1f, 2f, 3f, 5L, 0.1f);
        boolean hasVersionTag = false;
        for (byte x : b) if ((x & 0xFF) == 0x78) hasVersionTag = true; // field 15, wire 0
        assertTrue(hasVersionTag, "serialized RobotPose should include proto_version (field 15)");
    }
}
