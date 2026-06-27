package com.heimdall;

import com.heimdall.proto.ProtoReader;
import com.heimdall.proto.ProtoWriter;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQException;

/**
 * ZeroMQ client that talks to a running Heimdall instance on the Jetson.
 *
 * <p>Ports (must match heimdall's CommLayer::Config):
 * <ul>
 *   <li>5555 — robot PUSH → Jetson PULL (robot pose at ~50 Hz)
 *   <li>5556 — Jetson PUSH → robot PULL (detection frames, ~30 Hz)
 * </ul>
 *
 * <p>All socket I/O runs on a daemon background thread — safe to call
 * {@link #sendPose} and {@link #getLatestFrame} from any robot thread.
 *
 * <p>Typical usage inside a WPILib subsystem:
 * <pre>{@code
 *   private final HeimdallClient heimdall = new HeimdallClient("10.42.0.2");
 *
 *   @Override public void periodic() {
 *       heimdall.sendPose(odometry.getX(), odometry.getY(),
 *                         odometry.getRotation().getRadians());
 *       DetectionFrame frame = heimdall.getLatestFrame();
 *       if (frame != null && frame.isHealthy()) { ... }
 *   }
 * }</pre>
 */
public final class HeimdallClient implements AutoCloseable {

    public static final int DEFAULT_POSE_PORT       = 5555;
    public static final int DEFAULT_DETECTION_PORT  = 5556;
    public static final int DEFAULT_VISION_POSE_PORT = 5558;

    // Jetson healthy timeout: if no frame arrives within this many ms, isHealthy() → false.
    private static final long STALE_FRAME_MS = 500;

    private final String jetsonHost;
    private final int posePort;
    private final int detectionPort;
    private final int visionPosePort;

    private final ZContext zmqCtx = new ZContext();

    // Main thread writes here; IO thread reads and clears atomically.
    private final AtomicReference<PoseSnapshot> pendingPose = new AtomicReference<>();

    // IO thread writes; main thread reads.
    private final AtomicReference<DetectionFrame>    latestFrame     = new AtomicReference<>();
    private final AtomicLong                         lastFrameMs     = new AtomicLong(0);
    private final AtomicReference<VisionPoseEstimate> latestVisionPose = new AtomicReference<>();

    private volatile boolean running = true;
    private final Thread ioThread;

    public HeimdallClient(String jetsonHost) {
        this(jetsonHost, DEFAULT_POSE_PORT, DEFAULT_DETECTION_PORT, DEFAULT_VISION_POSE_PORT);
    }

    public HeimdallClient(String jetsonHost, int posePort, int detectionPort, int visionPosePort) {
        this.jetsonHost    = jetsonHost;
        this.posePort      = posePort;
        this.detectionPort = detectionPort;
        this.visionPosePort = visionPosePort;

        ioThread = new Thread(this::ioLoop, "heimdall-io");
        ioThread.setDaemon(true);
        ioThread.start();
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Queue a robot pose to be sent to the Jetson.
     * Call this from your odometry update (matches RoboRIO ~50 Hz control loop).
     *
     * @param x          field-relative x, meters
     * @param y          field-relative y, meters
     * @param headingRad field-relative heading, radians CCW from +X
     */
    public void sendPose(double x, double y, double headingRad) {
        sendPose(x, y, headingRad, System.nanoTime());
    }

    /** Same as {@link #sendPose(double, double, double)} with an explicit timestamp. */
    public void sendPose(double x, double y, double headingRad, long timestampNs) {
        pendingPose.set(new PoseSnapshot(x, y, headingRad, timestampNs));
    }

    /**
     * Returns the most recent detection frame from the Jetson, or {@code null}
     * if no frame has been received yet.
     */
    public DetectionFrame getLatestFrame() {
        return latestFrame.get();
    }

    /**
     * Returns the most recent AprilTag-derived vision pose from the Jetson, or {@code null}
     * if no estimate has arrived yet. Each call returns the same object until a new one arrives.
     * Consume this in periodic() and call drive.addVisionMeasurement().
     */
    public VisionPoseEstimate getLatestVisionPose() {
        return latestVisionPose.getAndSet(null); // consume once
    }

    /**
     * True when the Jetson pipeline is running and frames are arriving.
     * Becomes false if no frame arrives within 500 ms or the last frame had healthy=false.
     */
    public boolean isHealthy() {
        DetectionFrame f = latestFrame.get();
        return f != null && f.isHealthy()
                && (System.currentTimeMillis() - lastFrameMs.get()) < STALE_FRAME_MS;
    }

    /** Seconds since the last detection frame was received. Returns {@link Double#MAX_VALUE} before first frame. */
    public double getTimeSinceLastFrameSecs() {
        long last = lastFrameMs.get();
        return last == 0 ? Double.MAX_VALUE : (System.currentTimeMillis() - last) / 1000.0;
    }

    /** Whether the background IO thread is alive (does NOT mean the Jetson is reachable). */
    public boolean isConnected() {
        return ioThread.isAlive();
    }

    @Override
    public void close() {
        running = false;
        zmqCtx.close(); // unblocks any recv() call in the IO thread via ETERM
        try {
            ioThread.join(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // -------------------------------------------------------------------------
    // Background IO thread
    // -------------------------------------------------------------------------

    private void ioLoop() {
        ZMQ.Socket push       = zmqCtx.createSocket(SocketType.PUSH);
        ZMQ.Socket pull       = zmqCtx.createSocket(SocketType.PULL);
        ZMQ.Socket visionSub  = zmqCtx.createSocket(SocketType.SUB);
        try {
            push.connect("tcp://" + jetsonHost + ":" + posePort);
            pull.setReceiveTimeOut(5); // 5 ms receive timeout → ~200 Hz poll rate
            pull.connect("tcp://" + jetsonHost + ":" + detectionPort);
            visionSub.setReceiveTimeOut(0); // non-blocking
            visionSub.connect("tcp://" + jetsonHost + ":" + visionPosePort);
            visionSub.subscribe(new byte[0]); // subscribe to all messages

            while (running) {
                // 1. Drain pending pose — take the most recent one, discard older.
                PoseSnapshot pose = pendingPose.getAndSet(null);
                if (pose != null) {
                    byte[] bytes = ProtoWriter.serializeRobotPose(
                            (float) pose.x, (float) pose.y,
                            (float) pose.headingRad, pose.timestampNs);
                    push.send(bytes, ZMQ.DONTWAIT);
                }

                // 2. Receive detection frame (blocks up to 5 ms).
                byte[] data = pull.recv(0);
                if (data != null && data.length > 0) {
                    try {
                        DetectionFrame frame = ProtoReader.parseDetectionFrame(data);
                        latestFrame.set(frame);
                        lastFrameMs.set(System.currentTimeMillis());
                    } catch (Exception ignored) {
                        // malformed message — drop silently
                    }
                }

                // 3. Poll vision pose (non-blocking, ~10 Hz from Jetson).
                byte[] vdata = visionSub.recv(ZMQ.DONTWAIT);
                if (vdata != null && vdata.length > 0) {
                    try {
                        latestVisionPose.set(ProtoReader.parseVisionPose(vdata));
                    } catch (Exception ignored) {}
                }
            }
        } catch (org.zeromq.ZMQException e) {
            // ETERM = context closed during recv; expected on shutdown
            if (e.getErrorCode() != zmq.ZError.ETERM) {
                System.err.println("[heimdall] IO thread ZMQ error: " + e.getMessage());
            }
        } finally {
            push.close();
            pull.close();
            visionSub.close();
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private static final class PoseSnapshot {
        final double x, y, headingRad;
        final long timestampNs;

        PoseSnapshot(double x, double y, double headingRad, long timestampNs) {
            this.x           = x;
            this.y           = y;
            this.headingRad  = headingRad;
            this.timestampNs = timestampNs;
        }
    }
}
