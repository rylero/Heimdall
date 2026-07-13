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
 *   <li>5555 — robot PUSH → Jetson PULL (robot pose)
 *   <li>5556 — Jetson PUB → robot SUB (detection frames; latest-value, drops stale)
 *   <li>5558 — Jetson PUB → robot SUB (AprilTag vision pose)
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

    // Jetson healthy timeout: if no frame arrives within this window, isHealthy() → false.
    // Measured with System.nanoTime() (monotonic) — NOT wall-clock — so a DS/NTP time jump
    // on the rio can't flap health (§2E).
    private static final long STALE_FRAME_NS = 500_000_000L; // 500 ms

    // IO-loop wait budget. Bounds the added latency on a queued pose to this many ms
    // (was effectively the old 5 ms detection recv timeout, see ioLoop/5.3).
    private static final long POLL_TIMEOUT_MS = 2;

    private final String jetsonHost;
    private final int posePort;
    private final int detectionPort;
    private final int visionPosePort;

    private final ZContext zmqCtx = new ZContext();

    // Main thread writes here; IO thread reads and clears atomically.
    private final AtomicReference<PoseSnapshot> pendingPose = new AtomicReference<>();

    // IO thread writes; main thread reads.
    private final AtomicReference<DetectionFrame>    latestFrame     = new AtomicReference<>();
    private final AtomicLong                         lastFrameNs     = new AtomicLong(0); // System.nanoTime()
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
     * Call this from your odometry update.
     *
     * @param x          field-relative x, meters
     * @param y          field-relative y, meters
     * @param headingRad field-relative heading, radians CCW from +X
     */
    public void sendPose(double x, double y, double headingRad) {
        sendPose(x, y, headingRad, 0.0, System.nanoTime());
    }

    /**
     * Queue a robot pose with gyro angular velocity for constrained PnP on the Jetson.
     * Providing vyaw enables the Whacknet-style translation-only solve, eliminating tag ambiguity.
     *
     * @param x             field-relative x, meters
     * @param y             field-relative y, meters
     * @param headingRad    field-relative heading, radians CCW from +X
     * @param vyawRadPerSec angular velocity rad/s CCW positive (e.g. {@code Math.toRadians(navX.getRate())})
     */
    public void sendPose(double x, double y, double headingRad, double vyawRadPerSec) {
        pendingPose.set(new PoseSnapshot(x, y, headingRad, (float) vyawRadPerSec, System.nanoTime()));
    }

    /** Same as {@link #sendPose(double, double, double, double)} with an explicit timestamp. */
    public void sendPose(double x, double y, double headingRad, double vyawRadPerSec, long timestampNs) {
        pendingPose.set(new PoseSnapshot(x, y, headingRad, (float) vyawRadPerSec, timestampNs));
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
                && (System.nanoTime() - lastFrameNs.get()) < STALE_FRAME_NS;
    }

    /** Seconds since the last detection frame was received. Returns {@link Double#MAX_VALUE} before first frame. */
    public double getTimeSinceLastFrameSecs() {
        long last = lastFrameNs.get();
        return last == 0 ? Double.MAX_VALUE : (System.nanoTime() - last) / 1e9;
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
        ZMQ.Socket push        = zmqCtx.createSocket(SocketType.PUSH);
        ZMQ.Socket trackSub    = zmqCtx.createSocket(SocketType.SUB);
        ZMQ.Socket visionSub   = zmqCtx.createSocket(SocketType.SUB);
        ZMQ.Poller poller      = zmqCtx.createPoller(2);
        try {
            push.connect("tcp://" + jetsonHost + ":" + posePort);

            // Conflate: keep only the newest message per sub, matching the Jetson PUB's
            // latest-value design so a momentary stall reads fresh data, not stale backlog
            // (5.4). Must be set before connect. Non-blocking recv; the Poller does the wait.
            trackSub.setConflate(true);
            trackSub.setReceiveTimeOut(0);
            trackSub.connect("tcp://" + jetsonHost + ":" + detectionPort);
            trackSub.subscribe(new byte[0]);

            visionSub.setConflate(true);
            visionSub.setReceiveTimeOut(0);
            visionSub.connect("tcp://" + jetsonHost + ":" + visionPosePort);
            visionSub.subscribe(new byte[0]);

            poller.register(trackSub, ZMQ.Poller.POLLIN);
            poller.register(visionSub, ZMQ.Poller.POLLIN);

            while (running) {
                // 1. Send the pending pose FIRST, every iteration — the send path is no longer
                //    gated behind a blocking detection recv (5.3). Previously a pose set just
                //    after recv() started waited the full 5 ms poll timeout before leaving.
                PoseSnapshot pose = pendingPose.getAndSet(null);
                if (pose != null) {
                    byte[] bytes = ProtoWriter.serializeRobotPose(
                            (float) pose.x, (float) pose.y,
                            (float) pose.headingRad, pose.timestampNs, pose.vyaw);
                    push.send(bytes, ZMQ.DONTWAIT);
                }

                // 2. Wait for either sub, but wake often enough to keep pose latency low.
                //    A new pose set during this poll waits at most POLL_TIMEOUT_MS, not 5 ms.
                poller.poll(POLL_TIMEOUT_MS);

                if (poller.pollin(0)) {
                    byte[] data = trackSub.recv(ZMQ.DONTWAIT);
                    if (data != null && data.length > 0) {
                        try {
                            latestFrame.set(ProtoReader.parseDetectionFrame(data));
                            lastFrameNs.set(System.nanoTime());
                        } catch (Exception ignored) {
                            // malformed message — drop silently
                        }
                    }
                }

                if (poller.pollin(1)) {
                    byte[] vdata = visionSub.recv(ZMQ.DONTWAIT);
                    if (vdata != null && vdata.length > 0) {
                        try {
                            latestVisionPose.set(ProtoReader.parseVisionPose(vdata));
                        } catch (Exception ignored) {}
                    }
                }
            }
        } catch (org.zeromq.ZMQException e) {
            // ETERM = context closed during recv; expected on shutdown
            if (e.getErrorCode() != zmq.ZError.ETERM) {
                System.err.println("[heimdall] IO thread ZMQ error: " + e.getMessage());
            }
        } finally {
            push.close();
            trackSub.close();
            visionSub.close();
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private static final class PoseSnapshot {
        final double x, y, headingRad;
        final float  vyaw;
        final long   timestampNs;

        PoseSnapshot(double x, double y, double headingRad, float vyaw, long timestampNs) {
            this.x           = x;
            this.y           = y;
            this.headingRad  = headingRad;
            this.vyaw        = vyaw;
            this.timestampNs = timestampNs;
        }
    }
}
