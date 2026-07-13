package com.heimdall;

/**
 * AprilTag-derived robot pose estimate received from the Jetson.
 *
 * <p>Carries latency + solve-quality metadata so the robot can (a) place the measurement in its
 * own FPGA timebase without any clock-sync protocol and (b) set dynamic std devs.
 *
 * <p>Typical use:
 * <pre>{@code
 *   VisionPoseEstimate v = heimdall.getLatestVisionPose();
 *   if (v != null) {
 *       double[] sd = v.suggestedStdDevs();
 *       drive.addVisionMeasurement(
 *           new Pose2d(v.getX(), v.getY(), new Rotation2d(v.getHeadingRad())),
 *           v.getTimestampSecs(Timer.getFPGATimestamp()),
 *           VecBuilder.fill(sd[0], sd[0], sd[1]));
 *   }
 * }</pre>
 */
public final class VisionPoseEstimate {
    private final double x;
    private final double y;
    private final double headingRad;
    private final long   timestampNs;      // Jetson capture time (CLOCK_MONOTONIC)
    private final long   latencyNs;        // age of the estimate when the Jetson sent it
    private final int    tagCount;
    private final double avgTagDistance;   // meters
    private final double reprojError;      // px
    private final double ambiguity;        // 0 = unambiguous
    private final int    solveMode;        // 0 = gyro-constrained, 1 = IPPE fallback

    public VisionPoseEstimate(double x, double y, double headingRad, long timestampNs,
                              long latencyNs, int tagCount, double avgTagDistance,
                              double reprojError, double ambiguity, int solveMode) {
        this.x              = x;
        this.y              = y;
        this.headingRad     = headingRad;
        this.timestampNs    = timestampNs;
        this.latencyNs      = latencyNs;
        this.tagCount       = tagCount;
        this.avgTagDistance = avgTagDistance;
        this.reprojError    = reprojError;
        this.ambiguity      = ambiguity;
        this.solveMode      = solveMode;
    }

    public double getX()              { return x; }
    public double getY()              { return y; }
    public double getHeadingRad()     { return headingRad; }
    public long   getTimestampNs()    { return timestampNs; }
    public long   getLatencyNs()      { return latencyNs; }
    public double getLatencySecs()    { return latencyNs * 1e-9; }
    public int    getTagCount()       { return tagCount; }
    public double getAvgTagDistance() { return avgTagDistance; }
    public double getReprojError()    { return reprojError; }
    public double getAmbiguity()      { return ambiguity; }
    public int    getSolveMode()      { return solveMode; }
    public boolean isConstrainedSolve() { return solveMode == 0; }

    /**
     * FPGA-timebase timestamp for {@code addVisionMeasurement()}, computed from the reported
     * latency: {@code fpgaTimeNowSecs - latency}. Pass {@code Timer.getFPGATimestamp()}. This is
     * the correct way to timestamp the measurement — it needs no Jetson↔robot clock sync (§2A.1).
     */
    public double getTimestampSecs(double fpgaTimeNowSecs) {
        return fpgaTimeNowSecs - getLatencySecs();
    }

    /**
     * Raw Jetson capture time in seconds (CLOCK_MONOTONIC domain). NOT in the robot's FPGA
     * timebase — do not feed this to addVisionMeasurement(); use {@link #getTimestampSecs(double)}.
     */
    public double getCaptureMonotonicSecs() { return timestampNs * 1e-9; }

    /**
     * Heuristic (xyStdDev, thetaStdDev) for addVisionMeasurement, in meters and radians. Grows
     * with tag distance and ambiguity, shrinks with tag count; the gyro-constrained solve trusts
     * heading tightly (its heading comes from the robot's own gyro). Tune the base constants to
     * your camera/field; this is a sane starting point, not a calibrated model.
     */
    public double[] suggestedStdDevs() {
        double xy = 0.02 * (1.0 + avgTagDistance * avgTagDistance) / Math.max(1, tagCount);
        if (solveMode != 0) {
            xy *= (1.0 + 4.0 * ambiguity);   // IPPE fallback: distrust ambiguous solves
        }
        double theta = (solveMode == 0) ? 0.02 : xy;  // constrained heading is gyro-tight
        return new double[] { xy, theta };
    }
}
