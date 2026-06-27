package com.heimdall;

/** AprilTag-derived robot pose estimate received from the Jetson. */
public final class VisionPoseEstimate {
    private final double x;
    private final double y;
    private final double headingRad;
    private final long   timestampNs;

    public VisionPoseEstimate(double x, double y, double headingRad, long timestampNs) {
        this.x           = x;
        this.y           = y;
        this.headingRad  = headingRad;
        this.timestampNs = timestampNs;
    }

    public double getX()           { return x; }
    public double getY()           { return y; }
    public double getHeadingRad()  { return headingRad; }
    public long   getTimestampNs() { return timestampNs; }

    /** Timestamp in seconds — convenience for WPILib addVisionMeasurement(). */
    public double getTimestampSecs() { return timestampNs * 1e-9; }
}
