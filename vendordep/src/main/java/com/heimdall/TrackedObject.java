package com.heimdall;

/** Field-relative tracked object with Kalman-filtered pose, velocity, and acceleration. */
public final class TrackedObject {
    private final int trackId;
    private final int classId;
    private final double x;
    private final double y;
    private final double vx;
    private final double vy;
    private final double ax;   // 0.0 when filter model is not CONSTANT_ACCELERATION
    private final double ay;
    private final double confidence;

    public TrackedObject(int trackId, int classId, double x, double y,
                         double vx, double vy, double ax, double ay, double confidence) {
        this.trackId    = trackId;
        this.classId    = classId;
        this.x          = x;
        this.y          = y;
        this.vx         = vx;
        this.vy         = vy;
        this.ax         = ax;
        this.ay         = ay;
        this.confidence = confidence;
    }

    public int    getTrackId()    { return trackId; }
    public int    getClassId()    { return classId; }
    public double getX()          { return x; }
    public double getY()          { return y; }
    public double getVx()         { return vx; }
    public double getVy()         { return vy; }
    public double getAx()         { return ax; }
    public double getAy()         { return ay; }
    public double getConfidence() { return confidence; }

    @Override
    public String toString() {
        return String.format(
            "TrackedObject{id=%d cls=%d pos=(%.2f,%.2f) vel=(%.2f,%.2f) acc=(%.2f,%.2f) conf=%.2f}",
            trackId, classId, x, y, vx, vy, ax, ay, confidence);
    }
}
