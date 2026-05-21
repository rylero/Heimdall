package com.heimdall;

/** Field-relative tracked object with Kalman-filtered pose and velocity. */
public final class TrackedObject {
    private final int trackId;
    private final int classId;
    private final double x;   // field-relative meters
    private final double y;
    private final double vx;  // velocity m/s
    private final double vy;
    private final double confidence;

    public TrackedObject(int trackId, int classId, double x, double y,
                         double vx, double vy, double confidence) {
        this.trackId = trackId;
        this.classId = classId;
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.confidence = confidence;
    }

    public int getTrackId()       { return trackId; }
    public int getClassId()       { return classId; }
    public double getX()          { return x; }
    public double getY()          { return y; }
    public double getVx()         { return vx; }
    public double getVy()         { return vy; }
    public double getConfidence() { return confidence; }

    @Override
    public String toString() {
        return String.format("TrackedObject{id=%d cls=%d pos=(%.2f,%.2f) vel=(%.2f,%.2f) conf=%.2f}",
                trackId, classId, x, y, vx, vy, confidence);
    }
}
