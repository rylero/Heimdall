package com.heimdall;

/** Constant-velocity motion extrapolation for tracked objects. */
public final class MotionPrediction {

    private MotionPrediction() {}

    /**
     * Predicted field-relative position of {@code obj} after {@code dtSeconds},
     * assuming it continues at its current velocity.
     */
    public static FieldPoint predictPosition(TrackedObject obj, double dtSeconds) {
        return new FieldPoint(
                obj.getX() + obj.getVx() * dtSeconds,
                obj.getY() + obj.getVy() * dtSeconds);
    }
}
