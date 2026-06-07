package com.heimdall;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class MotionPredictionTest {

    private static TrackedObject moving(double x, double y, double vx, double vy) {
        return new TrackedObject(1, 0, x, y, vx, vy, 0.0, 0.0, 1.0);
    }

    @Test
    void extrapolatesAlongConstantVelocity() {
        FieldPoint p = MotionPrediction.predictPosition(moving(1.0, 2.0, 3.0, -1.0), 2.0);
        assertEquals(7.0, p.x(), 1e-9);
        assertEquals(0.0, p.y(), 1e-9);
    }

    @Test
    void stationaryObjectDoesNotMove() {
        FieldPoint p = MotionPrediction.predictPosition(moving(5.0, 5.0, 0.0, 0.0), 10.0);
        assertEquals(5.0, p.x(), 1e-9);
        assertEquals(5.0, p.y(), 1e-9);
    }

    @Test
    void zeroDtReturnsCurrentPosition() {
        FieldPoint p = MotionPrediction.predictPosition(moving(1.0, 2.0, 3.0, 4.0), 0.0);
        assertEquals(1.0, p.x(), 1e-9);
        assertEquals(2.0, p.y(), 1e-9);
    }
}
