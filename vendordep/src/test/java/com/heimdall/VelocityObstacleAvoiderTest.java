package com.heimdall;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class VelocityObstacleAvoiderTest {

    private static final double TOL = 1e-6;
    private static final FieldPoint ORIGIN = new FieldPoint(0.0, 0.0);

    private static TrackedObject obstacle(double x, double y, double vx, double vy) {
        return new TrackedObject(1, 0, x, y, vx, vy, 0.0, 0.0, 1.0);
    }

    // ── isVelocitySafe ────────────────────────────────────────────────────────

    @Test
    void headOnCourseIntoStationaryObstacleIsUnsafe() {
        FieldPoint obstaclePos = new FieldPoint(10.0, 0.0);
        boolean safe = VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, 1.0, 0.0, obstaclePos, 0.0, 0.0, 1.0, 100.0);
        assertFalse(safe);
    }

    @Test
    void courseThatClearsObstacleByMoreThanCombinedRadiusIsSafe() {
        // Straight-line travel along +x; obstacle offset 4m laterally, radius 3m -> clears by 1m.
        FieldPoint obstaclePos = new FieldPoint(10.0, 4.0);
        boolean safe = VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, 1.0, 0.0, obstaclePos, 0.0, 0.0, 3.0, 100.0);
        assertTrue(safe);
    }

    @Test
    void courseThatPassesInsideCombinedRadiusIsUnsafe() {
        // Same setup, but obstacle is only 2m laterally offset -> inside the 3m combined radius.
        FieldPoint obstaclePos = new FieldPoint(10.0, 2.0);
        boolean safe = VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, 1.0, 0.0, obstaclePos, 0.0, 0.0, 3.0, 100.0);
        assertFalse(safe);
    }

    @Test
    void obstacleMovingAwayFasterThanClosingSpeedIsSafe() {
        // Robot moving at 1 m/s toward an obstacle that's retreating at 5 m/s -- relative
        // velocity points backward, so closest approach was in the past (clamped to t=0).
        FieldPoint obstaclePos = new FieldPoint(10.0, 0.0);
        boolean safe = VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, 1.0, 0.0, obstaclePos, 5.0, 0.0, 1.0, 100.0);
        assertTrue(safe);
    }

    @Test
    void zeroRelativeVelocityIsUnsafeOnlyWhenAlreadyOverlapping() {
        double matchedVx = 2.0;
        double matchedVy = 2.0;

        FieldPoint overlapping = new FieldPoint(0.5, 0.0);
        assertFalse(VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, matchedVx, matchedVy, overlapping, matchedVx, matchedVy, 1.0, 10.0));

        FieldPoint clear = new FieldPoint(2.0, 0.0);
        assertTrue(VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, matchedVx, matchedVy, clear, matchedVx, matchedVy, 1.0, 10.0));
    }

    @Test
    void collisionBeyondTimeHorizonIsTreatedAsSafe() {
        // Closest approach to this stationary obstacle occurs at t=100s; with only a
        // 5s trusted horizon the clamped closest point is still 95m away -- safe for now.
        FieldPoint obstaclePos = new FieldPoint(100.0, 0.0);
        boolean safe = VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, 1.0, 0.0, obstaclePos, 0.0, 0.0, 1.0, 5.0);
        assertTrue(safe);
    }

    // ── computeSafeVelocity ───────────────────────────────────────────────────

    @Test
    void noObstaclesReturnsPreferredVelocityUnchanged() {
        VelocityObstacleAvoider.AvoidanceResult result = VelocityObstacleAvoider.computeSafeVelocity(
                ORIGIN, 5.0, 0.0, 5.0, List.of(), 1.0, 5.0);

        assertFalse(result.adjusted());
        assertEquals(5.0, result.vx(), TOL);
        assertEquals(0.0, result.vy(), TOL);
    }

    @Test
    void blockedPreferredDirectionYieldsCollisionFreeAlternative() {
        TrackedObject blocker = obstacle(3.0, 0.0, 0.0, 0.0);
        VelocityObstacleAvoider.AvoidanceResult result = VelocityObstacleAvoider.computeSafeVelocity(
                ORIGIN, 5.0, 0.0, 5.0, List.of(blocker), 1.0, 10.0);

        assertTrue(result.adjusted());
        // Whatever was chosen, it must actually be collision-free against the blocker.
        FieldPoint blockerPos = new FieldPoint(blocker.getX(), blocker.getY());
        assertTrue(VelocityObstacleAvoider.isVelocitySafe(
                ORIGIN, result.vx(), result.vy(), blockerPos,
                blocker.getVx(), blocker.getVy(), 1.0, 10.0));
        // And it must respect the speed cap (samples are bounded by maxSpeed).
        assertTrue(Math.hypot(result.vx(), result.vy()) <= 5.0 + TOL);
    }

    @Test
    void obstacleCollocatedWithRobotLeavesNoSafeSampleAndStops() {
        // An obstacle exactly at the robot's position is within combinedRadius of every
        // candidate velocity's near-term path (closest-approach point stays at the origin),
        // so every sample is unsafe and the avoider falls back to a full stop.
        TrackedObject collocated = obstacle(0.0, 0.0, 0.0, 0.0);
        VelocityObstacleAvoider.AvoidanceResult result = VelocityObstacleAvoider.computeSafeVelocity(
                ORIGIN, 5.0, 0.0, 5.0, List.of(collocated), 1.0, 5.0);

        assertTrue(result.adjusted());
        assertEquals(0.0, result.vx(), TOL);
        assertEquals(0.0, result.vy(), TOL);
    }
}
