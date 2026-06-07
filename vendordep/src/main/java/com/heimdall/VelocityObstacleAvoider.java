package com.heimdall;

import java.util.List;

/**
 * Reactive local collision avoidance via velocity obstacles, assuming circular
 * footprints for both the robot and tracked obstacles (their combined radius is
 * supplied by the caller — e.g. robot bounding-circle radius + game-piece radius).
 */
public final class VelocityObstacleAvoider {

    private static final int SPEED_LEVELS = 5;
    private static final int ANGLE_STEPS  = 16;

    private VelocityObstacleAvoider() {}

    /**
     * A velocity command the robot should follow instead of its preferred one.
     *
     * @param vx       chosen x velocity, m/s
     * @param vy       chosen y velocity, m/s
     * @param adjusted false if the preferred velocity was already collision-free
     *                 and is returned unchanged; true if it had to be replaced —
     *                 either with the closest collision-free alternative found, or
     *                 with a full stop (0, 0) if no sampled candidate was safe
     */
    public record AvoidanceResult(double vx, double vy, boolean adjusted) {
    }

    /**
     * True if traveling at {@code (vx, vy)} from {@code robotPos} keeps the robot
     * more than {@code combinedRadius} away from {@code obstaclePos} (moving at
     * {@code (obsVx, obsVy)}) for the next {@code timeHorizonSeconds}.
     *
     * <p>Implemented by checking the closest approach along the relative-velocity
     * ray — equivalent to a velocity-obstacle cone membership test, without
     * constructing explicit cone geometry.
     */
    public static boolean isVelocitySafe(FieldPoint robotPos, double vx, double vy,
                                          FieldPoint obstaclePos, double obsVx, double obsVy,
                                          double combinedRadius, double timeHorizonSeconds) {
        double relVx = vx - obsVx;
        double relVy = vy - obsVy;
        double relPx = obstaclePos.x() - robotPos.x();
        double relPy = obstaclePos.y() - robotPos.y();

        double relSpeedSq = relVx * relVx + relVy * relVy;
        double radiusSq = combinedRadius * combinedRadius;

        if (relSpeedSq < 1e-12) {
            // No relative motion: only unsafe if already overlapping.
            return relPx * relPx + relPy * relPy > radiusSq;
        }

        // Time of closest approach along the relative-velocity ray, clamped to
        // [0, horizon] — approaches in the past (diverging) or beyond the trusted
        // prediction horizon don't count as collisions.
        double t = (relPx * relVx + relPy * relVy) / relSpeedSq;
        t = Math.max(0.0, Math.min(timeHorizonSeconds, t));

        double closestPx = relPx - relVx * t;
        double closestPy = relPy - relVy * t;
        return closestPx * closestPx + closestPy * closestPy > radiusSq;
    }

    /**
     * Returns {@code (preferredVx, preferredVy)} unchanged if it's collision-free
     * against every obstacle; otherwise samples candidate velocities (a ring of
     * speed levels x angles, up to {@code maxSpeed}) and returns the closest
     * collision-free one. If none of the samples are safe, returns a full stop.
     */
    public static AvoidanceResult computeSafeVelocity(FieldPoint robotPos,
            double preferredVx, double preferredVy, double maxSpeed,
            List<TrackedObject> obstacles, double combinedRadius, double timeHorizonSeconds) {

        if (isSafeAgainstAll(robotPos, preferredVx, preferredVy,
                obstacles, combinedRadius, timeHorizonSeconds)) {
            return new AvoidanceResult(preferredVx, preferredVy, false);
        }

        double bestVx = 0.0;
        double bestVy = 0.0;
        double bestCost = Double.POSITIVE_INFINITY;

        for (int s = 0; s < SPEED_LEVELS; s++) {
            double speed = maxSpeed * (s + 1) / SPEED_LEVELS;
            for (int a = 0; a < ANGLE_STEPS; a++) {
                double angle = 2.0 * Math.PI * a / ANGLE_STEPS;
                double vx = speed * Math.cos(angle);
                double vy = speed * Math.sin(angle);

                if (!isSafeAgainstAll(robotPos, vx, vy, obstacles, combinedRadius, timeHorizonSeconds)) {
                    continue;
                }

                double dvx = vx - preferredVx;
                double dvy = vy - preferredVy;
                double cost = dvx * dvx + dvy * dvy;
                if (cost < bestCost) {
                    bestCost = cost;
                    bestVx = vx;
                    bestVy = vy;
                }
            }
        }

        // If no sample was safe, bestVx/bestVy remain (0, 0) — a full stop, the safe default.
        return new AvoidanceResult(bestVx, bestVy, true);
    }

    private static boolean isSafeAgainstAll(FieldPoint robotPos, double vx, double vy,
            List<TrackedObject> obstacles, double combinedRadius, double timeHorizonSeconds) {
        for (TrackedObject obstacle : obstacles) {
            FieldPoint obstaclePos = new FieldPoint(obstacle.getX(), obstacle.getY());
            if (!isVelocitySafe(robotPos, vx, vy, obstaclePos,
                    obstacle.getVx(), obstacle.getVy(), combinedRadius, timeHorizonSeconds)) {
                return false;
            }
        }
        return true;
    }
}
