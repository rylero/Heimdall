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
     *                 (if no candidate was collision-free) with the least-bad
     *                 candidate that maximizes clearance
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
        double clearanceSq = closestApproachSq(robotPos, vx, vy, obstaclePos,
                obsVx, obsVy, timeHorizonSeconds);
        return clearanceSq > combinedRadius * combinedRadius;
    }

    /**
     * Squared closest-approach distance between the robot (traveling {@code (vx,vy)} from
     * {@code robotPos}) and the obstacle over {@code [0, timeHorizonSeconds]}. Smaller =
     * nearer miss; used both for the safety test and to rank "least-bad" fallbacks.
     */
    public static double closestApproachSq(FieldPoint robotPos, double vx, double vy,
                                           FieldPoint obstaclePos, double obsVx, double obsVy,
                                           double timeHorizonSeconds) {
        double relVx = vx - obsVx;
        double relVy = vy - obsVy;
        double relPx = obstaclePos.x() - robotPos.x();
        double relPy = obstaclePos.y() - robotPos.y();

        double relSpeedSq = relVx * relVx + relVy * relVy;

        if (relSpeedSq < 1e-12) {
            // No relative motion: distance is just the current separation.
            return relPx * relPx + relPy * relPy;
        }

        // Time of closest approach along the relative-velocity ray, clamped to
        // [0, horizon] — approaches in the past (diverging) or beyond the trusted
        // prediction horizon don't count.
        double t = (relPx * relVx + relPy * relVy) / relSpeedSq;
        t = Math.max(0.0, Math.min(timeHorizonSeconds, t));

        double closestPx = relPx - relVx * t;
        double closestPy = relPy - relVy * t;
        return closestPx * closestPx + closestPy * closestPy;
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

        final double radiusSq = combinedRadius * combinedRadius;

        if (minClearanceSq(robotPos, preferredVx, preferredVy, obstacles, timeHorizonSeconds) > radiusSq) {
            return new AvoidanceResult(preferredVx, preferredVy, false);
        }

        // Among all sampled candidates track two things: the collision-free one closest to
        // the preferred velocity, and — if NONE is safe — the "least-bad" one that maximizes
        // clearance. A full stop (0,0) is NOT assumed safe (5.18): a moving obstacle can hit a
        // stationary robot, so it is evaluated as just another candidate. Falling back to the
        // max-clearance velocity lets the robot dodge instead of sitting still and getting hit.
        double bestVx = 0.0, bestVy = 0.0, bestCost = Double.POSITIVE_INFINITY;
        boolean anySafe = false;

        double fbVx = 0.0, fbVy = 0.0, fbClearance = Double.NEGATIVE_INFINITY;

        for (int s = 0; s <= SPEED_LEVELS; s++) {
            // s == 0 → speed 0 (the full-stop candidate, evaluated once at angle 0).
            double speed = maxSpeed * s / SPEED_LEVELS;
            int angleSteps = (s == 0) ? 1 : ANGLE_STEPS;
            for (int a = 0; a < angleSteps; a++) {
                double angle = 2.0 * Math.PI * a / ANGLE_STEPS;
                double vx = speed * Math.cos(angle);
                double vy = speed * Math.sin(angle);

                double clearance = minClearanceSq(robotPos, vx, vy, obstacles, timeHorizonSeconds);
                if (clearance > radiusSq) {
                    double dvx = vx - preferredVx;
                    double dvy = vy - preferredVy;
                    double cost = dvx * dvx + dvy * dvy;
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestVx = vx;
                        bestVy = vy;
                        anySafe = true;
                    }
                } else if (clearance > fbClearance) {
                    fbClearance = clearance;
                    fbVx = vx;
                    fbVy = vy;
                }
            }
        }

        return anySafe
                ? new AvoidanceResult(bestVx, bestVy, true)
                : new AvoidanceResult(fbVx, fbVy, true);  // least-bad, not a blind stop
    }

    /** Minimum closest-approach (squared) over all obstacles; +inf when there are none. */
    private static double minClearanceSq(FieldPoint robotPos, double vx, double vy,
            List<TrackedObject> obstacles, double timeHorizonSeconds) {
        double min = Double.POSITIVE_INFINITY;
        for (TrackedObject obstacle : obstacles) {
            FieldPoint obstaclePos = new FieldPoint(obstacle.getX(), obstacle.getY());
            double d = closestApproachSq(robotPos, vx, vy, obstaclePos,
                    obstacle.getVx(), obstacle.getVy(), timeHorizonSeconds);
            if (d < min) min = d;
        }
        return min;
    }
}
