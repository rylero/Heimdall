package com.heimdall;

import java.util.Optional;
import java.util.OptionalDouble;

/** Solves for when/where a robot can intercept a constant-velocity tracked object. */
public final class InterceptSolver {

    private static final double EPS = 1e-9;

    private InterceptSolver() {}

    /**
     * One intercept candidate: where the robot should aim and when it'll get there.
     *
     * @param timeSeconds      time until the robot reaches {@code point}
     * @param point            field-relative point to drive toward
     * @param isExactIntercept true if {@code point} is a genuine intercept (robot and
     *                         target arrive simultaneously); false if the true intercept
     *                         lies beyond the trusted horizon and {@code point} is the
     *                         furthest predicted position still within it — chase it,
     *                         but don't commit (re-solve next cycle; it may upgrade to
     *                         a true intercept as the robot closes the gap)
     */
    public record Solution(double timeSeconds, FieldPoint point, boolean isExactIntercept) {
    }

    /**
     * Solves for the earliest point at which a robot at {@code (robotX, robotY)} moving
     * at {@code robotSpeed} (m/s) can catch {@code target}, assuming the target continues
     * at constant velocity.
     *
     * <p>Returns {@link Optional#empty()} only when the target is permanently uncatchable
     * at the given robot speed (it's moving away faster than the robot can close). When a
     * true intercept exists but lies beyond {@code maxLookaheadSeconds} — past which the
     * constant-velocity assumption is no longer trustworthy — returns a best-effort
     * {@code Solution} aimed at the predicted position at the horizon instead, flagged
     * with {@code isExactIntercept = false}.
     */
    public static Optional<Solution> solve(TrackedObject target, double robotX, double robotY,
                                            double robotSpeed, double maxLookaheadSeconds) {
        double dx = target.getX() - robotX;
        double dy = target.getY() - robotY;
        double vx = target.getVx();
        double vy = target.getVy();

        // |target(t) - robot| = robotSpeed * t, squared, gives a*t^2 + b*t + c = 0
        double a = vx * vx + vy * vy - robotSpeed * robotSpeed;
        double b = 2.0 * (dx * vx + dy * vy);
        double c = dx * dx + dy * dy;

        OptionalDouble root = smallestPositiveRoot(a, b, c);
        if (root.isEmpty()) {
            return Optional.empty(); // target outruns the robot forever — uncatchable
        }

        double trueTime = root.getAsDouble();
        if (trueTime <= maxLookaheadSeconds) {
            return Optional.of(new Solution(trueTime,
                    MotionPrediction.predictPosition(target, trueTime), true));
        }

        // Solvable, but past the trust horizon — chase the furthest point we still trust.
        return Optional.of(new Solution(maxLookaheadSeconds,
                MotionPrediction.predictPosition(target, maxLookaheadSeconds), false));
    }

    /** Smallest positive real root of {@code a*t^2 + b*t + c = 0}, or empty if none exists. */
    private static OptionalDouble smallestPositiveRoot(double a, double b, double c) {
        if (Math.abs(a) < EPS) {
            // Degenerates to linear: b*t + c = 0
            if (Math.abs(b) < EPS) {
                return OptionalDouble.empty();
            }
            double t = -c / b;
            return t > EPS ? OptionalDouble.of(t) : OptionalDouble.empty();
        }

        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return OptionalDouble.empty();
        }

        double sqrtDisc = Math.sqrt(discriminant);
        double lo = (-b - sqrtDisc) / (2 * a);
        double hi = (-b + sqrtDisc) / (2 * a);
        if (lo > hi) {
            double tmp = lo;
            lo = hi;
            hi = tmp;
        }

        if (lo > EPS) {
            return OptionalDouble.of(lo);
        }
        if (hi > EPS) {
            return OptionalDouble.of(hi);
        }
        return OptionalDouble.empty();
    }
}
