package com.heimdall;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class InterceptSolverTest {

    private static final double TOL = 1e-3;

    private static TrackedObject moving(double x, double y, double vx, double vy) {
        return new TrackedObject(1, 0, x, y, vx, vy, 0.0, 0.0, 1.0);
    }

    @Test
    void stationaryTargetInterceptsAtStraightLineTime() {
        TrackedObject target = moving(10.0, 0.0, 0.0, 0.0);
        Optional<InterceptSolver.Solution> result = InterceptSolver.solve(target, 0.0, 0.0, 5.0, 10.0);

        assertTrue(result.isPresent());
        InterceptSolver.Solution solution = result.get();
        assertTrue(solution.isExactIntercept());
        assertEquals(2.0, solution.timeSeconds(), TOL);          // 10m at 5m/s
        assertEquals(10.0, solution.point().x(), TOL);
        assertEquals(0.0, solution.point().y(), TOL);
    }

    @Test
    void fasterRobotCatchesSlowerMovingTarget() {
        // Target starts at (10,0) moving away at 1 m/s; robot at origin moving at 5 m/s.
        TrackedObject target = moving(10.0, 0.0, 1.0, 0.0);
        Optional<InterceptSolver.Solution> result = InterceptSolver.solve(target, 0.0, 0.0, 5.0, 10.0);

        assertTrue(result.isPresent());
        InterceptSolver.Solution solution = result.get();
        assertTrue(solution.isExactIntercept());
        assertEquals(2.5, solution.timeSeconds(), TOL);
        assertEquals(12.5, solution.point().x(), TOL);
        assertEquals(0.0, solution.point().y(), TOL);
    }

    @Test
    void targetFasterThanRobotIsUncatchable() {
        // Target starts at (10,0) moving away at 5 m/s; robot can only do 1 m/s.
        TrackedObject target = moving(10.0, 0.0, 5.0, 0.0);
        Optional<InterceptSolver.Solution> result = InterceptSolver.solve(target, 0.0, 0.0, 1.0, 10.0);

        assertTrue(result.isEmpty());
    }

    @Test
    void solutionBeyondHorizonFallsBackToBestEffortChasePoint() {
        // Same scenario as fasterRobotCatchesSlowerMovingTarget (true intercept at t=2.5s),
        // but the horizon cuts off at 1.0s.
        TrackedObject target = moving(10.0, 0.0, 1.0, 0.0);
        Optional<InterceptSolver.Solution> result = InterceptSolver.solve(target, 0.0, 0.0, 5.0, 1.0);

        assertTrue(result.isPresent());
        InterceptSolver.Solution solution = result.get();
        assertFalse(solution.isExactIntercept());
        assertEquals(1.0, solution.timeSeconds(), TOL);
        // Best-effort point is just the predicted position at the horizon, not the true intercept.
        FieldPoint expected = MotionPrediction.predictPosition(target, 1.0);
        assertEquals(expected.x(), solution.point().x(), TOL);
        assertEquals(expected.y(), solution.point().y(), TOL);
    }

    @Test
    void degenerateCaseWhenSpeedsMatchStillSolves() {
        // Target speed (|vx,vy| = 5) equals robot speed -> quadratic's leading term vanishes,
        // forcing the linear-equation fallback path.
        TrackedObject target = moving(10.0, 0.0, -3.0, 4.0);
        Optional<InterceptSolver.Solution> result = InterceptSolver.solve(target, 0.0, 0.0, 5.0, 10.0);

        assertTrue(result.isPresent());
        InterceptSolver.Solution solution = result.get();
        assertTrue(solution.isExactIntercept());
        assertEquals(5.0 / 3.0, solution.timeSeconds(), TOL);
        assertEquals(5.0, solution.point().x(), TOL);
        assertEquals(20.0 / 3.0, solution.point().y(), TOL);
    }
}
