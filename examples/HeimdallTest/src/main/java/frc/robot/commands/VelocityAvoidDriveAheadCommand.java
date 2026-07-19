package frc.robot.commands;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;
import java.util.Set;
import org.littletonrobotics.junction.Logger;

/**
 * Test-mode behaviour (predictive velocity avoidance): on press, drive to a fixed field point
 * {@link #GOAL_AHEAD_M} straight ahead of where the robot was at press time, steering around any
 * MOVING tracked objects.
 *
 * <p>Unlike the reactive potential field, this reuses the object-velocity math from {@link
 * PredictiveInterceptCommand}: for each moving obstacle it computes the closest-approach point
 * using the relative velocity (obstacle velocity minus the robot's intended velocity), and if that
 * predicted miss falls inside the safety radius it adds a sideways push away from the miss point --
 * getting out of the way before the obstacle arrives rather than reacting to current proximity.
 * Plain {@link ChassisSpeeds} velocity control, no PathPlanner. Ends on arrival.
 */
public final class VelocityAvoidDriveAheadCommand {
  private static final double GOAL_AHEAD_M = 3.0;

  // Goal attraction (matches the other velocity test commands).
  private static final double LINEAR_KP = 1.0; // (m/s) per meter of remaining distance
  private static final double MAX_LINEAR_MPS = 1.0;
  private static final double THETA_KP = 2.0;
  private static final double MAX_ANGULAR_RAD_PS = 1.5;
  private static final double STOP_TOLERANCE_M = 0.15;

  // Predictive avoidance. Which objects count as obstacles (moving or stationary) -- including ones
  // that just left view -- is handled by ObstacleMemory.
  private static final double SAFE_RADIUS_M = 0.9; // desired clearance at closest approach
  private static final double AVOID_HORIZON_S =
      2.5; // ignore closest approaches farther out than this
  private static final double AVOID_GAIN = 1.5; // (m/s) per meter of predicted intrusion
  private static final double MAX_AVOID_MPS = 2.0;
  // Below this predicted miss distance the approach is treated as head-on, and avoidance steers
  // perpendicular to the approach rather than uselessly pushing straight back.
  private static final double HEADON_EPS_M = 0.25;

  private VelocityAvoidDriveAheadCommand() {}

  /** Builds the command. Requires {@code drive}. */
  public static Command create(Drive drive, Heimdall heimdall) {
    return Commands.defer(() -> driveToGoalAhead(drive, heimdall), Set.of(drive))
        .withName("VelocityAvoidDriveAhead");
  }

  /** Captures the goal 3 m ahead (at schedule time) and drives there with predictive avoidance. */
  private static Command driveToGoalAhead(Drive drive, Heimdall heimdall) {
    Pose2d start = drive.getPose();
    Translation2d goal =
        start
            .getTranslation()
            .plus(new Translation2d(GOAL_AHEAD_M, 0.0).rotateBy(start.getRotation()));

    // Fresh per run so obstacle memory doesn't leak across presses.
    ObstacleMemory memory = new ObstacleMemory();

    return Commands.run(() -> stepTowardGoal(drive, heimdall, goal, memory), drive)
        .until(() -> drive.getPose().getTranslation().getDistance(goal) <= STOP_TOLERANCE_M)
        .andThen(drive::stop)
        .withName("VelocityAvoidDriveAhead/ToGoal");
  }

  private static void stepTowardGoal(
      Drive drive, Heimdall heimdall, Translation2d goal, ObstacleMemory memory) {
    Pose2d pose = drive.getPose();
    Translation2d position = pose.getTranslation();

    // Intended (attractive) velocity toward the goal -- same P-controller as the chase command.
    Translation2d toGoal = goal.minus(position);
    double distance = toGoal.getNorm();
    double attractSpeed = Math.min(LINEAR_KP * distance, MAX_LINEAR_MPS);
    Translation2d toGoalUnit = distance > 1e-6 ? toGoal.div(distance) : new Translation2d(1.0, 0.0);
    Translation2d intended = toGoalUnit.times(attractSpeed);

    // Predictive avoidance: for each moving obstacle (memory dead-reckons ones that just left the
    // FOV), propagate its position relative to the robot (moving at the intended velocity) to the
    // point of closest approach and push away from a predicted intrusion.
    double now = Timer.getFPGATimestamp();
    memory.update(heimdall.getTrackedObjects(), now);
    Translation2d avoidance = Translation2d.kZero;
    for (ObstacleMemory.Obstacle obs : memory.obstacles(now)) {
      Translation2d obstacleVel = obs.velocity();
      Translation2d relPos = obs.position().minus(position);
      Translation2d relVel = obstacleVel.minus(intended); // obstacle motion as seen by the robot
      double relSpeedSq = relVel.getX() * relVel.getX() + relVel.getY() * relVel.getY();
      if (relSpeedSq < 1e-6) {
        continue; // no relative motion
      }

      // Time of closest approach: minimize |relPos + relVel * t|.
      double tca = -(relPos.getX() * relVel.getX() + relPos.getY() * relVel.getY()) / relSpeedSq;
      if (tca <= 0.0 || tca > AVOID_HORIZON_S) {
        continue; // already receding, or too far in the future to matter
      }

      Translation2d miss = relPos.plus(relVel.times(tca)); // robot -> obstacle at closest approach
      double missDist = miss.getNorm();
      if (missDist >= SAFE_RADIUS_M) {
        continue; // clears with margin
      }

      // Push away from the predicted miss point, scaled by how far it intrudes and how soon.
      double intrusion = SAFE_RADIUS_M - missDist;
      double urgency = (AVOID_HORIZON_S - tca) / AVOID_HORIZON_S; // 1 = imminent, 0 = at horizon
      Translation2d awayDir;
      if (missDist > HEADON_EPS_M) {
        awayDir = miss.div(missDist).unaryMinus(); // clear lateral miss -- push off it
      } else {
        // Near head-on: the miss point is ~on the robot, so pushing "away" is backward and useless.
        // Steer perpendicular to the approach, toward the side the goal is on (default one side).
        double rp = relPos.getNorm();
        Translation2d obstacleDir = rp > 1e-6 ? relPos.div(rp) : toGoalUnit;
        Translation2d perp = new Translation2d(-obstacleDir.getY(), obstacleDir.getX());
        double side =
            obstacleDir.getX() * toGoalUnit.getY() - obstacleDir.getY() * toGoalUnit.getX();
        awayDir = side < 0.0 ? perp.unaryMinus() : perp;
      }
      avoidance = avoidance.plus(awayDir.times(AVOID_GAIN * intrusion * urgency));
    }
    if (avoidance.getNorm() > MAX_AVOID_MPS) {
      avoidance = avoidance.div(avoidance.getNorm()).times(MAX_AVOID_MPS);
    }

    // Combine and re-cap to the linear speed limit.
    Translation2d command = intended.plus(avoidance);
    if (command.getNorm() > MAX_LINEAR_MPS) {
      command = command.div(command.getNorm()).times(MAX_LINEAR_MPS);
    }

    Logger.recordOutput("TestMode/AvoidGoal", new Pose2d(goal, pose.getRotation()));

    // Face the direction of travel (hold heading when nearly stopped).
    double omega = 0.0;
    if (command.getNorm() > 1e-3) {
      double headingError = command.getAngle().minus(pose.getRotation()).getRadians();
      omega = MathUtil.clamp(THETA_KP * headingError, -MAX_ANGULAR_RAD_PS, MAX_ANGULAR_RAD_PS);
    }

    drive.runVelocity(
        ChassisSpeeds.fromFieldRelativeSpeeds(
            command.getX(), command.getY(), omega, pose.getRotation()));
  }
}
