package frc.robot.commands;

import com.heimdall.TrackedObject;
import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;
import java.util.Set;
import org.littletonrobotics.junction.Logger;

/**
 * Test-mode behaviour (potential-field avoidance): on press, drive to a fixed field point {@link
 * #GOAL_AHEAD_M} straight ahead of where the robot was at press time, steering around any MOVING
 * tracked objects. Avoidance is a reactive potential field -- goal attraction plus a repulsion from
 * each moving obstacle that grows as the robot nears it. Plain {@link ChassisSpeeds} velocity
 * control, no PathPlanner. Ends on arrival.
 */
public final class PotentialFieldDriveAheadCommand {
  private static final double GOAL_AHEAD_M = 3.0;

  // Goal attraction.
  private static final double LINEAR_KP = 1.0; // (m/s) per meter of remaining distance
  private static final double MAX_LINEAR_MPS = 1.0;
  private static final double THETA_KP = 2.0;
  private static final double MAX_ANGULAR_RAD_PS = 1.5;
  private static final double STOP_TOLERANCE_M = 0.15;

  // Obstacle repulsion. Only objects moving faster than MOVING_SPEED_MPS are avoided, and only
  // within AVOID_RADIUS_M. Repulsion magnitude = AVOID_GAIN * (1/d - 1/R): zero at the radius,
  // blowing up as the robot nears the obstacle.
  private static final double MOVING_SPEED_MPS = 0.3;
  private static final double AVOID_RADIUS_M = 1.5;
  private static final double AVOID_GAIN = 1.5;
  private static final double MAX_AVOID_MPS = 2.0;

  private PotentialFieldDriveAheadCommand() {}

  /** Builds the command. Requires {@code drive}. */
  public static Command create(Drive drive, Heimdall heimdall) {
    return Commands.defer(() -> driveToGoalAhead(drive, heimdall), Set.of(drive))
        .withName("PotentialFieldDriveAhead");
  }

  /** Captures the goal 3 m ahead (at schedule time) and drives there with obstacle avoidance. */
  private static Command driveToGoalAhead(Drive drive, Heimdall heimdall) {
    Pose2d start = drive.getPose();
    Translation2d goal =
        start
            .getTranslation()
            .plus(new Translation2d(GOAL_AHEAD_M, 0.0).rotateBy(start.getRotation()));

    return Commands.run(() -> stepTowardGoal(drive, heimdall, goal), drive)
        .until(() -> drive.getPose().getTranslation().getDistance(goal) <= STOP_TOLERANCE_M)
        .andThen(drive::stop)
        .withName("PotentialFieldDriveAhead/ToGoal");
  }

  private static void stepTowardGoal(Drive drive, Heimdall heimdall, Translation2d goal) {
    Pose2d pose = drive.getPose();
    Translation2d position = pose.getTranslation();

    // Attractive velocity toward the goal, magnitude proportional to remaining distance.
    Translation2d toGoal = goal.minus(position);
    double distance = toGoal.getNorm();
    double attractSpeed = Math.min(LINEAR_KP * distance, MAX_LINEAR_MPS);
    Translation2d velocity =
        distance > 1e-6 ? toGoal.div(distance).times(attractSpeed) : Translation2d.kZero;

    // Repulsive velocity from each moving obstacle within the influence radius.
    Translation2d repulsion = Translation2d.kZero;
    for (TrackedObject obj : heimdall.getTrackedObjects()) {
      if (Math.hypot(obj.getVx(), obj.getVy()) < MOVING_SPEED_MPS) {
        continue; // stationary -- ignore
      }
      Translation2d away = position.minus(new Translation2d(obj.getX(), obj.getY()));
      double d = away.getNorm();
      if (d < 1e-6 || d >= AVOID_RADIUS_M) {
        continue;
      }
      double mag = AVOID_GAIN * (1.0 / d - 1.0 / AVOID_RADIUS_M);
      repulsion = repulsion.plus(away.div(d).times(mag));
    }
    if (repulsion.getNorm() > MAX_AVOID_MPS) {
      repulsion = repulsion.div(repulsion.getNorm()).times(MAX_AVOID_MPS);
    }

    // Combine and re-cap to the linear speed limit.
    Translation2d command = velocity.plus(repulsion);
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
