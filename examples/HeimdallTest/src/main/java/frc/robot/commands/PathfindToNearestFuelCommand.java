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
import java.util.Optional;
import org.littletonrobotics.junction.Logger;

/**
 * Test-mode behaviour: drives straight at the nearest visible fuel game piece with a simple
 * proportional velocity controller (plain {@link ChassisSpeeds}, no PathPlanner pathfinding) and
 * stops on arrival. Holds position while no fuel is visible. Single-shot -- it ends once the robot
 * reaches the fuel, so re-enabling Test mode re-selects and drives to the nearest fuel again.
 */
public final class PathfindToNearestFuelCommand {
  // Proportional gains: commanded speed = gain * error, capped at the limits below.
  private static final double LINEAR_KP = 1.0; // (m/s) per meter of position error
  private static final double THETA_KP = 2.0; // (rad/s) per radian of heading error
  // Hard speed caps -- deliberately slow, well under the drivetrain maxima.
  private static final double MAX_LINEAR_MPS = 0.6;
  private static final double MAX_ANGULAR_RAD_PS = 1.5;
  // Considered "arrived" (and the command ends) inside this radius of the fuel.
  private static final double STOP_TOLERANCE_M = 0.15;

  private PathfindToNearestFuelCommand() {}

  /** Builds the test-mode command. Requires {@code drive}. */
  public static Command create(Drive drive, Heimdall heimdall) {
    return Commands.run(() -> driveTowardNearestFuel(drive, heimdall), drive)
        .until(() -> arrived(drive, heimdall))
        .andThen(drive::stop)
        .withName("PathfindToNearestFuel");
  }

  private static Translation2d fuelTranslation(TrackedObject fuel) {
    return new Translation2d(fuel.getX(), fuel.getY());
  }

  /** True once a fuel is visible and the robot is within {@link #STOP_TOLERANCE_M} of it. */
  private static boolean arrived(Drive drive, Heimdall heimdall) {
    Translation2d origin = drive.getPose().getTranslation();
    return heimdall
        .getNearestFuel(origin)
        .map((fuel) -> fuelTranslation(fuel).getDistance(origin) <= STOP_TOLERANCE_M)
        .orElse(false);
  }

  /** One control cycle: proportional field-relative velocity straight at the nearest fuel. */
  private static void driveTowardNearestFuel(Drive drive, Heimdall heimdall) {
    Pose2d pose = drive.getPose();
    Optional<TrackedObject> fuel = heimdall.getNearestFuel(pose.getTranslation());
    if (fuel.isEmpty()) {
      // Nothing to chase yet -- hold position (still requiring drive so the joystick default
      // command can't move us).
      drive.stop();
      return;
    }

    Translation2d target = fuelTranslation(fuel.get());
    Translation2d error = target.minus(pose.getTranslation());
    double distance = error.getNorm();

    Logger.recordOutput("TestMode/NearestFuel", new Pose2d(target, error.getAngle()));

    // Field-relative translational velocity: point the speed vector at the fuel, magnitude
    // proportional to distance and capped at the drivetrain max.
    double speed = Math.min(LINEAR_KP * distance, MAX_LINEAR_MPS);
    Translation2d fieldVelocity =
        distance > 1e-6 ? error.div(distance).times(speed) : Translation2d.kZero;

    // Turn to face the fuel as we approach.
    double headingError = error.getAngle().minus(pose.getRotation()).getRadians();
    double omega = MathUtil.clamp(THETA_KP * headingError, -MAX_ANGULAR_RAD_PS, MAX_ANGULAR_RAD_PS);

    drive.runVelocity(
        ChassisSpeeds.fromFieldRelativeSpeeds(
            fieldVelocity.getX(), fieldVelocity.getY(), omega, pose.getRotation()));
  }
}
