package frc.robot.commands;

import com.heimdall.TrackedObject;
import com.pathplanner.lib.path.PathConstraints;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;
import java.util.Optional;
import java.util.Set;
import org.littletonrobotics.junction.Logger;

/**
 * Test-mode behaviour: waits until a fuel game piece is visible, then pathfinds to the nearest one
 * (facing it) and stops on arrival. Single-shot -- it ends once the robot reaches the fuel, so
 * re-enabling Test mode re-selects and drives to the nearest fuel again.
 */
public final class PathfindToNearestFuelCommand {
  // Acceleration limits for the pathfind (translational velocity + angular velocity are taken from
  // the drivetrain's own maxima). Conservative so the robot arrives cleanly rather than overshoots.
  private static final double MAX_ACCEL_MPS2 = 3.0;
  private static final double MAX_ANGULAR_ACCEL_RAD_PS2 = 4.0 * Math.PI;

  private PathfindToNearestFuelCommand() {}

  /** Builds the test-mode command. Requires {@code drive}. */
  public static Command create(Drive drive, Heimdall heimdall) {
    // Hold position (requiring drive, so the joystick default command can't move us) until a fuel
    // is visible, then pathfind to the nearest one and stop.
    return Commands.run(drive::stop, drive)
        .until(() -> heimdall.getNearestFuel(drive.getPose().getTranslation()).isPresent())
        .andThen(Commands.defer(() -> pathfindToNearestFuel(drive, heimdall), Set.of(drive)))
        .withName("PathfindToNearestFuel");
  }

  private static Command pathfindToNearestFuel(Drive drive, Heimdall heimdall) {
    Optional<TrackedObject> fuel = heimdall.getNearestFuel(drive.getPose().getTranslation());
    if (fuel.isEmpty()) {
      // Nothing visible anymore (target lost between the wait and the defer) -- do nothing.
      return Commands.none();
    }

    Translation2d target = new Translation2d(fuel.get().getX(), fuel.get().getY());
    Rotation2d facing = target.minus(drive.getPose().getTranslation()).getAngle();
    Pose2d goal = new Pose2d(target, facing);

    PathConstraints constraints =
        new PathConstraints(
            drive.getMaxLinearSpeedMetersPerSec(),
            MAX_ACCEL_MPS2,
            drive.getMaxAngularSpeedRadPerSec(),
            MAX_ANGULAR_ACCEL_RAD_PS2);

    Logger.recordOutput("TestMode/NearestFuel", goal);

    // goalEndVelocity 0.0 -> the pathfinder decelerates to a stop at the fuel.
    return drive.pathfindToPoseFaced(goal, constraints, () -> target, 0.0);
  }
}
