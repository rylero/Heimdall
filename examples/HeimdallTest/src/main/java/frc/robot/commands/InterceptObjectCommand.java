package frc.robot.commands;

import com.heimdall.FieldPoint;
import com.heimdall.InterceptSolver;
import com.heimdall.InterceptSolver.Solution;
import com.heimdall.TrackedObject;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.wpilibj2.command.Command;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;
import java.util.Optional;
import org.littletonrobotics.junction.Logger;

/** Drives the robot toward a constant-velocity intercept point for the best tracked object. */
public class InterceptObjectCommand extends Command {
  private static final double MAX_LOOKAHEAD_SECONDS = 3.0;
  private static final double ARRIVAL_TOLERANCE_METERS = 0.15;

  private final Drive drive;
  private final Heimdall heimdall;
  private boolean hit;

  public InterceptObjectCommand(Drive drive, Heimdall heimdall) {
    this.drive = drive;
    this.heimdall = heimdall;
    addRequirements(drive);
  }

  @Override
  public void initialize() {
    hit = false;
  }

  @Override
  public void execute() {
    if (hit) {
      drive.stop();
      return;
    }

    Optional<TrackedObject> target = heimdall.getBestTrackedObject();
    if (target.isEmpty()) {
      drive.stop();
      return;
    }

    Translation2d robotPos = drive.getPose().getTranslation();
    double maxSpeed = drive.getMaxLinearSpeedMetersPerSec();

    Optional<Solution> solution =
        InterceptSolver.solve(
            target.get(), robotPos.getX(), robotPos.getY(), maxSpeed, MAX_LOOKAHEAD_SECONDS);

    if (solution.isEmpty()) {
      // Target is uncatchable at our top speed -- hold position rather than chase forever.
      drive.stop();
      return;
    }

    FieldPoint aim = solution.get().point();
    Translation2d aimPoint = new Translation2d(aim.x(), aim.y());
    Translation2d toAim = aimPoint.minus(robotPos);
    double distance = toAim.getNorm();

    Logger.recordOutput("Intercept/AimPoint", new Pose2d(aimPoint, drive.getRotation()));
    Logger.recordOutput("Intercept/IsExactIntercept", solution.get().isExactIntercept());
    Logger.recordOutput("Intercept/TimeSeconds", solution.get().timeSeconds());

    if (distance < ARRIVAL_TOLERANCE_METERS) {
      hit = true;
      heimdall.markObjectHit();
      drive.stop();
      return;
    }

    Translation2d direction = toAim.div(distance);
    drive.runVelocity(
        ChassisSpeeds.fromFieldRelativeSpeeds(
            direction.getX() * maxSpeed, direction.getY() * maxSpeed, 0.0, drive.getRotation()));
  }

  @Override
  public boolean isFinished() {
    return hit;
  }

  @Override
  public void end(boolean interrupted) {
    drive.stop();
  }
}
