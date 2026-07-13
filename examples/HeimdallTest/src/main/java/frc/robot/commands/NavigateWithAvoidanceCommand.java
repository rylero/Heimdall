package frc.robot.commands;

import com.heimdall.FieldPoint;
import com.heimdall.TrackedObject;
import com.heimdall.VelocityObstacleAvoider;
import com.heimdall.VelocityObstacleAvoider.AvoidanceResult;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.wpilibj2.command.Command;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;
import java.util.List;
import org.littletonrobotics.junction.Logger;

/**
 * Drives the robot toward a fixed goal point, reactively rerouting around tracked obstacles via
 * velocity obstacles ({@link VelocityObstacleAvoider}).
 */
public class NavigateWithAvoidanceCommand extends Command {
  private static final double ARRIVAL_TOLERANCE_METERS = 0.15;
  private static final double COMBINED_RADIUS_METERS = Drive.DRIVE_BASE_RADIUS + 0.75;
  private static final double TIME_HORIZON_SECONDS = 2.0;

  private enum Outcome {
    IN_PROGRESS,
    REACHED_GOAL,
    COLLIDED,
    TIMEOUT
  }

  private final Drive drive;
  private final Heimdall heimdall;
  private final Translation2d goal;
  private Outcome outcome;

  public NavigateWithAvoidanceCommand(Drive drive, Heimdall heimdall, Translation2d goal) {
    this.drive = drive;
    this.heimdall = heimdall;
    this.goal = goal;
    addRequirements(drive);
  }

  @Override
  public void initialize() {
    outcome = Outcome.IN_PROGRESS;
  }

  @Override
  public void execute() {
    if (outcome != Outcome.IN_PROGRESS) {
      drive.stop();
      return;
    }

    Translation2d robotPos = drive.getPose().getTranslation();
    double maxSpeed = drive.getMaxLinearSpeedMetersPerSec();
    List<TrackedObject> obstacles = heimdall.getTrackedObjects();

    // Collision is checked before arrival -- reaching the goal while overlapping an
    // obstacle counts as a collision, not a clean win.
    for (TrackedObject obstacle : obstacles) {
      double dx = obstacle.getX() - robotPos.getX();
      double dy = obstacle.getY() - robotPos.getY();
      if (Math.hypot(dx, dy) < COMBINED_RADIUS_METERS) {
        outcome = Outcome.COLLIDED;
        heimdall.markObjectHit();
        drive.stop();
        Logger.recordOutput("Navigate/Outcome", outcome.toString());
        return;
      }
    }

    Translation2d toGoal = goal.minus(robotPos);
    double distanceToGoal = toGoal.getNorm();
    if (distanceToGoal < ARRIVAL_TOLERANCE_METERS) {
      outcome = Outcome.REACHED_GOAL;
      drive.stop();
      Logger.recordOutput("Navigate/Outcome", outcome.toString());
      return;
    }

    Translation2d preferredDirection = toGoal.div(distanceToGoal);
    double preferredVx = preferredDirection.getX() * maxSpeed;
    double preferredVy = preferredDirection.getY() * maxSpeed;

    AvoidanceResult safe =
        VelocityObstacleAvoider.computeSafeVelocity(
            new FieldPoint(robotPos.getX(), robotPos.getY()),
            preferredVx,
            preferredVy,
            maxSpeed,
            obstacles,
            COMBINED_RADIUS_METERS,
            TIME_HORIZON_SECONDS);

    Logger.recordOutput("Navigate/AimPoint", new Pose2d(goal, drive.getRotation()));
    Logger.recordOutput("Navigate/PreferredVelocity", new double[] {preferredVx, preferredVy});
    Logger.recordOutput("Navigate/SafeVelocity", new double[] {safe.vx(), safe.vy()});
    Logger.recordOutput("Navigate/Adjusted", safe.adjusted());
    Logger.recordOutput("Navigate/Outcome", outcome.toString());

    drive.runVelocity(
        ChassisSpeeds.fromFieldRelativeSpeeds(safe.vx(), safe.vy(), 0.0, drive.getRotation()));
  }

  @Override
  public boolean isFinished() {
    return outcome != Outcome.IN_PROGRESS;
  }

  @Override
  public void end(boolean interrupted) {
    if (outcome == Outcome.IN_PROGRESS) {
      outcome = Outcome.TIMEOUT;
      Logger.recordOutput("Navigate/Outcome", outcome.toString());
    }
    drive.stop();
  }
}
