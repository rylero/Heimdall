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
 * Test-mode behaviour: intercepts a MOVING fuel game piece. Instead of chasing the fuel's current
 * position (pure pursuit, which lags behind a moving target), it solves for the lead point where
 * the robot and fuel arrive at the same time and drives there with a plain {@link ChassisSpeeds}
 * velocity. Falls back to pure pursuit when the target is stationary or uncatchable. Holds position
 * while no fuel is visible; ends once within {@link #STOP_TOLERANCE_M} of the actual fuel.
 */
public final class PredictiveInterceptCommand {
  // Assumed cruise speed used both to solve the intercept and to cap the commanded velocity.
  private static final double INTERCEPT_SPEED_MPS = 1.5;
  private static final double LINEAR_KP = 1.5; // (m/s) per meter, decel as we close on the fuel
  private static final double THETA_KP = 2.0; // (rad/s) per radian of heading error
  private static final double MAX_ANGULAR_RAD_PS = 2.0;
  private static final double STOP_TOLERANCE_M = 0.15;

  private PredictiveInterceptCommand() {}

  /** Builds the intercept command. Requires {@code drive}. */
  public static Command create(Drive drive, Heimdall heimdall) {
    return Commands.run(() -> driveTowardIntercept(drive, heimdall), drive)
        .until(() -> arrived(drive, heimdall))
        .andThen(drive::stop)
        .withName("PredictiveIntercept");
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

  private static void driveTowardIntercept(Drive drive, Heimdall heimdall) {
    Pose2d pose = drive.getPose();
    Optional<TrackedObject> fuel = heimdall.getNearestFuel(pose.getTranslation());
    if (fuel.isEmpty()) {
      drive.stop(); // hold position, blocking the joystick default command
      return;
    }

    TrackedObject t = fuel.get();
    Translation2d rel = fuelTranslation(t).minus(pose.getTranslation()); // robot -> fuel now
    Translation2d fuelVel = new Translation2d(t.getVx(), t.getVy());

    // Lead vector from the robot to the predicted intercept point. Falls back to the fuel's
    // current position (pure pursuit) when there is no positive-time solution.
    Translation2d lead = interceptLead(rel, fuelVel, INTERCEPT_SPEED_MPS);
    if (lead == null) {
      lead = rel;
    }

    double leadDistance = lead.getNorm();
    double distanceToFuel = rel.getNorm();

    Logger.recordOutput(
        "TestMode/InterceptPoint", new Pose2d(pose.getTranslation().plus(lead), lead.getAngle()));

    // Speed proportional to the remaining distance to the actual fuel (so it decelerates on
    // arrival), directed along the lead vector, capped at cruise speed.
    double speed = Math.min(LINEAR_KP * distanceToFuel, INTERCEPT_SPEED_MPS);
    Translation2d fieldVelocity =
        leadDistance > 1e-6 ? lead.div(leadDistance).times(speed) : Translation2d.kZero;

    double headingError = lead.getAngle().minus(pose.getRotation()).getRadians();
    double omega = MathUtil.clamp(THETA_KP * headingError, -MAX_ANGULAR_RAD_PS, MAX_ANGULAR_RAD_PS);

    drive.runVelocity(
        ChassisSpeeds.fromFieldRelativeSpeeds(
            fieldVelocity.getX(), fieldVelocity.getY(), omega, pose.getRotation()));
  }

  /**
   * Solves the intercept: find the smallest time t &gt; 0 at which a robot leaving now at {@code
   * robotSpeed} can reach a fuel currently at {@code rel} (relative to the robot) moving at {@code
   * fuelVel}. Returns the lead vector (robot -&gt; intercept point), or {@code null} if the target
   * cannot be caught at that speed.
   *
   * <p>|rel + fuelVel*t| = robotSpeed*t expands to a*t^2 + b*t + c = 0.
   */
  private static Translation2d interceptLead(
      Translation2d rel, Translation2d fuelVel, double robotSpeed) {
    double a =
        fuelVel.getX() * fuelVel.getX() + fuelVel.getY() * fuelVel.getY() - robotSpeed * robotSpeed;
    double b = 2.0 * (rel.getX() * fuelVel.getX() + rel.getY() * fuelVel.getY());
    double c = rel.getX() * rel.getX() + rel.getY() * rel.getY();

    double t = smallestPositiveRoot(a, b, c);
    if (Double.isNaN(t)) {
      return null;
    }
    return new Translation2d(rel.getX() + fuelVel.getX() * t, rel.getY() + fuelVel.getY() * t);
  }

  /** Smallest strictly-positive root of a*x^2 + b*x + c, or NaN if none exists. */
  private static double smallestPositiveRoot(double a, double b, double c) {
    if (Math.abs(a) < 1e-9) {
      // Degenerate to linear b*x + c = 0 (robot speed ~= target speed).
      if (Math.abs(b) < 1e-9) {
        return Double.NaN;
      }
      double x = -c / b;
      return x > 1e-6 ? x : Double.NaN;
    }
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
      return Double.NaN;
    }
    double sqrt = Math.sqrt(disc);
    double x1 = (-b - sqrt) / (2.0 * a);
    double x2 = (-b + sqrt) / (2.0 * a);
    double lo = Math.min(x1, x2);
    double hi = Math.max(x1, x2);
    if (lo > 1e-6) {
      return lo;
    }
    if (hi > 1e-6) {
      return hi;
    }
    return Double.NaN;
  }
}
