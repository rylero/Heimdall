package frc.robot.subsystems.heimdall;

import static frc.robot.subsystems.vision.VisionConstants.aprilTagLayout;

import edu.wpi.first.wpilibj.Timer;

/**
 * Sim IO that synthesizes a single object moving horizontally across the field at constant speed,
 * bypassing the Heimdall client/ZMQ entirely. The trajectory (lane, span, speed, and whether it
 * bounces back and forth or makes a single sweep-and-park run) is configurable so the same class
 * can stand in for either the intercept-ball or a dodge-obstacle.
 */
public class HeimdallIOSim implements HeimdallIO {
  private static final int TRACK_ID = 0;
  private static final int CLASS_ID = 0;

  private final double speedMetersPerSec;
  private final double startX;
  private final double endX;
  private final double laneY;
  private final boolean oscillate;
  private double startTime = Timer.getFPGATimestamp();
  private boolean hit = false;
  private double hitX;

  /**
   * @param startFraction fraction of field length where the object's run begins
   * @param endFraction fraction of field length where the object's run ends -- it parks here if
   *     {@code oscillate} is false, or reverses direction here if {@code oscillate} is true
   * @param laneFraction fraction of field width at which the object travels (its constant Y)
   * @param speedMetersPerSec constant travel speed, m/s
   * @param oscillate if true, the object bounces back and forth between the start and end
   *     indefinitely (good for stress-testing reactive avoidance); if false, it sweeps once from
   *     start to end and parks there (e.g. so an intercept command has a catchable target)
   */
  public HeimdallIOSim(
      double startFraction,
      double endFraction,
      double laneFraction,
      double speedMetersPerSec,
      boolean oscillate) {
    double fieldLength = aprilTagLayout.getFieldLength();
    this.speedMetersPerSec = speedMetersPerSec;
    this.oscillate = oscillate;
    startX = fieldLength * startFraction;
    endX = fieldLength * endFraction;
    laneY = aprilTagLayout.getFieldWidth() * laneFraction;
  }

  /** Restarts the object at the near edge of its lane -- call at the start of each auto run. */
  @Override
  public void reset() {
    startTime = Timer.getFPGATimestamp();
    hit = false;
  }

  /** Freezes the object in place at its current position -- call once the robot contacts it. */
  @Override
  public void markHit() {
    if (!hit) {
      hit = true;
      hitX = positionAt(Timer.getFPGATimestamp() - startTime).x();
    }
  }

  @Override
  public void updateInputs(HeimdallIOInputs inputs) {
    double x;
    double vx;
    if (hit) {
      x = hitX;
      vx = 0.0;
    } else {
      Position p = positionAt(Timer.getFPGATimestamp() - startTime);
      x = p.x();
      vx = p.vx();
    }

    inputs.connected = true;
    inputs.objects =
        new TrackedTarget[] {new TrackedTarget(TRACK_ID, CLASS_ID, x, laneY, vx, 0.0, 1.0)};
  }

  private record Position(double x, double vx) {}

  /** Computes the object's field-X position and velocity {@code elapsedSeconds} after start. */
  private Position positionAt(double elapsedSeconds) {
    double span = endX - startX;
    double distance = speedMetersPerSec * elapsedSeconds;

    if (oscillate) {
      // Triangle wave: reflects off each end of the span indefinitely.
      double cycle = 2.0 * span;
      double distanceInCycle = distance % cycle;
      boolean forwardLeg = distanceInCycle <= span;
      double traveled = forwardLeg ? distanceInCycle : cycle - distanceInCycle;
      return new Position(startX + traveled, forwardLeg ? speedMetersPerSec : -speedMetersPerSec);
    }

    double traveled = Math.min(distance, span);
    boolean stillMoving = traveled < span;
    return new Position(startX + traveled, stillMoving ? speedMetersPerSec : 0.0);
  }
}
