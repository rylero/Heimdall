package frc.robot.subsystems.heimdall;

import org.littletonrobotics.junction.AutoLog;

public interface HeimdallIO {
  @AutoLog
  public static class HeimdallIOInputs {
    public boolean connected = false;
    public double timeSinceLastFrameSecs = Double.MAX_VALUE;
    public String lastError = "";
    public TrackedTarget[] objects = new TrackedTarget[0];

    // AprilTag vision pose — valid=true means a fresh estimate arrived this cycle.
    public boolean aprilTagPoseValid = false;
    public double aprilTagX = 0.0;
    public double aprilTagY = 0.0;
    public double aprilTagHeadingRad = 0.0;
    public double aprilTagTimestampSecs = 0.0;
  }

  /** A single tracked object reported by Heimdall, field-relative in meters and m/s. */
  public static record TrackedTarget(
      int trackId, int classId, double x, double y, double vx, double vy, double confidence) {}

  public default void updateInputs(HeimdallIOInputs inputs) {}

  /** Restarts any time-based simulated trajectory. No-op for real hardware IO. */
  public default void reset() {}

  /** Freezes any time-based simulated trajectory in place. No-op for real hardware IO. */
  public default void markHit() {}
}
