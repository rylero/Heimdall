package frc.robot.subsystems.heimdall;

import com.heimdall.TrackedObject;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import org.littletonrobotics.junction.Logger;

public class Heimdall extends SubsystemBase {
  private final HeimdallIO io;
  private final HeimdallIOInputsAutoLogged inputs = new HeimdallIOInputsAutoLogged();

  public Heimdall(HeimdallIO io) {
    this.io = io;
  }

  @Override
  public void periodic() {
    io.updateInputs(inputs);
    Logger.processInputs("Heimdall", inputs);

    Logger.recordOutput(
        "Heimdall/TrackedObjectPositions",
        getTrackedObjects().stream()
            .map((obj) -> new Pose2d(obj.getX(), obj.getY(), Rotation2d.kZero))
            .toArray(Pose2d[]::new));
  }

  public boolean isConnected() {
    return inputs.connected;
  }

  /** Restarts any time-based simulated trajectory. No-op for real hardware IO. */
  public void reset() {
    io.reset();
  }

  /** Freezes any time-based simulated trajectory in place. No-op for real hardware IO. */
  public void markObjectHit() {
    io.markHit();
  }

  /** Returns the currently tracked objects, translated into vendordep TrackedObject form. */
  public List<TrackedObject> getTrackedObjects() {
    return Arrays.stream(inputs.objects)
        .map(
            (t) ->
                new TrackedObject(
                    t.trackId(),
                    t.classId(),
                    t.x(),
                    t.y(),
                    t.vx(),
                    t.vy(),
                    0.0,
                    0.0,
                    t.confidence()))
        .toList();
  }

  /** Returns the highest-confidence tracked object, if any are visible. */
  public Optional<TrackedObject> getBestTrackedObject() {
    return getTrackedObjects().stream()
        .max(Comparator.comparingDouble(TrackedObject::getConfidence));
  }
}
