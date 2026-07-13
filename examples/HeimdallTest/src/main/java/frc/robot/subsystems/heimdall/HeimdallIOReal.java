package frc.robot.subsystems.heimdall;

import com.heimdall.DetectionFrame;
import com.heimdall.HeimdallClient;
import com.heimdall.TrackEvent;
import com.heimdall.TrackEventType;
import com.heimdall.TrackedObject;
import com.heimdall.VisionPoseEstimate;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.wpilibj.Timer;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/** Real hardware IO -- bridges the ZeroMQ {@link HeimdallClient} to {@link HeimdallIO}. */
public class HeimdallIOReal implements HeimdallIO {
  private final HeimdallClient client;
  private final Supplier<Pose2d> poseSupplier;

  // Maintains a current snapshot keyed by trackId: CONFIRMED/UPDATED upsert, LOST removes.
  private final Map<Integer, TrackedObject> tracks = new LinkedHashMap<>();
  private long lastProcessedTimestampNs = -1;
  private String lastError = "";

  // Estimated pipeline latency from image capture to robot receipt (seconds).
  // Used to convert Jetson CLOCK_MONOTONIC timestamps to the FPGA time domain
  // that WPILib's pose estimator expects.
  private static final double PIPELINE_LATENCY_SECS = 0.10;

  // The Jetson pipeline sends a tracking frame every cycle even when nothing is detected,
  // so a healthy connection keeps timeSinceLastFrameSecs near zero. If it grows past this,
  // the pipeline has died or restarted -- there is no "resync" message, so a crashed/restarted
  // Jetson process leaves stale tracks in `tracks` forever unless we evict them ourselves.
  private static final double STALE_CONNECTION_SECS = 1.0;

  public HeimdallIOReal(String jetsonHost, Supplier<Pose2d> poseSupplier) {
    this.client = new HeimdallClient(jetsonHost);
    this.poseSupplier = poseSupplier;
  }

  @Override
  public void updateInputs(HeimdallIOInputs inputs) {
    Pose2d pose = poseSupplier.get();
    try {
      client.sendPose(pose.getX(), pose.getY(), pose.getRotation().getRadians());
    } catch (Exception e) {
      lastError = "sendPose: " + e.getMessage();
    }

    try {
      DetectionFrame frame = client.getLatestFrame();
      if (frame != null && frame.getTimestampNs() != lastProcessedTimestampNs) {
        lastProcessedTimestampNs = frame.getTimestampNs();
        for (TrackEvent event : frame.getEvents()) {
          TrackedObject obj = event.getObject();
          if (event.getType() == TrackEventType.LOST) {
            tracks.remove(obj.getTrackId());
          } else {
            tracks.put(obj.getTrackId(), obj);
          }
        }
      }
    } catch (Exception e) {
      lastError = "getLatestFrame: " + e.getMessage();
    }

    VisionPoseEstimate vpe = client.getLatestVisionPose();
    inputs.aprilTagPoseValid = vpe != null;
    if (vpe != null) {
      inputs.aprilTagX = vpe.getX();
      inputs.aprilTagY = vpe.getY();
      inputs.aprilTagHeadingRad = vpe.getHeadingRad();
      // The timestamp from the Jetson is CLOCK_MONOTONIC (kernel boot time).
      // WPILib's pose estimator expects FPGA time.  Approximate it by
      // subtracting the estimated pipeline latency from current FPGA time.
      inputs.aprilTagTimestampSecs = Timer.getFPGATimestamp() - PIPELINE_LATENCY_SECS;
    }

    inputs.connected = client.isHealthy();
    inputs.timeSinceLastFrameSecs = client.getTimeSinceLastFrameSecs();
    inputs.lastError = lastError;

    if (inputs.timeSinceLastFrameSecs > STALE_CONNECTION_SECS) {
      tracks.clear();
    }

    inputs.objects =
        tracks.values().stream()
            .map(
                (t) ->
                    new TrackedTarget(
                        t.getTrackId(),
                        t.getClassId(),
                        t.getX(),
                        t.getY(),
                        t.getVx(),
                        t.getVy(),
                        t.getConfidence()))
            .toArray(TrackedTarget[]::new);
  }
}
