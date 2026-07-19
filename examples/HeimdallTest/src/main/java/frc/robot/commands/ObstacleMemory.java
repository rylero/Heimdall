package frc.robot.commands;

import com.heimdall.TrackedObject;
import edu.wpi.first.math.geometry.Translation2d;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Short-term memory of obstacles for the drive-ahead avoidance commands. An object that gets very
 * close leaves the camera FOV and stops being detected right when the robot is on top of it;
 * without memory the avoidance would forget it at the worst moment. This keeps each recently seen
 * object (moving or stationary) for {@link #HOLD_SECONDS} after its last detection, dead-reckoning
 * its position forward from its last known velocity so the avoidance keeps pushing until the robot
 * has cleared it.
 *
 * <p>One instance per command run (created in the deferred command body), so state resets on each
 * press. Not thread-safe -- only touched from the command's periodic on the main robot loop.
 */
final class ObstacleMemory {
  /** How long a vanished obstacle is remembered and dead-reckoned before it is dropped. */
  private static final double HOLD_SECONDS = 1.5;

  /** A remembered obstacle, position propagated to the query time. */
  record Obstacle(Translation2d position, Translation2d velocity) {}

  private static final class Entry {
    Translation2d position;
    Translation2d velocity;
    double lastSeen;
  }

  private final Map<Integer, Entry> entries = new HashMap<>();

  /** Refreshes memory with the currently tracked objects (moving or stationary). */
  void update(List<TrackedObject> tracked, double now) {
    for (TrackedObject o : tracked) {
      Entry e = entries.computeIfAbsent(o.getTrackId(), (k) -> new Entry());
      e.position = new Translation2d(o.getX(), o.getY());
      e.velocity = new Translation2d(o.getVx(), o.getVy());
      e.lastSeen = now;
    }
  }

  /**
   * Returns the remembered obstacles with positions propagated to {@code now}, dropping any older
   * than {@link #HOLD_SECONDS}. Currently visible objects come back at their fresh position (age
   * ~0); recently vanished ones are dead-reckoned along their last velocity.
   */
  List<Obstacle> obstacles(double now) {
    List<Obstacle> out = new ArrayList<>();
    Iterator<Entry> it = entries.values().iterator();
    while (it.hasNext()) {
      Entry e = it.next();
      double age = now - e.lastSeen;
      if (age > HOLD_SECONDS) {
        it.remove();
        continue;
      }
      Translation2d predicted = e.position.plus(e.velocity.times(age));
      out.add(new Obstacle(predicted, e.velocity));
    }
    return out;
  }
}
