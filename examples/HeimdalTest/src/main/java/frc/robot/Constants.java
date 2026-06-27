package frc.robot;

import edu.wpi.first.wpilibj.RobotBase;

/**
 * This class defines the runtime mode used by AdvantageKit. The mode is always "real" when running
 * on a roboRIO. Change the value of "simMode" to switch between "sim" (physics sim) and "replay"
 * (log replay from a file).
 */
public final class Constants {
  public static final Mode simMode = Mode.SIM;
  public static final Mode currentMode = RobotBase.isReal() ? Mode.REAL : simMode;

  public static enum Mode {
    /** Running on a real robot. */
    REAL,

    /** Running a physics simulator. */
    SIM,

    /** Replaying from a log file. */
    REPLAY
  }

  public static final int MAX_PHEONIX_RETRIES = 5;

  public final class PathfindingConfig {
    public static final double DRIVE_RESUME_DEADBAND = 0.2;
  }

  /**
   * When true, suppresses non-essential Logger.recordOutput calls to reduce log file size. All IO
   * inputs (Logger.processInputs) are always logged regardless of this flag.
   */
  public static final boolean MINIMAL_LOGGING = false;

  public static final double loopPeriodSecs = 0.02;
}
