package frc.robot.auto;

import static frc.robot.subsystems.vision.VisionConstants.aprilTagLayout;

import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj.smartdashboard.SendableChooser;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import frc.robot.commands.InterceptObjectCommand;
import frc.robot.commands.NavigateWithAvoidanceCommand;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.heimdall.Heimdall;

public class AutoRoutines {
  private final Drive drive;
  private final Heimdall heimdall;

  public AutoRoutines(Drive drive, Heimdall heimdall) {
    this.drive = drive;
    this.heimdall = heimdall;
  }

  public SendableChooser<Command> buildAutoChooser() {
    SendableChooser<Command> chooser = new SendableChooser<>();
    chooser.setDefaultOption("Do Nothing", Commands.none());
    chooser.addOption("Intercept Sim Ball", interceptSimBallAuto());
    chooser.addOption("Navigate Dodging Obstacle", navigateDodgeObstacleAuto());
    return chooser;
  }

  /**
   * Resets the robot to a fixed start pose, then chases the simulated ball as it moves at constant
   * velocity horizontally across the field (see {@link
   * frc.robot.subsystems.heimdall.HeimdallIOSim}).
   */
  private Command interceptSimBallAuto() {
    Pose2d startPose =
        new Pose2d(
            aprilTagLayout.getFieldLength() / 2.0,
            aprilTagLayout.getFieldWidth() * 0.1,
            Rotation2d.fromDegrees(90));

    return Commands.sequence(
            Commands.runOnce(() -> drive.setPose(startPose), drive),
            Commands.runOnce(heimdall::reset),
            new InterceptObjectCommand(drive, heimdall))
        .withTimeout(8.0);
  }

  /**
   * Resets the robot to a corner start pose, then drives it to the diagonally-opposite corner while
   * reactively dodging an obstacle that oscillates back and forth across a lane positioned near the
   * start of the robot's path -- so avoidance is exercised immediately rather than only once the
   * robot reaches mid-field (see {@link frc.robot.subsystems.heimdall.HeimdallIOSim} -- requires
   * the Heimdall sim IO in {@code RobotContainer} to be configured with an oscillating, close-in
   * trajectory, e.g. {@code new HeimdallIOSim(0.15, 0.45, 0.3, 1.5, true)}).
   */
  private Command navigateDodgeObstacleAuto() {
    double fieldLength = aprilTagLayout.getFieldLength();
    double fieldWidth = aprilTagLayout.getFieldWidth();

    Pose2d startPose = new Pose2d(fieldLength * 0.1, fieldWidth * 0.1, Rotation2d.fromDegrees(45));
    Translation2d goalPoint = new Translation2d(fieldLength * 0.9, fieldWidth * 0.9);

    return Commands.sequence(
            Commands.runOnce(() -> drive.setPose(startPose), drive),
            Commands.runOnce(heimdall::reset),
            new NavigateWithAvoidanceCommand(drive, heimdall, goalPoint))
        .withTimeout(10.0);
  }
}
