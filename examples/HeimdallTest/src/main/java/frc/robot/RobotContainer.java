package frc.robot;

import static frc.robot.subsystems.vision.VisionConstants.*;

import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import edu.wpi.first.wpilibj2.command.button.CommandPS4Controller;
import edu.wpi.first.wpilibj2.command.sysid.SysIdRoutine;
import frc.robot.auto.AutoRoutines;
import frc.robot.commands.DriveCommands;
import frc.robot.subsystems.drive.Drive;
import frc.robot.subsystems.drive.GyroIO;
import frc.robot.subsystems.drive.GyroIOPigeon2;
import frc.robot.subsystems.drive.MapleSimSwerve;
import frc.robot.subsystems.drive.ModuleIO;
import frc.robot.subsystems.drive.ModuleIOSim;
import frc.robot.subsystems.drive.ModuleIOTalonFX;
import frc.robot.subsystems.heimdall.Heimdall;
import frc.robot.subsystems.heimdall.HeimdallIO;
import frc.robot.subsystems.heimdall.HeimdallIOReal;
import frc.robot.subsystems.heimdall.HeimdallIOSim;
import frc.robot.util.RobotIdentity;
import org.ironmaple.simulation.SimulatedArena;
import org.ironmaple.simulation.drivesims.SwerveDriveSimulation;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.networktables.LoggedDashboardChooser;

public class RobotContainer {
  // Subsystems
  private final Drive drive;
  private final Heimdall heimdall;

  // Controller
  private final CommandPS4Controller controller = new CommandPS4Controller(0);

  // Dashboard inputs
  private final LoggedDashboardChooser<Command> autoChooser;

  private SwerveDriveSimulation swerveDriveSimulation = null;

  public static boolean isRed() {
    var alliance = edu.wpi.first.wpilibj.DriverStation.getAlliance();
    return alliance.isPresent()
        && alliance.get() == edu.wpi.first.wpilibj.DriverStation.Alliance.Red;
  }

  public RobotContainer() {
    switch (Constants.currentMode) {
      case REAL:
        drive =
            new Drive(
                new GyroIOPigeon2(),
                new ModuleIOTalonFX(RobotIdentity.getTunerConstants().FrontLeft),
                new ModuleIOTalonFX(RobotIdentity.getTunerConstants().FrontRight),
                new ModuleIOTalonFX(RobotIdentity.getTunerConstants().BackLeft),
                new ModuleIOTalonFX(RobotIdentity.getTunerConstants().BackRight),
                swerveDriveSimulation);
        heimdall =
            new Heimdall(
                new HeimdallIOReal("10.62.38.200", drive::getPose),
                (pose, ts) ->
                    drive.addVisionMeasurement(
                        pose, ts, VecBuilder.fill(0.5, 0.5, Math.toRadians(10))));
        break;

      case SIM:
        swerveDriveSimulation =
            MapleSimSwerve.createSimulationDrive(RobotIdentity.getTunerConstants());
        drive =
            new Drive(
                new GyroIO() {},
                new ModuleIOSim(swerveDriveSimulation.getModules()[0]),
                new ModuleIOSim(swerveDriveSimulation.getModules()[1]),
                new ModuleIOSim(swerveDriveSimulation.getModules()[2]),
                new ModuleIOSim(swerveDriveSimulation.getModules()[3]),
                swerveDriveSimulation);
        // Dodge-obstacle trajectory: oscillates back and forth near the robot's start corner
        // so avoidance is exercised immediately. For the "Intercept Sim Ball" auto, swap back
        // to a single-sweep-and-park config instead, e.g. new HeimdallIOSim(0.1, 0.9, 0.5, 2.0,
        // false).
        heimdall = new Heimdall(new HeimdallIOSim(0.15, 0.45, 0.3, 1.5, true));
        break;

      default:
        drive =
            new Drive(
                new GyroIO() {},
                new ModuleIO() {},
                new ModuleIO() {},
                new ModuleIO() {},
                new ModuleIO() {},
                swerveDriveSimulation);
        heimdall = new Heimdall(new HeimdallIO() {});
        break;
    }

    // Set up auto routines
    LoggedDashboardChooser<Command> tempChooser;
    try {
      AutoRoutines autoRoutines = new AutoRoutines(drive, heimdall);
      tempChooser = new LoggedDashboardChooser<>("Auto Choices", autoRoutines.buildAutoChooser());
      tempChooser.addOption(
          "Drive Wheel Radius Characterization", DriveCommands.wheelRadiusCharacterization(drive));
      tempChooser.addOption(
          "Drive Simple FF Characterization", DriveCommands.feedforwardCharacterization(drive));
      tempChooser.addOption(
          "Drive SysId (Quasistatic Forward)",
          drive.sysIdQuasistatic(SysIdRoutine.Direction.kForward));
      tempChooser.addOption(
          "Drive SysId (Quasistatic Reverse)",
          drive.sysIdQuasistatic(SysIdRoutine.Direction.kReverse));
      tempChooser.addOption(
          "Drive SysId (Dynamic Forward)", drive.sysIdDynamic(SysIdRoutine.Direction.kForward));
      tempChooser.addOption(
          "Drive SysId (Dynamic Reverse)", drive.sysIdDynamic(SysIdRoutine.Direction.kReverse));
    } catch (Exception e) {
      e.printStackTrace();
      Alert alert = new Alert("auto failed to load", AlertType.kError);
      alert.set(true);
      tempChooser = new LoggedDashboardChooser<Command>("Auto Choices");
    }
    autoChooser = tempChooser;

    configureButtonBindings();
  }

  private void configureButtonBindings() {
    drive.setDefaultCommand(
        DriveCommands.joystickDrive(
            drive,
            () -> -controller.getLeftY(),
            () -> -controller.getLeftX(),
            () -> -controller.getRightX()));

    controller
        .square()
        .onTrue(
            Commands.runOnce(
                    () ->
                        drive.setPose(
                            new edu.wpi.first.math.geometry.Pose2d(
                                drive.getPose().getTranslation(),
                                isRed()
                                    ? edu.wpi.first.math.geometry.Rotation2d.fromDegrees(0)
                                    : edu.wpi.first.math.geometry.Rotation2d.fromDegrees(180))),
                    drive)
                .ignoringDisable(true));

    // X-lock wheels
    controller.circle().whileTrue(Commands.run(drive::stopWithX, drive));
  }

  public Command getAutonomousCommand() {
    return autoChooser.get();
  }

  public void updateSimulation() {
    SimulatedArena.getInstance().simulationPeriodic();

    if (swerveDriveSimulation != null) {
      Logger.recordOutput(
          "Odometry/SimulatedPose", swerveDriveSimulation.getSimulatedDriveTrainPose());
    }
  }
}
