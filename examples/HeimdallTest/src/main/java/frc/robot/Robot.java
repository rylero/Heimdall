package frc.robot;

import com.ctre.phoenix6.CANBus;
import com.ctre.phoenix6.swerve.SwerveModuleConstants;
import com.ctre.phoenix6.swerve.SwerveModuleConstants.DriveMotorArrangement;
import com.ctre.phoenix6.swerve.SwerveModuleConstants.SteerMotorArrangement;
import com.pathplanner.lib.commands.PathfindingCommand;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj.RobotController;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.CommandScheduler;
import frc.robot.util.RobotIdentity;
import org.ironmaple.simulation.SimulatedArena;
import org.ironmaple.simulation.seasonspecific.evergreen.ArenaEvergreen;
import org.littletonrobotics.junction.LogFileUtil;
import org.littletonrobotics.junction.LoggedRobot;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.networktables.NT4Publisher;
import org.littletonrobotics.junction.wpilog.WPILOGReader;
import org.littletonrobotics.junction.wpilog.WPILOGWriter;

public class Robot extends LoggedRobot {
  private Command autonomousCommand;
  private RobotContainer robotContainer;
  private final Alert lowBatteryAlert =
      new Alert("Battery voltage low (<11V) — check before match", AlertType.kWarning);
  private final Alert canivoreAlert =
      new Alert("CANivore bus error — check CAN wiring and termination", AlertType.kError);
  private final CANBus canivoreBus = new CANBus("canivore");

  public Robot() {
    // Record metadata
    Logger.recordMetadata("ProjectName", BuildConstants.MAVEN_NAME);
    Logger.recordMetadata("BuildDate", BuildConstants.BUILD_DATE);
    Logger.recordMetadata("GitSHA", BuildConstants.GIT_SHA);
    Logger.recordMetadata("GitDate", BuildConstants.GIT_DATE);
    Logger.recordMetadata("GitBranch", BuildConstants.GIT_BRANCH);
    Logger.recordMetadata("RoborioSerial", RobotController.getSerialNumber());
    switch (BuildConstants.DIRTY) {
      case 0:
        Logger.recordMetadata("GitDirty", "All changes committed");
        break;
      case 1:
        Logger.recordMetadata("GitDirty", "Uncomitted changes");
        break;
      default:
        Logger.recordMetadata("GitDirty", "Unknown");
        break;
    }

    // Set up data receivers & replay source
    switch (Constants.currentMode) {
      case REAL:
        Logger.addDataReceiver(new WPILOGWriter());
        Logger.addDataReceiver(new NT4Publisher());
        break;
      case SIM:
        Logger.addDataReceiver(new NT4Publisher());
        Logger.addDataReceiver(new WPILOGWriter("logs/"));
        break;
      case REPLAY:
        setUseTiming(false);
        String logPath = LogFileUtil.findReplayLog();
        Logger.setReplaySource(new WPILOGReader(logPath));
        Logger.addDataReceiver(new WPILOGWriter(LogFileUtil.addPathSuffix(logPath, "_sim")));
        break;
    }

    Logger.start();

    // Check for valid swerve config
    var modules =
        new SwerveModuleConstants[] {
          RobotIdentity.getTunerConstants().FrontLeft,
          RobotIdentity.getTunerConstants().FrontRight,
          RobotIdentity.getTunerConstants().BackLeft,
          RobotIdentity.getTunerConstants().BackRight
        };
    for (var constants : modules) {
      if (constants.DriveMotorType != DriveMotorArrangement.TalonFX_Integrated
          || constants.SteerMotorType != SteerMotorArrangement.TalonFX_Integrated) {
        throw new RuntimeException(
            "You are using an unsupported swerve configuration, which this template does not support without manual customization.");
      }
    }

    if (Constants.currentMode == Constants.Mode.SIM) {
      SimulatedArena.overrideInstance(new ArenaEvergreen(true));
    }

    robotContainer = new RobotContainer();

    PathfindingCommand.warmupCommand().ignoringDisable(true).schedule();
  }

  @Override
  public void robotPeriodic() {
    CommandScheduler.getInstance().run();
    lowBatteryAlert.set(RobotController.getBatteryVoltage() < 11.0);
    canivoreAlert.set(!canivoreBus.getStatus().Status.isOK());
  }

  @Override
  public void disabledInit() {}

  @Override
  public void disabledPeriodic() {}

  @Override
  public void autonomousInit() {
    autonomousCommand = robotContainer.getAutonomousCommand();
    if (Constants.currentMode == frc.robot.Constants.Mode.SIM) {
      SimulatedArena.getInstance().resetFieldForAuto();
    }
    if (autonomousCommand != null) {
      autonomousCommand.schedule();
    }
  }

  @Override
  public void autonomousPeriodic() {}

  @Override
  public void teleopInit() {
    if (autonomousCommand != null) {
      autonomousCommand.cancel();
    }
  }

  @Override
  public void teleopPeriodic() {}

  @Override
  public void testInit() {
    CommandScheduler.getInstance().cancelAll();
    // Test mode = "drive to the nearest fuel and stop" (see PathfindToNearestFuelCommand).
    robotContainer.getTestModeCommand().schedule();
  }

  @Override
  public void testPeriodic() {}

  @Override
  public void testExit() {}

  @Override
  public void simulationInit() {}

  @Override
  public void simulationPeriodic() {
    robotContainer.updateSimulation();

    // var nt = NetworkTableInstance.getDefault();
    // DriverStationSim.setEnabled(nt.getEntry("/Sim/Enable").getBoolean(false));
    // DriverStationSim.setAutonomous(nt.getEntry("/Sim/Autonomous").getBoolean(false));
    // DriverStationSim.setTest(nt.getEntry("/Sim/Test").getBoolean(false));
    // DriverStationSim.notifyNewData();
  }
}
