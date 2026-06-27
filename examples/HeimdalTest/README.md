# HeimdalTest — Example Robot

A complete FRC robot project showing how to integrate Heimdall into a swerve drive robot using AdvantageKit.

## What's in here

- `subsystems/heimdall/` — the Heimdall integration layer
  - `HeimdallIO.java` / `HeimdallIOSim.java` / `HeimdallIOReal.java` — IO interface + sim/real implementations
  - `Heimdall.java` — subsystem that exposes `getTrackedTargets()` and feeds drive pose to the Jetson
- `commands/InterceptObjectCommand.java` — example command that drives toward a tracked target
- `commands/NavigateWithAvoidanceCommand.java` — example command using tracked targets as dynamic obstacles

The rest (drive subsystem, vision, auto) is standard AdvantageKit swerve boilerplate included for context.

## Setup

### 1. Publish the vendordep

From the root of your Heimdall clone:

```bash
cd vendordep
./gradlew publish
```

This writes the JAR to `vendordep/maven/`.

### 2. Point build.gradle at it

`build.gradle` is pre-configured to resolve the vendordep via a relative path (`../../vendordep/maven`), which works when this project lives at `examples/HeimdalTest/` inside the Heimdall repo. If you copy the project elsewhere, update the `url` in the `repositories` block to match your local path.

### 3. Configure team number

Edit `.wpilib/wpilib_preferences.json` and set `"teamNumber"` to your team.

### 4. Configure the Jetson IP

In `HeimdallIOReal.java`, set the host to your Jetson's static IP:

```java
new HeimdallClient("10.TE.AM.200", ...)
```

### 5. Deploy

```bash
./gradlew deploy
```

## Sim support

`HeimdallIOSim.java` generates synthetic tracked targets so you can develop and test commands without a Jetson connected. Run the simulator normally via WPILib VS Code extension or:

```bash
./gradlew simulateJava
```
