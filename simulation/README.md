# Simulation

NVIDIA Isaac Sim setup for validating the full perception → MPC pipeline
before field deployment.

## Setup

Isaac Sim is installed on the lab workstation (RTX 5070 Ti).
Launch with ROS2 bridge:

```bash
~/launch_isaacsim_ros2.sh
```

## Planned Scenes

- [ ] Crop row scene (corn, early stage)
- [ ] Crop row scene (corn, late stage)
- [ ] Crop row scene (soybean, early stage)
- [ ] Crop row scene (soybean, late stage)

## Workstream

1. Build crop row Omniverse scene
2. Wire mock perception publisher → Het's MPC
3. Validate state estimator in simulation
4. Tune MPC gains before field testing

See the 5-day Isaac Sim learning plan in `docs/`.
