# Navigation

Het's ROS2 navigation stack for the Amiga robot.

## Nodes

| Node | Description | Status |
|---|---|---|
| Global Planner | Waypoint-based global path planning | ✅ Built |
| State Estimator | Fuses perception + GPS + odometry | ✅ Built |
| MPC Controller | Model predictive control → cmd_vel | ✅ Built |

## Inputs from Perception

| Topic | Type | Description |
|---|---|---|
| `/crop_row/lateral_m` | Float32MultiArray | Lateral offset from row center |
| `/crop_row/heading_rad` | Float32MultiArray | Heading error relative to row |
| `/crop_row/confidence` | Float32MultiArray | Ensemble perception confidence |

## Output

| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | Twist | Velocity commands to Amiga |
