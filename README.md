# Agricultural Robotics — Amiga Robot

**NextGen Embodied AI Solutions Lab • UIUC**

Crop row guidance and navigation system for the farm-ng Amiga robot,
targeting multi-growth-stage robustness across corn and soybean crops.

## Repository Structure

```
agricultural_robotics_amiga/
├── perception/          ← Multi-branch crop row perception pipeline
│   ├── ros2_ws/         ← ROS2 workspace
│   ├── data/            ← Datasets (gitignored, structure tracked)
│   ├── checkpoints/     ← Model weights (gitignored, structure tracked)
│   └── scripts/         ← SAM pseudo-labeling, training, eval scripts
├── navigation/          ← Het's state estimator + MPC navigation stack
├── simulation/          ← Isaac Sim crop row scenes and scripts
├── gps/                 ← RTK GPS arbitration and correction pipeline
└── notebooks/           ← EDA and visualization notebooks
```

## Perception Pipeline

```
OAK-D Wide
├── RGB  → Growth Stage Classifier → branch weights
│        → ExG Branch              → mask + centerline
│        → YOLOv8-seg Branch       → mask + centerline
│        → YOLOv10-seg Branch      → mask + centerline
│              ↓
│        Weighted Fusion → fused centerline (pixels)
│
└── Depth → DepthProjection → lateral_m, heading_rad
                                    ↓
                          Het's State Estimator → MPC → cmd_vel → Amiga
```

## Hardware

| Component | Details |
|---|---|
| Robot | farm-ng Amiga ("lavender-latency") |
| Camera | OAK-D Wide (150° DFOV, Myriad X VPU) |
| GPS | Emlid RS3 base station + PointPerfect fallback |
| Compute | Alienware (RTX 5090) / Lab workstation (RTX 5070 Ti) |

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ameygavale/agricultural_robotics_amiga.git
cd agricultural_robotics_amiga

# Activate PyTorch env (Alienware)
conda activate torch_sm120

# Build ROS2 workspace
cd perception/ros2_ws
colcon build
source install/setup.bash

# Run fusion node
ros2 run crop_row_perception fusion_node \
    --ros-args --params-file src/crop_row_perception/config/params.yaml
```

## Publication Target

ICRA / IROS / Journal of Field Robotics
