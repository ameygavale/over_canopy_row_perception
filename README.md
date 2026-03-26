# Over-Canopy Crop Row Perception — Amiga Robot

**NextGen Embodied AI Solutions Lab • UIUC**

Multi-branch crop row perception pipeline for the farm-ng Amiga robot,
targeting multi-growth-stage robustness across corn and soybean crops.
Camera mounted at ~6ft height, 10-20° downward tilt from horizontal.

## Scope

This repo covers **perception only**. Navigation stack lives at:
→ [het915/farnav_amiga](https://github.com/het915/farnav_amiga)

## Repository Structure
```
over_canopy_row_perception/
├── perception/          ← Multi-branch crop row perception pipeline
│   ├── ros2_ws/         ← ROS2 workspace
│   ├── data/            ← Datasets (gitignored, structure tracked)
│   ├── checkpoints/     ← Model weights (gitignored, structure tracked)
│   └── scripts/         ← SAM pseudo-labeling, training, eval scripts
├── simulation/          ← Isaac Sim crop row scenes and scripts
├── gps/                 ← RTK GPS arbitration pipeline
└── notebooks/           ← EDA and visualization notebooks
```

## Perception Pipeline
```
OAK-D Wide (6ft height, 10-20° downward tilt)
├── RGB  → Growth Stage Classifier → branch weights
│        → ExG Branch              → mask + centerline
│        → YOLOv8-seg Branch       → mask + centerline
│        → YOLOv10-seg Branch      → mask + centerline
│              ↓
│        Weighted Fusion → fused centerline (pixels)
│
└── Depth → DepthProjection → lateral_m, heading_rad
```

## ROS2 Interface (outputs to Het's navigation stack)

| Topic | Type | Description |
|---|---|---|
| `/crop_row/lateral_m` | Float32MultiArray | Lateral offset from row center (meters) |
| `/crop_row/heading_rad` | Float32MultiArray | Heading error relative to row (radians) |
| `/crop_row/confidence` | Float32MultiArray | Ensemble perception confidence [0,1] |

## Datasets

| Dataset | Location | Status |
|---|---|---|
| Agroscapes | `perception/ros2_ws/src/crop_row_perception/agronav` | ✅ Submodule |
| JunfengGaolab CropRowDetection | `perception/data/raw/junfeng_croprow` | ✅ Submodule |
| VegAnn | `perception/data/raw/vegann/` | ✅ Extracted locally |

## Hardware

| Component | Details |
|---|---|
| Robot | farm-ng Amiga ("lavender-latency") |
| Camera | OAK-D Wide (150° DFOV, Myriad X VPU) |
| Mount height | ~6ft, 10-20° downward tilt |
| GPS | Emlid RS3 + PointPerfect fallback |
| Compute | Alienware RTX 5090 / Lab workstation RTX 5070 Ti |

## Setup
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ameygavale/over_canopy_row_perception.git
cd over_canopy_row_perception

# Activate PyTorch env
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
