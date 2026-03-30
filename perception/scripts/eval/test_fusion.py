import cv2
import numpy as np
import sys
import os
sys.path.insert(0, '/home/agavale2/over_canopy_row_perception/perception/ros2_ws/src/crop_row_perception')
from crop_row_perception.exg_branch import ExGBranch
from crop_row_perception.growth_stage import GrowthStageClassifier, STAGE_WEIGHTS
from ultralytics import YOLO

img_dir = "/home/agavale2/Documents/Amiga/agronav/inference/input"
out_dir = "/tmp/fusion_test"
os.makedirs(out_dir, exist_ok=True)

# Load branches
exg       = ExGBranch()
yolo      = YOLO('/home/agavale2/over_canopy_row_perception/perception/checkpoints/yolov8/yolov8n_crop_row_v1.pt')
classifier = GrowthStageClassifier(default_stage="corn_mid")

def fuse_centerlines(exg_cl, yolo_mask, w_exg, w_yolo):
    """Simple weighted centerline fusion."""
    # Extract YOLO centerline from combined mask
    yolo_cl = []
    cols = np.arange(yolo_mask.shape[1])
    for row_idx in range(yolo_mask.shape[0]):
        green_cols = cols[yolo_mask[row_idx] > 0]
        if len(green_cols) > 0:
            yolo_cl.append((int(green_cols.mean()), row_idx))
    yolo_cl = np.array(yolo_cl) if yolo_cl else np.empty((0, 2), dtype=int)

    # Use highest weight branch centerline
    if w_exg >= w_yolo and len(exg_cl) > 1:
        return exg_cl, "ExG"
    elif len(yolo_cl) > 1:
        return yolo_cl, "YOLO"
    elif len(exg_cl) > 1:
        return exg_cl, "ExG"
    return np.empty((0, 2), dtype=int), "none"

for fname in sorted(os.listdir(img_dir)):
    if not fname.endswith('.jpg'):
        continue

    bgr = cv2.imread(os.path.join(img_dir, fname))
    h, w = bgr.shape[:2]
    if w > 1280:
        scale = 1280 / w
        bgr = cv2.resize(bgr, (1280, int(h * scale)))

    # 1. Growth stage → weights
    stage   = classifier.predict(bgr)
    weights = classifier.get_weights(stage)
    w_exg, w_yolo8, _ = weights

    # 2. ExG branch
    exg_out = exg.compute(bgr)

    # 3. YOLO branch
    yolo_results = yolo(bgr, conf=0.50, verbose=False)
    yolo_combined = np.zeros(bgr.shape[:2], dtype=np.uint8)
    n_masks = 0
    if yolo_results[0].masks is not None:
        for mask in yolo_results[0].masks.data.cpu().numpy():
            binary = cv2.resize(
                (mask > 0.5).astype(np.uint8) * 255,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            yolo_combined = cv2.bitwise_or(yolo_combined, binary)
            n_masks += 1

    # 4. Fuse
    centerline, dominant = fuse_centerlines(
        exg_out["centerline"], yolo_combined, w_exg, w_yolo8
    )

    # 5. Build visualization
    debug = bgr.copy()

    # ExG mask — green tint
    debug[exg_out["mask"] > 0] = (
        debug[exg_out["mask"] > 0] * 0.5 + np.array([0, 200, 0]) * 0.5
    ).astype(np.uint8)

    # YOLO mask — blue tint
    debug[yolo_combined > 0] = (
        debug[yolo_combined > 0] * 0.5 + np.array([200, 0, 0]) * 0.5
    ).astype(np.uint8)

    # Horizon line
    cv2.line(debug, (0, exg_out["horizon_row"]),
             (bgr.shape[1], exg_out["horizon_row"]), (0, 255, 255), 2)

    # Fused centerline — yellow
    for (cx, cy) in centerline[::8]:
        cv2.circle(debug, (cx, cy), 4, (0, 255, 255), -1)

    # Labels
    cv2.putText(debug, f"{fname} | stage={stage} | dominant={dominant}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(debug, f"ExG conf={exg_out['confidence']:.2f} w={w_exg:.2f} | YOLO masks={n_masks} w={w_yolo8:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(debug, "GREEN=ExG | BLUE=YOLO | CYAN=fused centerline",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imwrite(os.path.join(out_dir, fname), debug)

    print(f"{fname}: stage={stage} | ExG conf={exg_out['confidence']:.2f} | "
          f"YOLO masks={n_masks} | dominant={dominant} | "
          f"centerline_pts={len(centerline)}")

print(f"\nDone! Open with: eog {out_dir}/")
