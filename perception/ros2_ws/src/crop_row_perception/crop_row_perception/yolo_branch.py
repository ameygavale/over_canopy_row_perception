import numpy as np
import cv2
from typing import Optional


class YOLOBranch:
    """
    Wrapper for YOLOv8-seg and YOLOv10-seg crop row segmentation.

    YOLOv8-seg  — better for mid-season, standard NMS pipeline
    YOLOv10-seg — better for late-season dense canopy, NMS-free architecture

    Both models are loaded lazily on first inference call.
    """

    def __init__(self, model_path: str, model_type: str = "yolov8"):
        """
        Args:
            model_path: path to .pt weights file
            model_type: 'yolov8' or 'yolov10'
        """
        self.model_path = model_path
        self.model_type = model_type
        self._model = None

    def _load(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        print(f"[YOLOBranch] Loaded {self.model_type} from {self.model_path}")

    def infer(self, bgr_image: np.ndarray, conf_threshold: float = 0.25) -> dict:
        """
        Run segmentation inference on a BGR image.

        Returns:
            dict with keys:
                masks       - list of binary masks (H x W uint8)
                centerline  - Nx2 array of (x, y) fused centerline points
                confidence  - float [0, 1] mean detection confidence
        """
        if self._model is None:
            self._load()

        results = self._model(bgr_image, conf=conf_threshold, verbose=False)

        masks = []
        confidences = []

        if results[0].masks is not None:
            for mask, conf in zip(
                results[0].masks.data.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy()
            ):
                binary = (mask > 0.5).astype(np.uint8) * 255
                # Resize mask to original image size
                binary = cv2.resize(
                    binary,
                    (bgr_image.shape[1], bgr_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                masks.append(binary)
                confidences.append(float(conf))

        # Fuse all masks into one combined mask
        combined = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, m)

        centerline = self._extract_centerline(combined)
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "masks": masks,
            "combined_mask": combined,
            "centerline": centerline,
            "confidence": confidence,
        }

    def _extract_centerline(self, mask: np.ndarray) -> np.ndarray:
        cols = np.arange(mask.shape[1])
        centerline = []
        for row_idx in range(mask.shape[0]):
            green_cols = cols[mask[row_idx] > 0]
            if len(green_cols) > 0:
                centerline.append((int(green_cols.mean()), row_idx))
        return np.array(centerline) if centerline else np.empty((0, 2), dtype=int)
