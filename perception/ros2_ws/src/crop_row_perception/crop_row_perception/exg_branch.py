import cv2
import numpy as np


class ExGBranch:
    """
    Excess Green Index (ExG) branch for crop row detection.
    ExG = (2G - R - B) / (R + G + B)

    Most reliable at early growth stages (sparse canopy).
    Confidence drops as canopy closes and row structure disappears.
    """

    def __init__(self, morph_kernel: int = 5, sky_fraction: float = 0.25):
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        self.sky_fraction = sky_fraction  # top fraction of frame to mask as sky

    def compute(self, bgr_image: np.ndarray) -> dict:
        """
        Run ExG pipeline on a BGR image.

        Returns:
            dict with keys:
                mask        - binary vegetation mask (uint8)
                centerline  - Nx2 array of (x, y) centerline points
                confidence  - float [0, 1], how much to trust this branch
                exg_raw     - raw ExG float map
        """
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        total = r + g + b + 1e-6
        exg = (2.0 * g - r - b) / total

        # Sky mask — reject top sky_fraction of frame
        h = exg.shape[0]
        sky_mask = np.zeros_like(exg, dtype=np.uint8)
        sky_mask[int(h * self.sky_fraction):, :] = 1

        # Otsu threshold on valid region
        exg_uint8 = np.clip(exg * 255, 0, 255).astype(np.uint8)
        _, thresh = cv2.threshold(
            exg_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh = thresh * sky_mask

        # Morphological cleanup
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)

        centerline = self._extract_centerline(cleaned)

        # Confidence: ratio of green pixels in valid zone, scaled
        valid_pixels = sky_mask.sum() + 1e-6
        green_pixels = (cleaned > 0).sum()
        confidence = float(green_pixels / valid_pixels)
        confidence = min(confidence * 3.0, 1.0)

        return {
            "mask": cleaned,
            "centerline": centerline,
            "confidence": confidence,
            "exg_raw": exg,
        }

    def _extract_centerline(self, mask: np.ndarray) -> np.ndarray:
        """Column-wise centroid of vegetation pixels per row."""
        cols = np.arange(mask.shape[1])
        centerline = []
        for row_idx in range(mask.shape[0]):
            green_cols = cols[mask[row_idx] > 0]
            if len(green_cols) > 0:
                centerline.append((int(green_cols.mean()), row_idx))
        return np.array(centerline) if centerline else np.empty((0, 2), dtype=int)
