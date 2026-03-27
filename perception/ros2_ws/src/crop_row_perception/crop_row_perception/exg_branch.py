import cv2
import numpy as np


class ExGBranch:
    """
    Excess Green Index (ExG) branch for crop row detection.
    ExG = (2G - R - B) / (R + G + B)

    Pipeline:
        1. HSV-based sky detection + dynamic horizon line
        2. Saturation-based soil/vegetation pre-discrimination
        3. ExG thresholding on valid (non-sky) region
        4. Soil false-positive removal
        5. Morphological cleanup
        6. Column-wise centerline extraction

    Tuned HSV thresholds (empirical, Agroscapes dataset):
        Sky:  H=[90,130], S=[20,200], V>=100  (blue sky)
              S<=40, V>=180                    (overcast/white sky)
        Soil: H=[5,35], S<130, V=[20,200]
        Veg:  H=[25,90], S>=130, V>=20
    """

    def __init__(self, morph_kernel: int = 5):
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    def _detect_sky_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        blue_sky = (
            (h >= 90) & (h <= 130) &
            (v >= 100) &
            (s >= 20) & (s <= 200)
        ).astype(np.uint8) * 255

        white_sky = (
            (s <= 40) & (v >= 180)
        ).astype(np.uint8) * 255

        sky = cv2.bitwise_or(blue_sky, white_sky)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, k)
        return sky

    def _detect_soil_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        soil = (
            (h >= 5) & (h <= 35) &
            (s < 130) &
            (v >= 20) & (v <= 200)
        ).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE, k)
        return soil

    def _find_horizon_row(self, sky_mask: np.ndarray) -> int:
        h, w = sky_mask.shape
        horizon_row = int(h * 0.20)
        for row in range(h - 1, -1, -1):
            if sky_mask[row].sum() / 255.0 / w > 0.15:
                horizon_row = row
                break
        return horizon_row

    def compute(self, bgr_image: np.ndarray) -> dict:
        h, w = bgr_image.shape[:2]

        # 1. Sky + soil masks
        sky_mask  = self._detect_sky_mask(bgr_image)
        soil_mask = self._detect_soil_mask(bgr_image)

        # 2. Dynamic horizon
        horizon_row = self._find_horizon_row(sky_mask)

        # 3. Valid region: below horizon, not sky
        valid_mask = np.zeros((h, w), dtype=np.uint8)
        valid_mask[horizon_row:, :] = 255
        valid_mask[sky_mask > 0] = 0

        # 4. ExG
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        total = r + g + b + 1e-6
        exg = (2.0 * g - r - b) / total

        # 5. Threshold on valid region
        exg_uint8 = np.clip(exg * 255, 0, 255).astype(np.uint8)
        _, thresh = cv2.threshold(
            exg_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh[valid_mask == 0] = 0

        # 6. Remove soil false positives
        thresh[soil_mask > 0] = 0

        # 7. Morphological cleanup
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)

        # 8. Centerline
        centerline = self._extract_centerline(cleaned)

        # 9. Confidence
        valid_pixels = (valid_mask > 0).sum() + 1e-6
        green_pixels = (cleaned > 0).sum()
        confidence = min(float(green_pixels / valid_pixels) * 3.0, 1.0)

        return {
            "mask":        cleaned,
            "centerline":  centerline,
            "confidence":  confidence,
            "exg_raw":     exg,
            "sky_mask":    sky_mask,
            "soil_mask":   soil_mask,
            "horizon_row": horizon_row,
        }

    def _extract_centerline(self, mask: np.ndarray) -> np.ndarray:
        cols = np.arange(mask.shape[1])
        centerline = []
        for row_idx in range(mask.shape[0]):
            green_cols = cols[mask[row_idx] > 0]
            if len(green_cols) > 0:
                centerline.append((int(green_cols.mean()), row_idx))
        return np.array(centerline) if centerline else np.empty((0, 2), dtype=int)
