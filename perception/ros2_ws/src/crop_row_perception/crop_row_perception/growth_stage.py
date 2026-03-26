import numpy as np


# Growth stage definitions
# Each stage drives ExG vs YOLO confidence weighting in fusion
GROWTH_STAGES = [
    "corn_early",
    "corn_mid",
    "corn_late",
    "corn_canopy",
    "soy_early",
    "soy_mid",
    "soy_late",
    "soy_canopy",
]

# Branch weights per growth stage [exg, yolov8, yolov10]
STAGE_WEIGHTS = {
    "corn_early":   [0.80, 0.15, 0.05],
    "corn_mid":     [0.40, 0.40, 0.20],
    "corn_late":    [0.15, 0.35, 0.50],
    "corn_canopy":  [0.05, 0.20, 0.75],
    "soy_early":    [0.80, 0.15, 0.05],
    "soy_mid":      [0.35, 0.40, 0.25],
    "soy_late":     [0.10, 0.30, 0.60],
    "soy_canopy":   [0.05, 0.15, 0.80],
}


class GrowthStageClassifier:
    """
    Lightweight growth stage classifier.

    TODO: Replace stub with trained MobileNetV2/EfficientNet-B0 classifier.
    Currently returns a fixed stage for pipeline testing.
    """

    def __init__(self, model_path: str = None, default_stage: str = "corn_mid"):
        self.model_path = model_path
        self.default_stage = default_stage
        self._model = None

        if model_path:
            self._load()

    def _load(self):
        # TODO: load trained classifier weights
        pass

    def predict(self, bgr_image: np.ndarray) -> str:
        """
        Predict growth stage from image.
        Returns one of GROWTH_STAGES strings.
        """
        if self._model is None:
            # Stub: return default stage until classifier is trained
            return self.default_stage

        # TODO: run inference with trained model
        return self.default_stage

    def get_weights(self, stage: str) -> list:
        """Return [exg_weight, yolov8_weight, yolov10_weight] for a stage."""
        return STAGE_WEIGHTS.get(stage, STAGE_WEIGHTS["corn_mid"])
