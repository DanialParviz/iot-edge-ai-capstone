"""Object detection with EfficientDet-Lite0 (8-bit TFLite) in MediaPipe.

NOTE: not yet verified from this repo; it follows the project's working setup
(`efficientdet_lite0_uint8.tflite`). The model file is git-ignored.
"""

from __future__ import annotations


class MediapipeDetector:
    def __init__(self, model_path: str, min_score: float = 0.5, max_results: int = 5) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            score_threshold=min_score,
            max_results=max_results,
        )
        self._detector = mp_vision.ObjectDetector.create_from_options(options)

    def detect(self, rgb_frame) -> list[tuple[str, float]]:
        """Returns (label, score) for each object found in an RGB frame."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        return [
            (d.categories[0].category_name, d.categories[0].score)
            for d in result.detections
            if d.categories
        ]
