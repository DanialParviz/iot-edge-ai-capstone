"""Face recognition: find a face (YuNet), embed it (EdgeFace XS), compare.

The comparison logic is pure Python and fully tested. The model wrappers need
OpenCV / PyTorch on the Pi and are NOT yet verified from this repo: they follow
the project's working setup (YuNet via OpenCV, EdgeFace XS, a saved reference
embedding) but the exact model file names and reference format should be checked
against the Pi.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

UNKNOWN = "Unknown"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding sizes differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def identify(
    embedding: Sequence[float], references: Mapping[str, Sequence[float]], threshold: float = 0.5
) -> tuple[str, float]:
    """Best matching name and its similarity, or ("Unknown", best) below ``threshold``."""
    best_name, best_sim = UNKNOWN, -1.0
    for name, ref in references.items():
        sim = cosine_similarity(embedding, ref)
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < threshold:
        return UNKNOWN, max(best_sim, 0.0)
    return best_name, best_sim


def load_reference(path: str) -> list[float]:
    """Load a saved reference embedding (a torch tensor in a .pt file)."""
    import torch

    tensor = torch.load(path, map_location="cpu")
    return [float(x) for x in tensor.flatten().tolist()]


class YuNetDetector:
    """Finds face boxes in a BGR frame with OpenCV's YuNet."""

    def __init__(self, model_path: str, score_threshold: float = 0.7) -> None:
        import cv2

        self._cv2 = cv2
        self._det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold)

    def detect(self, bgr_frame) -> list[tuple[int, int, int, int]]:
        h, w = bgr_frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(bgr_frame)
        if faces is None:
            return []
        return [tuple(int(v) for v in row[:4]) for row in faces]  # x, y, w, h


class EdgeFaceEmbedder:
    """512-number face embedding from EdgeFace XS (PyTorch)."""

    def __init__(self, variant: str = "edgeface_xs_gamma_06") -> None:
        import torch

        self._torch = torch
        self._model = torch.hub.load("otroshi/edgeface", variant, source="github", pretrained=True)
        self._model.eval()

    def embed(self, rgb_face) -> list[float]:
        """``rgb_face``: HxWx3 uint8 RGB crop of one face."""
        import cv2

        torch = self._torch
        face = cv2.resize(rgb_face, (112, 112))
        tensor = torch.from_numpy(face).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        with torch.no_grad():
            return [float(x) for x in self._model(tensor.unsqueeze(0))[0].tolist()]


def crop(frame, box: tuple[int, int, int, int], margin: float = 0.1):
    """Crop a box from a frame with a small margin, clamped to the frame."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, frame.shape[1]), min(y + h + my, frame.shape[0])
    return frame[y0:y1, x0:x1]


def best_face(boxes: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int, int, int]]:
    """The largest box (closest face), or None."""
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
