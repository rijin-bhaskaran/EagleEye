"""Small face-attribute network: gender and age group from one face crop.

MobileNetV3-small (about 1.5 M parameters, light enough for a Raspberry Pi and exportable to ONNX for the
accelerator plan) with two heads: gender (2 classes) and age group (9 FairFace groups). ``TorchAgeGender`` wraps
it in the ``predict(face_bgr) -> (p_male, p_female, p_child)`` interface the pipeline already uses, so it drops in
where the pretrained Levi-Hassner nets are.

    P(child) = P(age 0-2) + P(age 3-9) + ``teen_weight`` * P(age 10-19)

The 10-19 group mixes children (10-12) and teenagers, so how much of it counts as "child" is a tunable weight,
chosen on data that was not used for training (see ``passenger.tune_classifier``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

AGE_GROUPS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
CHILD_GROUPS = (0, 1)                 # 0-2 and 3-9: unambiguously children
TEEN_GROUP = 2                        # 10-19: ambiguous
INPUT = 112
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_net(pretrained: bool = False):
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    class FaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(0.2)
            self.gender = nn.Linear(576, 2)
            self.age = nn.Linear(576, len(AGE_GROUPS))

        def forward(self, x):
            f = self.drop(self.pool(self.features(x)).flatten(1))
            return self.gender(f), self.age(f)

    return FaceNet()


def to_input(face_bgr: np.ndarray, size: int = INPUT) -> np.ndarray:
    """BGR face crop -> normalised CHW float32 (what the network was trained on)."""
    img = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


def save_checkpoint(net, path: str | Path, meta: dict) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = build_net(pretrained=False)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net, ckpt.get("meta", {})


class TorchAgeGender:
    """Drop-in for ``AgeGenderNet``: ``predict(face_bgr) -> (p_male, p_female, p_child)``."""

    def __init__(self, weights: str | Path, teen_weight: float = 0.0) -> None:
        import torch

        self.net, self.meta = load_checkpoint(weights)
        self.teen_weight = float(teen_weight)
        self._torch = torch
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))     # leave cores for the driver monitor

    def raw_probs(self, faces_bgr: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(gender probabilities (N, 2), age-group probabilities (N, 9))."""
        torch = self._torch
        x = torch.from_numpy(np.stack([to_input(f) for f in faces_bgr]))
        with torch.inference_mode():
            g, a = self.net(x)
        return g.softmax(1).numpy(), a.softmax(1).numpy()

    def predict_batch(self, faces_bgr: Sequence[np.ndarray]) -> List[Tuple[float, float, float]]:
        pg, pa = self.raw_probs(faces_bgr)
        child = pa[:, list(CHILD_GROUPS)].sum(1) + self.teen_weight * pa[:, TEEN_GROUP]
        return [(float(m), float(f), float(min(c, 1.0))) for (m, f), c in zip(pg, child)]

    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float, float]:
        return self.predict_batch([face_bgr])[0]
