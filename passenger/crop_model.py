"""Male / female / child from the upper body of a person box (``passenger.classifier.model: crops``).

Several small networks (``passenger.train_crops``, MobileNetV3-small, 112 px) each read the same crop; their probabilities are
averaged and a fixed offset is added to each class's log-probability before the decision. The offsets correct the bias the
training data leaves behind (few women, many children) and were chosen on validation data: see
``models/passenger/crops/v4_blend.json`` and ``passenger_data/agent_reports/10_metrics_and_pipeline_recommendation.md``.

It has the same ``classify(frame, box) -> Attribute`` contract as ``PassengerClassifier``, so ``AttributeService`` smooths it per
tracked person exactly like the face-based classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .classifier import LABELS, UNKNOWN, Attribute
from .face_model import INPUT, to_input
from .seats import Box

TOP_FRACTION = 0.6                      # the crop is the top 60% of the person box: head and torso, as in the training crops


def person_crop(frame_bgr: np.ndarray, box: Box, top: float = TOP_FRACTION, size: int = INPUT) -> Optional[np.ndarray]:
    """``box`` is normalised (x1, y1, x2, y2). The top ``top`` of it, padded to a square by repeating the edge, ``size`` px.
    The same rule the training crops were made with; ``None`` for an empty box."""
    H, W = frame_bgr.shape[:2]
    x0, x1 = max(int(box[0] * W), 0), min(int(np.ceil(box[2] * W)), W)
    y0, y1 = max(int(box[1] * H), 0), min(int(np.ceil((box[1] + top * (box[3] - box[1])) * H)), H)
    if x1 <= x0 or y1 <= y0:
        return None
    c = frame_bgr[y0:y1, x0:x1]
    h, w = c.shape[:2]
    s = max(h, w)
    c = cv2.copyMakeBorder(c, (s - h) // 2, s - h - (s - h) // 2, (s - w) // 2, s - w - (s - w) // 2, cv2.BORDER_REPLICATE)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def _build_net(num_classes: int = 3):
    """The network ``passenger.train_crops`` trains and saves (kept here so running the monitor does not import the trainer)."""
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    base = mobilenet_v3_small(weights=None)
    return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(576, num_classes))


def load_crop_nets(paths: Sequence[str | Path]):
    import torch

    nets = []
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"crop classifier weights not found: {path}. Train them (python -m passenger.train_crops, docs/CAR_MODEL.md) "
                "and point passenger.classifier.crops.weights at the model.pt files, or use a face-based passenger.classifier.model.")
        net = _build_net(3)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
        nets.append(net.eval())
    return nets


class CropAttributeClassifier:
    def __init__(self, nets, log_offsets: Sequence[float] = (0.0, 0.0, 0.0), *, min_prob: Optional[float] = None,
                 min_person_px: int = 60, top: float = TOP_FRACTION) -> None:
        if not nets:
            raise ValueError("CropAttributeClassifier needs at least one network")
        if len(log_offsets) != len(LABELS):
            raise ValueError(f"log_offsets needs one value per class {LABELS}, got {len(log_offsets)}")
        self.nets, self.log_offsets = list(nets), tuple(float(v) for v in log_offsets)
        self.min_prob, self.min_person_px, self.top = min_prob, min_person_px, top

    def warm_up(self) -> None:
        """One throwaway pass: the first run through the networks takes far longer than the rest, and until it finishes every
        person on screen is ``unknown``. Doing it at start-up (before the video plays) keeps those seconds out of the dashboard."""
        self.classify(np.zeros((128, 96, 3), np.uint8), (0.0, 0.0, 1.0, 1.0))

    def classify(self, frame_bgr: np.ndarray, box: Box) -> Attribute:
        """``box`` is normalised (x1, y1, x2, y2). ``unknown`` for a person too small to read, or below ``min_prob``."""
        import torch

        if (box[3] - box[1]) * frame_bgr.shape[0] < self.min_person_px:
            return Attribute(UNKNOWN, 0.0)
        crop = person_crop(frame_bgr, box, self.top)
        if crop is None:
            return Attribute(UNKNOWN, 0.0)
        x = torch.from_numpy(to_input(crop)[None])
        with torch.inference_mode():
            p = torch.stack([net(x).softmax(1) for net in self.nets]).mean(0)
            prob = (torch.log(p + 1e-6) + torch.tensor(self.log_offsets)).softmax(1)[0]
        k = int(prob.argmax())
        conf = float(prob[k])
        if self.min_prob is not None and conf < self.min_prob:
            return Attribute(UNKNOWN, conf)
        return Attribute(LABELS[k], conf)

    @classmethod
    def from_config(cls, cfg) -> "CropAttributeClassifier":
        c = cfg.passenger.classifier
        crops = c.get("crops") or {}
        weights = crops.get("weights") or []
        if not weights:
            raise ValueError("passenger.classifier.model is 'crops' but passenger.classifier.crops.weights is empty")
        offsets = crops.get("log_offsets")
        if offsets is None:
            offsets = {name: 0.0 for name in LABELS}
        if set(offsets) != set(LABELS):
            raise ValueError(f"passenger.classifier.crops.log_offsets must name exactly {list(LABELS)}, got {sorted(offsets)}")
        clf = cls(load_crop_nets(weights), tuple(offsets[name] for name in LABELS), min_prob=crops.get("min_prob"),
                  min_person_px=int(c.get("min_person_px", 60)))
        clf.warm_up()
        return clf
