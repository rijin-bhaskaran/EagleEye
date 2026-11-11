"""Male / Female / Child classification of detected passengers.

Stages for one person: crop -> find a face -> age + gender CNNs -> label.
The answer is ``unknown`` whenever the evidence is weak (person too small, no
usable face, low confidence). A wrong label on the dashboard is worse than a
missing one, and the back rows of a bus are often too far away to read.

``AttributeService`` then smooths labels per tracked person, and spends only a
fixed budget of classifier calls per frame so the 10-12 FPS edge target holds.

Model note: the default age/gender weights are the Levi & Hassner (2015)
Adience CNNs, which are released for research use. Check the licence before
shipping them in a commercial product, and prefer fine-tuning on your own data.
Any object with ``predict(face_bgr) -> (p_male, p_female, p_child)`` can replace
``AgeGenderNet``.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from .seats import Box, Detection

LABELS = ("male", "female", "child")
UNKNOWN = "unknown"

_MEAN = (78.4263377603, 87.7689143744, 114.895847746)      # BGR mean used when the nets were trained
_CHILD_BUCKETS = 3                                          # (0-2) (4-6) (8-12) of the 8 age buckets


@dataclass(frozen=True)
class Attribute:
    label: str
    conf: float


class AttributeModel(Protocol):
    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float, float]: ...


FaceFinder = Callable[[np.ndarray], Optional[Tuple[int, int, int, int]]]     # crop -> (x, y, w, h) px or None


class AgeGenderNet:
    """Levi-Hassner age + gender CNNs run through OpenCV's DNN module (no extra dependency)."""

    def __init__(self, model_dir: str | Path = "models/passenger") -> None:
        d = Path(model_dir)
        self._gender = cv2.dnn.readNetFromCaffe(str(d / "gender_deploy.prototxt"), str(d / "gender_net.caffemodel"))
        self._age = cv2.dnn.readNetFromCaffe(str(d / "age_deploy.prototxt"), str(d / "age_net.caffemodel"))

    def predict(self, face_bgr: np.ndarray) -> Tuple[float, float, float]:
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), _MEAN, swapRB=False)
        self._gender.setInput(blob)
        p_gender = self._gender.forward().ravel()           # [male, female]
        self._age.setInput(blob)
        p_age = self._age.forward().ravel()                 # 8 buckets, youngest first
        return float(p_gender[0]), float(p_gender[1]), float(p_age[:_CHILD_BUCKETS].sum())


class MediaPipeFaceFinder:
    """Largest face inside a person crop. ``model_selection=1`` is the long-range (about 5 m) model."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        import mediapipe as mp                              # lazy: heavy import, optional in tests

        self._det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence)

    def __call__(self, crop_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = crop_bgr.shape[:2]
        res = self._det.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return None
        boxes = [d.location_data.relative_bounding_box for d in res.detections]
        b = max(boxes, key=lambda r: r.width * r.height)
        return int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)


class YuNetFaceFinder:
    """Largest face inside a person crop, using OpenCV's YuNet detector (no extra dependency).

    On the stock bus clips it found a face on 85 of 106 people where MediaPipe found 55, and the extra ones
    were real (profile views, blurred faces); see ``scripts/diagnose_faces.py``."""

    def __init__(self, weights: str | Path, score: float = 0.6) -> None:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"{weights} not found: run python scripts/fetch_passenger_models.py")
        self._det = cv2.FaceDetectorYN.create(str(weights), "", (320, 320), score, 0.3, 5000)

    def __call__(self, crop_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(crop_bgr)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])[:4]
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + fw), w), min(int(y + fh), h)
        return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def make_face_finder(c) -> FaceFinder:
    """The face finder named by ``passenger.classifier.face_detector`` (mediapipe | yunet).

    A missing YuNet file falls back to MediaPipe with a printed warning rather than losing classification."""
    if str(c.get("face_detector", "mediapipe")).lower() == "yunet":
        try:
            return YuNetFaceFinder(c.get("yunet_weights", "models/passenger/face_detection_yunet_2023mar.onnx"),
                                   c.get("yunet_score", 0.6))
        except FileNotFoundError as exc:
            print(f"[passenger] {exc}; falling back to MediaPipe", flush=True)
    return MediaPipeFaceFinder(c.face_confidence)


class PassengerClassifier:
    def __init__(self, model: AttributeModel, find_face: FaceFinder, *, min_person_px: int = 60,
                 min_face_px: int = 24, child_threshold: float = 0.5, min_conf: float = 0.6,
                 face_margin: float = 0.4) -> None:
        self.model, self.find_face = model, find_face
        self.min_person_px, self.min_face_px = min_person_px, min_face_px
        self.child_threshold, self.min_conf, self.face_margin = child_threshold, min_conf, face_margin

    def face_crop(self, frame_bgr: np.ndarray, box: Box) -> Optional[np.ndarray]:
        """The face image the model sees for this person (None if too small or no usable face)."""
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(int(box[0] * W), 0), max(int(box[1] * H), 0)
        x2, y2 = min(int(np.ceil(box[2] * W)), W), min(int(np.ceil(box[3] * H)), H)
        if y2 - y1 < self.min_person_px or x2 <= x1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]

        face = self.find_face(crop)
        if face is None or min(face[2], face[3]) < self.min_face_px:
            return None
        fx, fy, fw, fh = face
        m = int(self.face_margin * max(fw, fh))             # the nets were trained on loosely cropped faces
        ch, cw = crop.shape[:2]
        face_img = crop[max(fy - m, 0):min(fy + fh + m, ch), max(fx - m, 0):min(fx + fw + m, cw)]
        return face_img if face_img.size else None

    def classify(self, frame_bgr: np.ndarray, box: Box) -> Attribute:
        """``box`` is normalised (x1, y1, x2, y2)."""
        face_img = self.face_crop(frame_bgr, box)
        if face_img is None:
            return Attribute(UNKNOWN, 0.0)

        p_male, p_female, p_child = self.model.predict(face_img)
        if p_child >= self.child_threshold:
            return Attribute("child", p_child)
        label, conf = ("male", p_male) if p_male >= p_female else ("female", p_female)
        return Attribute(label, conf) if conf >= self.min_conf else Attribute(UNKNOWN, conf)

    @classmethod
    def from_config(cls, cfg) -> "PassengerClassifier":
        c = cfg.passenger.classifier
        name = str(c.get("model", "levi_hassner")).strip().lower()
        if name == "crops":
            raise ValueError("passenger.classifier.model is 'crops', which reads a person crop, not a face: build it with "
                             "passenger.classifier.make_classifier(cfg)")
        if name not in ("torch", "levi_hassner"):
            raise ValueError(f"passenger.classifier.model {name!r} is not one of torch, levi_hassner, crops")
        if name == "torch":                                     # the model trained on FairFace (passenger.train_faces)
            from .face_model import TorchAgeGender
            if not Path(c["weights"]).is_file():
                raise FileNotFoundError(f"{c['weights']} not found. Train it (python -m passenger.train_faces) or set "
                                        "passenger.classifier.model: levi_hassner (and child_threshold 0.95, min_conf 0.6)")
            model = TorchAgeGender(c["weights"], c.get("teen_weight", 0.0))
        else:                                                   # the pretrained Levi-Hassner nets
            model = AgeGenderNet(c.model_dir)
        return cls(model, make_face_finder(c),
                   min_person_px=c.min_person_px, min_face_px=c.min_face_px,
                   child_threshold=c.child_threshold, min_conf=c.min_conf,
                   face_margin=c.get("face_margin", 0.4))


def make_classifier(cfg):
    """The attribute classifier named by ``passenger.classifier.model``: ``crops`` (person-crop ensemble) or a face-based one."""
    if str(cfg.passenger.classifier.get("model", "levi_hassner")).strip().lower() == "crops":
        from .crop_model import CropAttributeClassifier
        return CropAttributeClassifier.from_config(cfg)
    return PassengerClassifier.from_config(cfg)


class AttributeService:
    """Per-track smoothing with a bounded number of classifier calls per frame."""

    def __init__(self, classifier: PassengerClassifier, *, window: int = 15, min_votes: int = 3,
                 budget: int = 3, stale_after: int = 300, retry_after: int = 3) -> None:
        self._clf = classifier
        self._window, self._min_votes, self._budget, self._stale_after = window, min_votes, budget, stale_after
        self._retry_after = retry_after                # frames to leave a person alone after a failed face read
        self._votes: Dict[int, Deque[Attribute]] = defaultdict(lambda: deque(maxlen=window))
        self._seen_at: Dict[int, int] = {}
        self._tried_at: Dict[int, int] = {}
        self._next_try: Dict[int, int] = {}
        self._tick = 0

    def update(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> Dict[int, str]:
        """Return track id -> smoothed label for every tracked detection in this frame."""
        self._tick += 1
        tracked = [d for d in detections if d.track_id is not None]
        for d in tracked:
            self._seen_at[d.track_id] = self._tick

        # Unlabelled people first, then settled ones; within each group, whoever was tried longest ago.
        # Rotating matters: a passenger whose face never reads (back row) keeps zero votes, and ordering by
        # vote count alone would let a few of them use the whole budget every frame and starve everyone else.
        def priority(d):
            tid = d.track_id
            return (len(self._votes[tid]) >= self._min_votes, self._tried_at.get(tid, -1))

        # A person facing away fails every time (a face read costs ~30 ms on CPU), so after a miss they are
        # skipped for a few frames and the budget goes to people who might actually show a face.
        due = [d for d in tracked if self._next_try.get(d.track_id, 0) <= self._tick]
        for d in sorted(due, key=priority)[: self._budget]:
            self._tried_at[d.track_id] = self._tick
            attr = self._clf.classify(frame_bgr, d.box)
            if attr.label != UNKNOWN:
                self._votes[d.track_id].append(attr)
            else:
                self._next_try[d.track_id] = self._tick + self._retry_after

        self._drop_stale()
        return {d.track_id: self.label(d.track_id) for d in tracked}

    def label(self, track_id: int) -> str:
        votes = self._votes.get(track_id)
        if not votes or len(votes) < self._min_votes:
            return UNKNOWN
        weight: Dict[str, float] = defaultdict(float)
        for v in votes:
            weight[v.label] += v.conf
        return max(weight, key=weight.get)

    def _drop_stale(self) -> None:
        for tid in [t for t, s in self._seen_at.items() if self._tick - s > self._stale_after]:
            self._votes.pop(tid, None)
            self._tried_at.pop(tid, None)
            self._next_try.pop(tid, None)
            del self._seen_at[tid]


class BackgroundAttributes:
    """Runs an ``AttributeService`` on its own thread so face reads never hold up the frame loop.

    ``update`` has the same signature as the service's: it hands the newest frame to the worker
    (older unprocessed ones are dropped, never queued) and returns the latest labels immediately.
    Labels therefore trail the video by one classifier pass, which is invisible for a person
    who stays on screen for seconds.
    """

    def __init__(self, service: AttributeService) -> None:
        import threading

        self._svc = service
        self._cond = threading.Condition()
        self._job = None
        self._labels: Dict[int, str] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="passenger-attributes", daemon=True)
        self._thread.start()

    def update(self, frame_bgr: np.ndarray, detections: Sequence[Detection]) -> Dict[int, str]:
        with self._cond:
            self._job = (frame_bgr.copy(), list(detections))
            self._cond.notify()
            return {d.track_id: self._labels.get(d.track_id, UNKNOWN)
                    for d in detections if d.track_id is not None}

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                frame, dets = self._job
                self._job = None
            try:
                labels = self._svc.update(frame, dets)
            except Exception as exc:                   # a bad face read must not kill labelling
                print(f"[passenger] classifier error: {type(exc).__name__}: {exc}", flush=True)
                continue
            with self._cond:
                self._labels = labels

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()
