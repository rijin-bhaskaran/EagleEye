"""Car-cabin detector and pipeline (``passenger.vehicle_type: car``).

One YOLO trained on ``CAR_CLASSES`` replaces the bus chain (person detector + face classifier + seat model):
a single pass finds the occupants with their male / female / child class, seatbelts and child seats, and
``CabinAnalyzer`` turns that into belt and child-seat verdicts. It returns the same ``PassengerState`` the runner,
hub and dashboard already use, with ``state.car`` filled in.
"""
from __future__ import annotations

import statistics
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

from .car_logic import CAR_CLASSES, PERSON_CLASSES, CabinAnalyzer, CarDetection
from .classifier import LABELS, UNKNOWN
from .pipeline import PassengerState
from .seats import Detection


class CarCabinDetector:
    def __init__(self, weights: str, conf: float = 0.35, imgsz: int = 640, backend: str = "torch") -> None:
        from ultralytics import YOLO

        if not Path(weights).is_file() and not Path(weights).is_dir():
            raise FileNotFoundError(
                f"car model weights not found: {weights}. Train one (docs/CAR_MODEL.md: passenger.datasets.car_cabin, "
                "then passenger.train_car) and point passenger.car.weights at it, or set passenger.vehicle_type: bus.")
        if backend not in ("torch", "openvino"):
            raise ValueError(f"unknown detector backend {backend!r} (torch or openvino)")
        if backend == "openvino" and weights.endswith(".pt"):
            from .detector import openvino_model
            try:
                weights = openvino_model(weights, imgsz)
            except Exception as exc:
                print(f"[passenger] OpenVINO unavailable ({type(exc).__name__}: {exc}); using PyTorch", flush=True)
        self._model = YOLO(weights, task="detect")
        self.conf, self.imgsz = conf, imgsz
        names = getattr(self._model, "names", None) or {}
        known = {str(n) for n in names.values()}
        missing = [c for c in CAR_CLASSES if c not in known]
        if known and len(missing) == len(CAR_CLASSES):
            raise ValueError(f"{weights} has classes {sorted(known)}, none of them a car-cabin class {list(CAR_CLASSES)}")
        if missing:
            print(f"[passenger] car model has no class for {missing}: those checks will stay silent", flush=True)

    def detect(self, frame_bgr: np.ndarray, track: bool = True) -> List[CarDetection]:
        kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        res = (self._model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **kw)
               if track else self._model.predict(frame_bgr, **kw))[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxyn.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(xyxy)
        out = []
        for b, c, k, i in zip(xyxy, confs, cls, ids):
            name = str(res.names[int(k)])
            if name in CAR_CLASSES:
                out.append(CarDetection(name, tuple(map(float, b)), float(c), None if i is None else int(i)))
        return out


class CarPipeline:
    """Same ``process(frame) -> PassengerState`` contract as ``PassengerPipeline``."""

    seatmap = None                     # no seat polygons in a car: the runner only draws them when asked to
    attributes = None
    car = True

    def __init__(self, detector, analyzer: CabinAnalyzer, seats: int = 5, count_window: int = 5) -> None:
        self.detector, self.analyzer, self.seats = detector, analyzer, max(int(seats), 1)
        self._counts: Deque[int] = deque(maxlen=max(count_window, 1))

    def process(self, frame_bgr: np.ndarray) -> PassengerState:
        dets = self.detector.detect(frame_bgr, track=True)
        cabin = self.analyzer.update(dets)
        self._counts.append(len(cabin.occupants))
        count = int(statistics.median(self._counts))
        composition = {"male": 0, "female": 0, "child": 0, "unknown": 0}
        for o in cabin.occupants:
            composition[o.label if o.label in PERSON_CLASSES else "unknown"] += 1
        people = [Detection(o.box, 1.0, o.track_id) for o in cabin.occupants]
        return PassengerState(
            count=count, occupied=[], vacant=[], total_seats=self.seats,
            composition=composition, detections=people,
            track_labels={o.track_id: o.label for o in cabin.occupants if o.track_id is not None},
            seats_seen=(min(count, self.seats), max(self.seats - count, 0)),
            car=cabin,
        )

    @classmethod
    def from_config(cls, cfg) -> "CarPipeline":
        c = cfg.passenger.car
        detector = CarCabinDetector(c.weights, c.conf, c.imgsz, c.get("backend", "torch"))
        a = c.rules
        analyzer = CabinAnalyzer(window=a.window, min_votes=a.min_votes, belt_overlap=a.belt_overlap, belt_on_ratio=a.belt_on_ratio,
                                 child_seat_overlap=a.child_seat_overlap, seat_window=a.seat_window)
        return cls(detector, analyzer, seats=c.seats, count_window=cfg.passenger.occupancy.count_window)


class TwoStageCarPipeline:
    """``passenger.car.pipeline: two_stage``: the person detector finds the occupants, then ``passenger.classifier`` (normally the
    ``crops`` ensemble) says male / female / child for each tracked person. No belt or child-seat checks: those need a cabin model.
    Same ``process(frame) -> PassengerState`` contract as the other pipelines; seats are a gauge against ``passenger.car.seats``."""

    seatmap = None                     # a car has no seat polygons
    classifier_label = "Crop model live"

    def __init__(self, detector, attributes=None, seats: int = 5, count_window: int = 5, stabiliser=None) -> None:
        self.detector, self.attributes, self.seats = detector, attributes, max(int(seats), 1)
        self.stabiliser = stabiliser                    # TrackStabiliser or None: calmer boxes, IDs that survive a lost frame
        self._counts: Deque[int] = deque(maxlen=max(count_window, 1))
        self._shown_labels: Dict[int, str] = {}         # last label of each box on screen, so a held box keeps it

    def process(self, frame_bgr: np.ndarray) -> PassengerState:
        raw = self.detector.detect(frame_bgr, track=True)
        shown, dets = self.stabiliser.update(raw) if self.stabiliser is not None else (raw, raw)
        labels = self.attributes.update(frame_bgr, dets) if self.attributes is not None else {}
        self._counts.append(len(dets))                  # `dets` = people seen this frame; held boxes are only drawn
        count = int(statistics.median(self._counts))
        composition = {name: 0 for name in (*LABELS, UNKNOWN)}
        for d in dets:
            composition[labels.get(d.track_id, UNKNOWN)] += 1
        track_labels = labels
        if self.stabiliser is not None:                 # a held box has no fresh label: show the one it had
            self._shown_labels.update(labels)
            track_labels = {d.track_id: self._shown_labels[d.track_id] for d in shown if d.track_id in self._shown_labels}
            track_labels.update(labels)
            self._shown_labels = dict(track_labels)
        return PassengerState(count=count, occupied=[], vacant=[], total_seats=self.seats, composition=composition,
                              detections=shown, track_labels=track_labels,
                              seats_seen=(min(count, self.seats), max(self.seats - count, 0)))

    @classmethod
    def from_config(cls, cfg, with_classifier: bool = True) -> "TwoStageCarPipeline":
        from .classifier import AttributeService, BackgroundAttributes, make_classifier
        from .detector import PersonDetector

        det = cfg.passenger.detector
        detector = PersonDetector(det.model, det.conf, det.imgsz, det.classes, backend=det.get("backend", "torch"),
                                  tracker=det.get("tracker"), iou=det.get("iou"))
        attributes = None
        if with_classifier:
            c = cfg.passenger.classifier
            attributes = AttributeService(make_classifier(cfg), window=c.window, min_votes=c.min_votes, budget=c.budget_per_frame)
            if c.get("background", True):
                attributes = BackgroundAttributes(attributes)      # classifier reads off the frame loop
        stabiliser = None
        st = cfg.passenger.car.get("stability")
        if st and st.get("enabled", True):
            from .stability import TrackStabiliser
            try:
                stabiliser = TrackStabiliser(**{k: v for k, v in st.items() if k != "enabled"})
            except TypeError as exc:
                raise ValueError(f"passenger.car.stability: {exc}") from exc
        return cls(detector, attributes, seats=cfg.passenger.car.seats, count_window=cfg.passenger.occupancy.count_window,
                   stabiliser=stabiliser)
