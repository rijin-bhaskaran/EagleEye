"""``passenger.vehicle_type`` in config.yaml picks the whole perception chain."""
from __future__ import annotations

VEHICLE_TYPES = ("bus", "car")


def vehicle_type(cfg) -> str:
    v = str(cfg.passenger.get("vehicle_type", "bus")).strip().lower()
    if v not in VEHICLE_TYPES:
        raise ValueError(f"passenger.vehicle_type must be one of {VEHICLE_TYPES}, got {v!r}")
    return v


def build_pipeline(cfg, with_classifier: bool = True):
    """The pipeline for the configured vehicle. ``bus``: person detector + face classifier + seat model.
    ``car``: ``passenger.car.pipeline`` picks ``two_stage`` (person detector + crop classifier) or ``cabin_model`` (one multi-class
    model: occupants, seatbelts, child seats; ``with_classifier`` does not apply to it)."""
    if vehicle_type(cfg) == "car":
        kind = str(cfg.passenger.car.get("pipeline", "cabin_model")).strip().lower()
        if kind == "two_stage":
            from .car_pipeline import TwoStageCarPipeline
            return TwoStageCarPipeline.from_config(cfg, with_classifier=with_classifier)
        if kind != "cabin_model":
            raise ValueError(f"passenger.car.pipeline must be 'two_stage' or 'cabin_model', got {kind!r}")
        from .car_pipeline import CarPipeline
        return CarPipeline.from_config(cfg)
    from .detector import PersonDetector
    from .pipeline import PassengerPipeline
    det = cfg.passenger.detector
    detector = PersonDetector(det.model, det.conf, det.imgsz, det.classes, backend=det.get("backend", "torch"))
    return PassengerPipeline.from_config(cfg, detector, with_classifier=with_classifier)
