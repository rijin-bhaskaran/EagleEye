# EagleEye Architecture

Living document. Updated at the end of every implementation phase (see [WORKFLOW.md](WORKFLOW.md)).

## 1. System overview

EagleEye has two independent perception systems that share one dashboard.

| System | Camera | Target FPS | Process | Entry point |
|---|---|---|---|---|
| DMS (driver monitoring) | Driver-facing | ~30 capture, inference per frame | `run.py` | `python run.py` |
| Passenger monitoring | Cabin-facing, sees to the last row | 10-12 target (`passenger.camera.fps`), paced | separate process | `python run_passenger.py` (or `python run_all.py` for both) |

The passenger system is a **separate process** so a heavy YOLO pass can never slow the driver loop
(the DMS already suffers a 30 -> 15 FPS drop when two copies compete for CPU; see `_acquire_single_instance` in `run.py`).

```
 driver cam ──> run.py ──> EagleEyePipeline ──> TelemetryHub ──> WebUI :8000 ──┬─ /video, /api/telemetry   ──> browser
                                                                  (proxy)      ├─ /video/passenger  ─┐
 cabin cam ──> run_passenger.py ──> PassengerPipeline ──> PassengerHub         └─ /api/passenger    ─┤
                (own process)                              └─> PassengerServer 127.0.0.1:8100  <────┘
                                                                  /video  /api/passenger  /healthz
 run_all.py: starts both, stops both when the driver monitor stops
```

## 2. Existing DMS (baseline, unchanged by passenger work)

| Module | Responsibility |
|---|---|
| `run.py` | Main loop, `LiveRenderer` thread (camera-rate video), command handling |
| `eagleeye/pipeline.py` | Per-frame perception: landmarks, metrics, objects, fusion |
| `eagleeye/fusion.py` | Rule / temporal / hybrid driver state |
| `eagleeye/config.py` + `config.yaml` | Defaults deep-merged with user YAML |
| `eagleeye/logger.py` | CSV frame log + SQLite event store (`logs/eagleeye.db`) |
| `eagleeye/webui/hub.py` | `TelemetryHub`: latest frame (condition-variable pub/sub), telemetry payload, command queue |
| `eagleeye/webui/server.py` | Stdlib `ThreadingHTTPServer` |
| `eagleeye/webui/index.html` | Single-page dashboard; polls `/api/telemetry` |

### Current HTTP API (`server.py`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/`, `/index.html` | Dashboard page |
| GET | `/assets/<file>` | Static images (`.png .svg .ico`), path-traversal guarded |
| GET | `/video` | MJPEG stream of the annotated driver feed |
| GET | `/api/telemetry` | Element-id -> value payload |
| GET | `/api/fleet` | Last 40 sessions from SQLite |
| GET | `/healthz` | Liveness |
| POST | `/api/command` | `recalibrate / mute / unmute / snapshot / stop` |
| GET | `/api/passenger` | Proxy of the passenger payload; `{"online": false}` when the monitor is down |
| GET | `/video/passenger` | Proxy of the passenger MJPEG stream; 503 when the monitor is down |

Passenger server (`passenger/server.py`, loopback only, read-only): `GET /video`, `GET /api/passenger` (payload + `age_s`), `GET /healthz`.
The dashboard treats the monitor as live only if the payload exists and `age_s <= 3`, otherwise it shows OFFLINE / STARTING / STALLED.

Process rules (`run_all.py`): driver exit stops everything; passenger exit leaves the driver monitor running; both webcams may not share an index.

## 3. Passenger system (planned)

Status per component is tracked in the table in [WORKFLOW.md](WORKFLOW.md).

```
passenger/                      code (Phases 1-5)
  data_manager.py               DataManager (+ CLI): the only reader/writer of passenger_data/  [done]
  datasets/open_images.py       Open Images -> male/female/child boxes (raw kept, labels derived) [done]
  registry.py                   model registry + promotion gate (no regression on same benchmark) [done]
  train.py                      retrain from the champion on all data, register candidate         [done]
  face_model.py                 MobileNetV3-small gender + age-group net, TorchAgeGender          [done]
  train_faces.py / face_eval.py / tune_classifier.py   train on FairFace, score, tune thresholds  [done]
  datasets/fairface.py          FairFace ingestion (raw kept, checksummed)                        [done]
  detector.py                   PersonDetector: YOLO person + ByteTrack                  [done]
  seats.py                      SeatMap (polygon masks, assignment), placeholder grid    [done]
  pipeline.py                   PassengerPipeline, SeatDebouncer, PassengerState         [done]
  metrics.py                    pure-function evaluation metrics                         [done]
  evaluate.py                   scores the model on labelled frames (CLI)                [done]
  runner.py                     PassengerRunner: paced read -> process -> draw -> publish [done]
  hub.py                        PassengerHub + dashboard payload / seat-map HTML          [done]
  overlay.py                    seat outlines + labelled person boxes                     [done]
  server.py                     loopback MJPEG + telemetry server                         [done]
  source.py                     resolve_source: passenger input = live camera or video file (config / --source)  [done]
run_passenger.py                passenger process entry point
scripts/make_demo_video.py      builds a test video for the passenger channel from photos in passenger_data/
run_all.py                      launcher for both processes
  classifier.py                 face -> age/gender CNN -> male/female/child/unknown;     [done]
                                AttributeService: per-track smoothing + call budget
scripts/fetch_passenger_models.py   downloads + SHA-256-verifies models/passenger/*
passenger_data/                 single entry point for all passenger data (see docs/DATASETS.md)
  raw/                          ORIGINALS, immutable, checksummed (`verify_raw`); never deleted by any code
    videos/<vehicle>/<id>.mp4     own cabin footage; file name = content hash id
    external/<dataset_id>/        open datasets as downloaded: images/, metadata/ (original CSVs, attribution), SOURCE.json
  processed/frames/<video_id>/  DERIVED: frames extracted from videos, named <video_id>_<frame>.jpg
  annotations/
    labels/                     YOUR hand-made YOLO labels (CVAT export lands here); precious
    external/<dataset_id>/labels/  DERIVED: labels converted from an open dataset
    <export_name>/              DERIVED: training set built by export_yolo (images/, labels/, data.yaml)
  manifest.json                 index: videos, external datasets, frames, splits, checksums, provenance
models/passenger/
  registry.json                 model versions: lineage, dataset fingerprints, benchmark scores, champion
```

Data flow (own footage): `register` -> `extract` -> `splits` -> annotate in CVAT (docs/ANNOTATION.md) -> `import-labels` -> `validate` -> `export`.
Data flow (open data): `passenger.datasets.open_images` downloads into `raw/external/`, converts labels, registers with provenance.
Both feed one view, `DataManager.iter_frames(split, source)`, which export, evaluation, training and the QA gate all use.
Classes: `male`, `female`, `child` (ids 0-2). Own videos are split as whole videos and keep their split forever; external images get a stable
hash-based split. The test split is never trained on.

Retraining (`passenger.train` + `passenger.registry`): export all data -> train from the champion -> score the candidate on the frozen test split
(registry stores the test-split fingerprint and refuses to compare scores from different test sets) -> `promote` only if recall, precision, AP,
count error and classification F1 did not regress beyond tolerance. `passenger.detector.classes` is `[0]` for the COCO model and `null` for a
model fine-tuned on male/female/child. Edge/ONNX/accelerator route: docs/EDGE_DEPLOYMENT.md (a plan, not yet built).

Per-frame pipeline: **detect** (YOLO `person`) -> **track** (stable IDs) ->
**seat occupancy** (person box vs. seat polygons) -> **classify** crop (Male / Female / Child, smoothed per track).

Data integrity rules (`DataManager`): dataset and vehicle names must be a single safe path segment (`labels` is reserved), because
dataset directories are deleted and rebuilt; a video keeps its split forever once assigned (new videos fill the split furthest below its
target share), so earlier evaluation results stay valid when footage is added.

Classifier scheduling (`AttributeService`): unlabelled people first, then settled ones; within each group, least recently tried. This
rotation prevents people with unreadable faces from using the whole per-frame budget.

Runtime resilience (`PassengerRunner`): a failing frame is logged and skipped; 30 consecutive failures stop the process with the error.

Seat occupancy rules: a person is scored against every seat polygon by (overlap area / person box area);
matches are one-to-one, best first, and need `min_overlap` (0.30). A seat turns occupied after `on_frames` consecutive
hits and vacant after `off_frames` misses; the count is a median over `count_window` frames.

Evaluation (`passenger/metrics.py`, `passenger/evaluate.py`): detection P/R/F1, AP50, mAP50-95 (class-agnostic);
count MAE / exact / bias; seat occupancy micro P/R/F1, macro-F1, per-seat F1, frame-exact; latency mean / p95 / max FPS.
Ground-truth occupancy is derived from labelled boxes through the same `SeatMap`.
Classification (`classification_metrics`): confusion matrix, per-class P/R/F1, macro-F1, coverage, accuracy on answered / overall;
abstentions count as misses for recall. Tracking-stability (label flips) arrives with the Phase 4 video run.

Classification: `unknown` is a first-class answer. A person is classified only if their box is >= 60 px tall and a face >= 24 px is found;
child is decided from the age net first (P(age <= 12) >= 0.5), then gender if confidence >= 0.6. Labels are voted per tracked person
(15-sample window, 3 votes to show) and at most 3 classifier calls run per frame.

Decisions on record:
- YOLOv8/11 pretrained, COCO `person` class; no retraining needed for detection.
- Gender/age: pretrained model (stopgap). Per-track majority vote over a window to suppress flicker.
- Dataset splits are by **video**, not by frame, to avoid train/val leakage.
- Annotation tool: CVAT (video interpolation, YOLO export).
- Seat polygons are config, per camera/vehicle layout.

## 4. Change log

See [WORKFLOW.md](WORKFLOW.md).

## Seat-state model (Phase 10)

`passenger/seat_state.py` runs a YOLO model trained on the busCabin dataset (classes Occupied / Unoccupied) every `seat_state.every` frames and keeps the last result in between. Occupied and vacant counts are the median of the last `window` reads. When enabled, `PassengerState.seats_seen = (occupied, vacant)` replaces the polygon-grid numbers in the payload (`paxOccupied`, `paxVacant`, `paxTotal`) and in the video HUD; the grid is the fallback when the weights are missing or the block is disabled. The passenger count is the median number of tracked people over `count_window` frames. Weights: `models/seat_state/v2b/run/weights/best.pt` (see WORKFLOW.md for scores and limits).

The dashboard seat map (`model_seat_grid_html`) stays the fixed configured grid. The seat-state model's rows are laid onto the grid rows (nearest first, left/right of the aisle) and mark the matching slots occupied, coloured by the person inside the box; unseen slots stay idle. The video overlay shows only person boxes and labels; counts live on the dashboard.

### Measured seat zones (Phase 10d)
With `passenger.seats.polygons_file` (or `polygons`) set, `SeatMap.calibrated` is true and the pipeline builds `SeatZones`. A seat is occupied when a seat-state "Occupied" box has its centre inside the seat polygon, debounced per slot in model reads (`on_reads` / `off_reads`). `PassengerState.occupied/vacant/total_seats` then describe those fixed seats and the dashboard shows the ordinary id grid. Without measured seats the pipeline falls back to the rank-based mapping of the fixed placeholder grid. Zones are drawn with `scripts/draw_seat_zones.py` on a fixed camera.

### Placeholder seat map without measured seats (Phase 10e)
Without `seats.polygons_file`, the model's seat counts go through `StableCount` (median of `count_reads` reads, changes only after `hold_reads` consistent reads, updated once per model read). `model_seat_grid_html` marks the first N slots of the fixed grid in order (front row first, left to right), coloured by the people labels in turn. It is a gauge of how many seats are taken, not which; measured zones (above) show which.

## Vehicle type: car (Phase 13)

`passenger.vehicle_type` (`bus` default | `car`) selects the perception chain in `passenger/vehicle.py::build_pipeline`; `run_passenger.py` calls it and nothing else changes (runner, hub, server, dashboard proxy are shared).

```
car:  frame -> CarCabinDetector (one YOLO + ByteTrack; classes male/female/child/seatbelt/no_seatbelt/child_seat)
           -> CabinAnalyzer (car_logic.py: per-track class vote, belt-to-person assignment, child-in-child-seat, alerts)
           -> PassengerState(car=CabinStatus) -> hub payload (+ "car" block, status HTML in paxSeatGrid) -> overlay
```

* `passenger/car_logic.py` pure rules, `car_pipeline.py` detector + pipeline, `datasets/car_cabin.py` remaps public YOLO datasets onto the six classes (group-level splits, licence required, provenance in `car_data/sources.json`), `train_car.py` trains and scores (per-class + child/child-seat/belt checks).
* The car model has no seat polygons; occupied/vacant are count vs `passenger.car.seats`. Weights are not in the repo; with none, `car` refuses to start.

### Vehicle profiles and the two-stage car chain

`passenger.vehicles.<bus|car>` holds each vehicle's weights, static video and any other differing setting. `eagleeye/config.py::load_config` deep-merges the profile named by `passenger.vehicle_type` over the shared `passenger` settings once, at load, so every consumer (`run_passenger.py`, `run_all.py`, the dashboard process, tests) reads the resolved values with no code change. A vehicle without a profile keeps the shared settings; a config with no `vehicles` block behaves exactly as before.

`passenger.car.pipeline: two_stage` (default in the shipped car profile; `cabin_model` stays the default for configs that do not name one):

```
car/two_stage:  frame -> PersonDetector (COCO YOLO + ByteTrack, the bus detector)
                      -> AttributeService(CropAttributeClassifier)  # top 60% of the person box -> 112 px -> N MobileNetV3-small -> mean probs
                         + log offsets (male 0, female +2.0, child +0.5) -> male / female / child, smoothed per track (background thread)
                      -> PassengerState(count, composition, seats_seen = count vs passenger.car.seats, car=None) -> hub -> overlay
```

`TrackStabiliser` (passenger/stability.py) sits between the detector and the classifier in `TwoStageCarPipeline` when `passenger.car.stability` is set: it returns `(shown, real)`. `real` (smoothed boxes, stable IDs) is counted and classified; `shown` adds boxes held for `hold_frames` after a loss and is what the overlay draws (a held box keeps its last label). A tracker ID never seen before that lands on a recently lost person (IoU >= `id_iou`, within `id_memory` frames) takes that person's stable ID so `AttributeService` keeps their votes. `PersonDetector` takes `tracker` (ByteTrack overrides written to a content-named file in the temp folder) and `iou` from `passenger.detector`.

`PassengerClassifier.from_config` builds only face models (`torch`, `levi_hassner`) and raises for anything else; build through `make_classifier`. The QA gate's benchmark is always the bus profile (`load_config(..., vehicle_type="bus")`), so its history does not change with `vehicle_type`.

`passenger.classifier.model: crops` selects the classifier through `classifier.make_classifier`, wherever a classifier is built. The pipeline names its model for the dashboard (`TwoStageCarPipeline.classifier_label`, shown as "Crop model live"). The crop rule (`crop_model.person_crop`) must stay identical to the one that built `crops_mfc_v4`; `tests/test_crop_classifier.py` pins it and checks that a `train_crops` checkpoint loads into the runtime network.
* Payload additions (car only): `car.{childPresent, childSeats, childrenUnseated, unbelted, alerts, occupants[]}`; `paxClassifier` reads "Cabin model live". The dashboard HTML is unchanged: the existing seat-map slot shows the per-occupant belt list and alerts.
* Guide and data sources: [docs/CAR_MODEL.md](docs/CAR_MODEL.md).
