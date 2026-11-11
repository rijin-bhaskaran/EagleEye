# Car cabin model (`passenger.vehicle_type: car`)

One YOLO model replaces the bus chain (person detector + face classifier + seat model). In one pass it finds:

| Class | Meaning |
|---|---|
| `male`, `female`, `child` | the occupant |
| `seatbelt` | a belt worn across an occupant |
| `no_seatbelt` | an occupant clearly without a belt (optional: only if your data labels it) |
| `child_seat` | an installed child seat, occupied or empty |

`passenger/car_logic.py` turns the boxes into verdicts (rules are in its docstring and tuned under `passenger.car.rules`):
belted / not belted / unknown per occupant; for a child, whether they sit in a child seat; how many child seats are installed;
alerts for *child on board but no child seat installed*, *child is not in a child seat*, *occupant without seatbelt*.

## Switch: bus or car

`passenger.vehicle_type` is the one switch. Each vehicle has a profile under `passenger.vehicles`; whatever a profile lists
replaces the shared setting of the same name (setting by setting), so the switch alone picks the **weights** and the **static video**
the dashboard plays:

```yaml
passenger:
  vehicle_type: car                          # bus | car   (restart run_passenger.py / run_all.py after changing it)
  vehicles:
    bus: {camera: {video_path: data/bus/vid-9.mp4}, detector: {...}, classifier: {model: torch, weights: ...}, seat_state: {enabled: true, weights: ...}}
    car: {camera: {video_path: data/car/car-vid-6.mp4}, detector: {...}, classifier: {model: crops, crops: {weights: [...], log_offsets: {...}}},
          seat_state: {enabled: false}, car: {pipeline: two_stage, seats: 5}}
```

Two car chains exist; `passenger.car.pipeline` picks one:

| `car.pipeline` | What runs | Status |
|---|---|---|
| `two_stage` (the shipped car profile) | person detector (COCO YOLOv8n) -> `crops` classifier (male / female / child from the upper body, 3 networks averaged, per-class log offsets) -> per-person smoothing | Works today. No belt or child-seat checks. |
| `cabin_model` (default when a config names no pipeline) | the single 6-class YOLO described above | Needs weights nobody has trained yet; refuses to start without them |

The crop classifier, its offsets and its measured accuracy are in `passenger_data/agent_reports/10_metrics_and_pipeline_recommendation.md`
(test macro-F1 0.76; about 20 test people, so treat it as directional). Nothing here is a substitute for a test set from the real camera.
With `cabin_model` and no weights file, `run_passenger.py` exits at once and says what to do. `bus` is unchanged.

### Calming the car feed (settings under `vehicles.car`)

Measured on the labelled car videos with the real background classifier thread (WORKFLOW.md, 2026-10-29 "flicker" entry):

| Setting | Does | Measured effect |
|---|---|---|
| `detector.conf: 0.10`, `detector.tracker` (`new_track_thresh 0.35`, `track_buffer 60`) | the tracker can carry a person through a weak frame; a new person still needs a confident box | the detected count changed on 8.7 -> 3.2 of 100 frames (validation) |
| `detector.iou: 0.4` | two boxes overlapping more than this are one person (model default 0.7) | duplicate box pairs 34 -> 1.6 per 100 frames; no person lost on the validation videos |
| `car.stability` (`hold_frames`, `id_memory`, `id_iou`, `ema`) | smooth boxes, keep a lost box on screen a few frames, a person the tracker renames keeps their ID and votes | IDs per real person 2.8 -> 1.8; unknown 7.8% -> 5.1% |
| `classifier.crops.log_offsets` | added to each class's log-probability | trades errors between classes, see below |

**The class boost is a trade, not a fix.** `female` and `child` offsets move errors between men, women and children; no value is right for all three. Validation -> test, single frame, men / women / children correct:

| `female / child` | validation | test |
|---|---|---|
| 0 / 0 | 95 / 43 / 67 | 81 / 51 / 82 |
| **1.5 / 1.0 (shipped, owner's choice)** | 87 / 74 / 62 | 71 / 70 / 76 |
| 1.5 / 1.5 (first choice, by validation balanced accuracy) | 85 / 69 / 82 | 69 / 62 / 83 |
| 2.0 / 0.5 (earlier) | 85 / 84 / 52 | 69 / 80 / 58 |

Set it from your own camera's data once you have it. The classifier itself is the limit (about 75% right per frame); more distinct real cabin people is what raises it.

Known limits: at night the detector reaches only 0.10-0.22 confidence, no track starts and people stay unlabelled (`new_track_thresh: 0.2` helps the night clip and adds flicker by day, so it is not shipped); belts and child seats are not checked by `two_stage`.

## Training data: what exists (searched 2026-10-23; NOT yet downloaded or licence-checked)

No single public dataset has occupants + seatbelts + child seats. Candidates found, each to be opened and its licence read
**before** use; `passenger.datasets.car_cabin add` refuses a dataset without `--license`.

| Candidate | Useful for | Caveat |
|---|---|---|
| Roboflow `seatbelt_ilg` (universe.roboflow.com/datascince/seatbelt_ilg) | `seatbelt` | listed as CC BY 4.0; contents not inspected (may be driver-only, no passengers) |
| SVIRO (sviro.kl.dfki.de) | rear-seat people incl. children/infants, child seats, infant seats | synthetic; licence not read; rear bench only |
| TICaM (ToF cabin dataset, BMVC 2021) | occupants, child seats | depth/IR focus; seatbelt labels unconfirmed; licence not read |
| Own cabin footage | everything, on your real camera | slowest; needs labelling with all six classes (the existing CVAT flow in `ANNOTATION.md` only has male/female/child) |

Expect to combine two or three sources. Synthetic data alone will not transfer to a real camera: plan to label some of your own footage.

## Build the training set

```bash
python -m passenger.datasets.car_cabin add "C:/dl/belts" --name belts --license "CC BY 4.0" --url <page> \
    --map seatbelt=seatbelt "no-seatbelt=no_seatbelt" person=drop
python -m passenger.datasets.car_cabin add "C:/dl/kids" --name kids --license "<read it>" \
    --map child=child "child seat=child_seat" adult=drop
python -m passenger.datasets.car_cabin stats
```

* Source classes need a `--map` entry (`x=drop` discards one); an unmapped class is an error, not a silent drop.
* Splits are by *group* (clip/scene from the file stem, or `--group-regex`), never by image, and a group's split never changes.
  If frames are named only by number, every frame becomes its own group and near-identical frames can straddle splits:
  pass `--group-regex` for such data or the test score is inflated.
* Every source's licence, URL, class map and counts are recorded in `car_data/sources.json`.
* `stats` warns when a class has under 200 training boxes. A class the model never saw cannot be checked: e.g. with no
  `child_seat` boxes the child-seat alert will fire for every child.

## Train and score

```bash
python -m passenger.train_car --epochs 60 --imgsz 640 --device 0 --name v1      # GPU machine, not the Pi
copy models\car_cabin\v1\run\weights\best.pt models\car_cabin\champion.pt
```

`metrics.json` has per-class precision/recall/AP, and the cabin-level numbers that matter: child-present precision/recall,
child-seat count error, belt recall. A class with no test boxes is `null` and listed under `untested_classes`: not measured, not good.

## Limits to know

* Gender from the whole body is harder than from a face; expect `male`/`female` to be the weakest classes. If it disappoints, the
  fix is a face-attribute stage on the `male|female|child` boxes (the bus pipeline already has one) and more cabin data.
* Seatbelt: a belt is thin and often hidden by an arm or coat. "Belted" needs a belt in only 30% of recent frames (`belt_on_ratio`)
  so a single sighting outweighs misses; the cost is that a belt that is never seen reads "NO BELT". Tune on your footage.
* A child in a child seat gets no belt verdict (the seat harness replaces it).
* No accuracy figures exist for this model yet: there are no weights and no data in the repo.
