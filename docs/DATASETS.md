# Datasets, data layout and retraining

## The rule: originals are never lost

`passenger_data/raw/` holds everything exactly as it was received. **No code in this repo edits or deletes it**, and
`python -m passenger.data_manager verify` re-checks every file against the checksum recorded when it was registered.
Everything else is *derived* and can be deleted and rebuilt from `raw/` + the human-made labels:

```
passenger_data/
  raw/                              ORIGINALS (immutable; back this up, it cannot be regenerated)
    videos/<vehicle>/<id>.mp4         your own cabin footage
    external/<dataset_id>/            open datasets exactly as downloaded
      images/  metadata/  SOURCE.json   (original CSVs, per-image attribution, licence, retrieval date, selection rule)
  processed/                        DERIVED (rebuildable)
    frames/<video_id>/*.jpg           frames extracted from your videos
  annotations/
    labels/*.txt                      YOUR hand-made labels (precious: back up, never regenerated)
    external/<dataset_id>/labels/     labels converted from an open dataset (rebuildable from raw/)
    <export_name>/                    training sets for YOLO (images/, labels/, data.yaml); rebuildable
  manifest.json                     index: videos, external datasets, splits, checksums, provenance
models/passenger/
  registry.json                     every model version: lineage, dataset fingerprints, benchmark scores, champion
  champion.pt  v1/ v2/ ...          weights (not in git)
```

Why originals are kept even after preprocessing: new labelling rules, a different frame rate, a new model input size, ONNX/accelerator
calibration images, a re-split or a bug found in a converter all need the original data, and none of those can be redone from derived files.

## Splits (what makes the scores trustworthy)

- **Your own videos** are assigned to train / val / test **as whole videos**, and a video's split **never changes** once assigned,
  even as footage is added. (Neighbouring frames look alike; a frame-level split would let the test set leak into training.)
- Clips of the **same people or scene** share a `group` (set with `register --group`); `assign_splits` / `assign_group` put a whole group into one split,
  so the same person never appears in train and test. Group an old clip with `DataManager.set_group` (its split is not touched).
- **External images** get a **stable split from a hash of their name**, so the same image is always in the same split regardless of what else is added.
- The **test split is never trained on.** The registry stores a fingerprint of the test split with every score and refuses to compare scores
  computed on different test sets. Adding data to the test split starts a new benchmark: re-score the champion on it first.

## Sources

| Source | Status | Licence | Used for |
|---|---|---|---|
| **Open Images V7**: `Man` / `Woman` / `Boy` / `Girl` boxes | **Ingested** (400 photos from the validation set + 376 from the test set) via `passenger.datasets.open_images` | Annotations CC BY 4.0, images CC BY 2.0 (per-image author in `metadata/attribution.csv`; attribution required) | Real-photo evaluation and retraining plumbing. General scenes, only 5 of the 400 are bus scenes (229 other vehicle scenes first-ranked). Not a bus cabin. |
| **FairFace** (108k faces, gender + 9 age groups) | **Ingested** (train + validation files, 0.25 padding) via `passenger.datasets.fairface` | CC BY 4.0 | Trains the face classifier (`passenger.train_faces`); its official validation file is the benchmark and is never trained on. Source: https://github.com/joojs/fairface |
| **Roboflow "Inside Bus View"** (1,378 images, occupied / unoccupied seats) | Candidate, not ingested | Public domain per its page | Seat-occupancy checks. Labels are seats, not people. Download needs a Roboflow account; export as YOLO and register with `DataManager.add_external_dataset`. |
| **Multi-view in-cabin bus dataset** (German city bus, 9,136 samples) | **Not used** | **No licence stated** | Real bus interiors, but only 3D occupant boxes (no 2D boxes, no gender/child labels). Ask the authors for a licence before any use. |
| **Your cabin footage** | Waiting for footage | Yours | The only data that measures real cabin accuracy. |

Check each licence on the dataset's own page before relying on it; listing sites can be wrong or out of date. Whether CC BY attribution
obligations extend to a model trained on the images is a legal question: ask whoever reviews licences for you before commercial release.

## Adding another dataset

Put the originals in `raw/external/<dataset_id>/` (keep their licence text and a `SOURCE.json`), convert their labels to
`(class_id, cx, cy, w, h)` with male = 0, female = 1, child = 2, then call `DataManager.add_external_dataset(dataset_id, meta, items)`.
`passenger/datasets/open_images.py` is the worked example. Nothing else changes: export, evaluation, training and the QA gate
pick the new data up automatically.

## Retraining when data arrives

```
python -m passenger.datasets.open_images ...        # or: register + label your own footage (docs/ANNOTATION.md)
python -m passenger.train --dataset passengers_v2 --epochs 30 --device 0
python -m passenger.registry list
python -m passenger.registry promote v2             # refuses if recall / precision / AP / count error / classification F1 regressed
```

`passenger.train` trains on the union of all data, starting from the current champion. Each model records its base, its data fingerprints and its
scores. To make the pipeline use a promoted model set `passenger.detector.model: models/passenger/champion.pt` in `config.yaml`
(and `classes: null` for a model fine-tuned on male / female / child). Train on a GPU machine, not on the Raspberry Pi.

## Retraining the face classifier

```
python -m passenger.datasets.fairface                    # once: download + register FairFace
python -m passenger.train_faces --epochs 4               # trains, registers a candidate in models/passenger/face/
python -m passenger.tune_classifier --version v2         # thresholds, chosen on open-data train+val only
python -m passenger.face_eval --version v2               # FairFace benchmark + the whole pipeline on the test photos
python -m passenger.registry --face promote v2           # only if no key metric regressed
```

When real cabin faces are labelled (docs/ANNOTATION.md), they can be added to this training in the same way. After adding data to the **test** split,
re-score the champions first (`python -m passenger.face_eval --version <champion>`, `python -m passenger.registry rescore`): scores from different test sets are refused.
