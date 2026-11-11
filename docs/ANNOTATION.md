# Annotation walkthrough: from a cabin video to a scored model

You do not need to have used an annotation tool before. This is the whole path, using **CVAT** (free, browser-based:
https://app.cvat.ai, or self-hosted). Menu names can differ a little between CVAT versions; the ideas do not.

Until you have real cabin footage, skip to [Using open data meanwhile](#using-open-data-meanwhile).

## 1. What you are producing

For every frame you label, draw **one rectangle around every person** and give it one of three labels:

| Label | Use for |
|---|---|
| `male` | an adult or teenager who looks male |
| `female` | an adult or teenager who looks female |
| `child` | someone who looks 12 or younger (any gender) |

**Rules that keep the numbers honest** (the scores are only as good as these):
1. **Label everybody** who is visible (head and upper body at least). A person you skip counts against the model as a
   false alarm every time it detects them, so unlabelled people make results look worse than they are.
2. Box the **whole visible person**, tight, including a seated person's visible torso. Do not include the seat.
3. Skip people who are only a sliver at the image edge or too blurred to tell they are a person. Skip posters, mirrors and reflections.
4. If you cannot tell adult from child, ask a colleague or leave that **frame** out. Do not guess. Wrong labels hurt more than missing frames.
5. Two people, two boxes, even if they overlap.
6. Be consistent: the same rule on every frame, ideally one labeller per clip, and a second person re-checking about 10% of frames.

## 2. Record footage worth labelling

- Use the **real camera at the real mounting position**. Seat positions and how small the back rows look depend on it.
- Vary: empty, half-full and full bus; day, dusk and night; different routes and different days.
  **Several different trips matter more than many minutes of one trip**: frames from one trip look alike, so they teach and test less than they seem to.
- Aim for **a few hundred labelled frames to start**, with at least about 100 boxes of each class in the test split. With fewer, a score swings
  wildly on one or two people (our open-data test split has only 13 female people, so its female score means little).
- Passengers are being recorded: get the legal basis, notices and consent your region requires, restrict who can open `raw/`, and agree how long it is kept.

## 3. Steps

All commands run from the repo root. `<id>` is the video id printed by `register`.

```
python -m passenger.data_manager register trip_0412.mp4 --vehicle bus1     # copies it into raw/ (never modified afterwards)
python -m passenger.data_manager extract <id> --fps 2                      # frames to label, 2 per second, near-duplicates dropped
python -m passenger.data_manager splits                                    # assigns whole videos to train / val / test
```

Frames appear in `passenger_data/processed/frames/<id>/`. Zip that folder.

### In CVAT
1. **Create a project** and add three labels of type *Rectangle*, **in exactly this order: `male`, `female`, `child`**.
   (The importer refuses an export whose order differs, because swapped order silently turns males into females.)
2. **Create a task** in the project and upload the **images** (the zip of extracted frames). Do **not** upload the video:
   CVAT then names the exported labels `frame_000000.txt`, which cannot be matched back to our frames.
3. **Label**: open the job, press `N` to start a rectangle, pick the label, drag the box. `F`/`D` move to the next/previous frame.
   Select a box to change its label, resize or delete it. Save often (`Ctrl+S`).
4. **Export**: in the task menu choose *Export task dataset* and format **YOLO 1.1**.
5. Unzip the export anywhere, for example `~/Downloads/trip_0412_export`.

### Back in the repo
```
python -m passenger.data_manager import-labels ~/Downloads/trip_0412_export    # only labels that match a known frame are accepted
python -m passenger.data_manager validate                                       # catches bad class ids and boxes outside the image
python -m passenger.data_manager stats                                          # boxes per class per split: check the balance
python -m passenger.data_manager verify                                         # proves raw/ is still byte-identical
```

## 4. Measure the model

```
python -m passenger.evaluate --split test --source videos --classifier
python scripts/qa_gate.py --data passenger_data            # then paste its block into WORKFLOW.md
```

Read the numbers like this:
- **Detection recall** = of the real people, how many it found. **Precision** = of what it found, how many were real.
- **Count MAE** = on average it is off by this many people.
- **Seat F1** (cabin footage only) = how well it knows which seats are taken. Look at `per_seat_f1` for the back rows.
- **Classification macro-F1 and per-class recall**: the honest numbers for male / female / child. `unknown` counts as a miss, so refusing cannot inflate them.
  `coverage` is how often it dared to answer.

## 5. Seat polygons (needed before seat numbers mean anything)

The seat map in `config.yaml` (`passenger.seats`) is a placeholder grid. Once you have a real frame, measure each seat as 4 points
(0-1 image coordinates) and put them under `passenger.seats.polygons` as `"R1L1": [[x, y], ...]`. The seat outlines drawn on the dashboard feed
show whether they line up. This step is manual today; a click-to-draw helper can be built once there is a real frame to work from.

## 6. When new footage or data arrives (retraining)

```
python -m passenger.data_manager register new_trip.mp4 --vehicle bus1      # ...extract, label, import as above
python -m passenger.train --dataset passengers_v3 --epochs 30 --device 0   # on a GPU machine; starts from the current champion
python -m passenger.registry promote v3                                    # only succeeds if nothing regressed on the same test set
```

The new model trains on **all** data (old and new), never sees the test split, and is promoted only if it does not score worse than the live model
on the same benchmark. See [DATASETS.md](DATASETS.md) for where the data lives and why the originals are always kept.

## Using open data meanwhile

No cabin footage yet? Real, licence-clean photos with male / female / child boxes can be pulled from Open Images:

```
python -m passenger.datasets.open_images --split validation --max-images 400
python -m passenger.evaluate --split test --source open_images_v7_validation --classifier
```

These are **general photos, not bus cabins**. They give real detection and classification numbers today and exercise the retraining path,
but they are not a substitute for cabin footage: the camera angle, crowding and seated postures are different.
