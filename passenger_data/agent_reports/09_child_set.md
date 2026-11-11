# 09 Child set `rf_children-on-car-detection_v2`: registered, deduped, grouped, exported (2026-10-29)

Nothing trained, registered or promoted; `config.yaml` and `models/passenger/registry.json` untouched. No project code added (builder scripts are scratch, session scratchpad `.../scratchpad/s9/`; `a1.py` audit, `a3.py` unlabelled-person scan, `a7.py` + `d1.py` perceptual check, `groups.py` my grouping, `c1.py` selection, `c2.py` export). New exports (older ones untouched): `person_det_v2`, `crops_mfc_v4`.

## 1. Registration
- Copied from stage4 to `passenger_data/raw/external/rf_children-on-car-detection_v2/` (774 files, 20 MB; train/images 385 + labels 385 + READMEs + data.yaml + SOURCE.json). Original SOURCE.json keys kept, added: licence `CC BY 4.0 (as stated by uploader / Roboflow metadata)`, `content_rights: unverified`, selection rule `full export as downloaded`, classes `0 child only (adults are not labelled)`, retrieval date, note (internal use only, not commercial, untrusted data never executed).
- Registered with `DataManager.register_raw_files` (774 files with sha256). `python -m passenger.data_manager verify` after registration: `raw data intact` (twice: right after registering, and again after the exports). Raw total now about 3.9 GB + 20 MB, far below the 8 GB cap.
- Data format as received: 385 images, all 640x640 (Roboflow resize, likely stretched: faces look slightly squeezed in many frames), 612 `child` boxes, 1 image without labels.
- Untrusted data handling: every read used `python -I`; nothing from the set was executed.

## 2. What I viewed
All 385 images with their boxes drawn (13 contact sheets, 30 per sheet, ordered by colour similarity so clips sit together), then the 34 images where yolov8n (conf 0.25, class person) found a person not covered by a label, then the final exported val (31), test (25) and 30 random train images from the export labels. Boxes are tight on the body or visible part in nearly all; weak spots: several head-only or hair-only boxes on a child leaning out of frame (phone clips, siblings), boxes cut at the frame edge, 1 junk label image (below). Many frames are rotated or lying children (phone clips held sideways).

## 3. Dedupe and exclusions (385 -> 293 images)
| step | images | note |
|---|---|---|
| in the set | 385 | |
| duplicate of a held image, 32x32 grey corr > 0.97 | -63 | vs person_det_v1 (all splits), car_cabin_v2, raw baby_adult, 6ckvk, ihdwp (6,267 held images). 31 baby_adult, 20 6ckvk, 11 det_v1 copies (same photos), 1 ihdwp. Gap is clean: next highest corr is 0.75 |
| duplicate by perceptual check (ORB + RANSAC inliers >= 20 against the 6 nearest held images by yolov8n-backbone embedding) | -3 | image 23, 22 and 354 are crops/re-exports of baby_adult photos (211, 159, 67 inliers) that 32x32 correlation misses (0.67 for #23). ORB also confirmed 59 of the 63 corr duplicates; nothing else reached 15 inliers |
| unlabelled adult visible | -10 | see section 5 (11 flagged, one of them already a duplicate) |
| junk labels (image 314: tiny boxes, large box covers legs only) | -1 | |
| near-duplicate frame inside the set (same group, grey corr > 0.92 or colour corr > 0.97, first kept) | -15 | |
| **kept** | **293** | boxes: 3 more dropped by the "< 8 px at 416" filter; 4 crops dropped by the "< 24 px" rule |

Group-level leakage check: group F4 (curly-haired child under a brick-pattern ceiling, 21 images) are other frames of the 6ckvk clip `photo_2023-11-12_20-27-*` whose frames are in person_det_v1 train; 19 of the 21 are near-identical copies of held frames and were dropped as duplicates (or as near-duplicates of each other); the 2 left inherit train (32x32 correlation alone would have put them anywhere). The other duplicates were single stock photos.

## 4. Groups (by viewing; frames of one phone video or family = one group, each stock photo its own group)
| group | images kept | what it is | split |
|---|---|---|---|
| `fam_siblings_seats` | 208 (of 222) | the same two small siblings (boy + girl) in a grey and a red child seat, pink backpack, many short phone videos with different clothes, one clip with a toy and the green tablet; I merged all clips that show the same seats, backpack, tablet/phone holder or children. The most numerous group by far | train |
| `fam_curly_brick` | 2 (of 21) | see above, same clip as held 6ckvk frames | train |
| `fam_dark_lying`, `fam_tropical_boy`, `fam_pair_belt` | 9, 4, 3 | small phone clips (lying on a dark bench at night; boy in tropical shirt; boy + girl with belt straps). Possibly the same children as `fam_siblings_seats` (not provable), which is one reason they stay in train | train |
| `fam_babies_blue_seat` | 26 | babies in a blue child seat, three outfits (pink shirt, white dress, striped pyjamas), one to three babies; frames of one or a few phone videos | **val** |
| `fam_sisters_grey_car` | 20 | two to three girls (lilac "Believe" T-shirt, floral top, a third in a Ladybug T-shirt) on a grey leather bench, phone video | **test** |
| stock/web photos | 21 groups of 1 image | 11 in train, 5 val (`stock_000, 008, 035, 050, 360`), 5 test (`stock_007, 024, 025, 030, 036`); chosen by a seeded hash of the group name, then viewed | train/val/test |
| `fam_night_adults` | 0 | purple night photos with adults, excluded (section 5) | none |

Why this split: train has about 1 distinct real in-car child and the set's most numerous group has only two children, so I gave train every group that cannot be split (the big family and the groups that may be the same children), and still moved whole families that are clearly different children (different car, seat and clothes) to val and test, plus single photos so each of val and test gets several distinct children. Val gets the babies (child-seat viewpoint, babies are the weakest detector case); test gets the older children seated on a bench (closest to children on a rear bench under a dash/ceiling camera). All of a group's frames are in one split. Alternative I did not take: leaving `fam_siblings_seats` out of train to report its separate score; it is better as the largest train group.

Distinct children per split, this set only (by eye; family counts are estimates, +/-1):
| split | families | single photos | distinct children (about) | images | boxes |
|---|---|---|---|---|---|
| train | siblings 2-3, curly 1, dark_lying 1-2, tropical 1, pair_belt 2 (may overlap the siblings) | 11 | **about 17-20** | 237 | 415 |
| val | babies 1-3 | 5 | **6-8** | 31 | 31 |
| test | sisters 2-3 | 5 | **7-8** | 25 | 45 |
Real in-car children before this set (08 report): train 1, val 2, test 3. After: train about 18-21, val about 8-10, test about 10-11 (these are all children in the child class, adults are in the car-cabin frames).

Per-source tag in provenance (`source` column): `coc2_phone` (phone-video families, one source domain) and `coc2_stock` (stock/web photos, watermarked). `group` holds `coc2:<group name>`. Per-source scores can be reported by filtering on that column in `person_det_v2/provenance.csv` and `crops_mfc_v4/provenance.csv`.

## 5. Unlabelled adults and other people (the detector risk)
The dataset labels `child` only, so an adult next to the child is an unlabelled person and would teach a detector that people are background. Method: yolov8n person detections (conf >= 0.25, box > 2% of the image) that are not covered by a label (IoU or containment < 0.5) = 42 images (34 not already duplicates); I viewed them with the detection drawn. Result:
- **Real unlabelled adults, excluded from every split (not added as boxes):** images 9 (man buckling the child), 14 (woman beside a baby), 26 (smiling mother), 354 (driver, also a duplicate), 358, 362 (women in the front seats, 362 with a second unlabelled child), and 350, 353, 359, 361, 363 (a purple night/IR set with 2-3 adults each; 363 has no label at all). 11 images, 10 after the duplicate step. I did not add adult boxes: yolov8n adult boxes there were not reliable enough (partial bodies, seats detected as people) and the owner asked for correctness over count. The same rule also keeps them out of the crops set, though their child boxes were right.
- **False alarms (23 images, kept):** the detector fired on the same child (box much larger than the label), on a seat headrest, or on a second child whose partial box was labelled; viewed one by one.
- **Not removed, small residual risk:** adult hands or arms only (images 60, 59, 62 left edge), a second small child partly behind a seat in the sibling clips whose box is a head or hair only, and background passengers too small for yolov8n to catch at 0.25. I read all 385 images and saw no other adult. Estimated remaining unlabelled-adult images in the kept set: 0-3 of 293, not measurable without a second reviewer.
- The 510 new car_cabin_v2 frames and v1 content carry the labelling quality described in 05/08 (adults and children all labelled by tracked boxes, no known unlabelled people other than flags noted there).

## 6. `person_det_v2`
One class `person` (`data.yaml` same format as v1: `path`, `train/val/test: images/<split>`, `nc: 1`, `names: {0: person}`), long side 416 px (child set 640x640 -> 416x416 INTER_AREA; car frames <= 1280 -> 416; never upscaled), names unique, `provenance.csv` columns as v1. Built as: person_det_v1 copied unchanged (1,369 images) + 510 car_cabin_v2 frames that are not in v1 (the 311 v1 car frames are not duplicated; asserted: same split, same box count) + 293 child-set images (all boxes -> person, 3 boxes < 8 px removed). doll/teddy negatives: the 40 v1 negatives kept (train, empty labels).

| | train | val | test | total |
|---|---|---|---|---|
| images | 1,579 | 355 | 238 | 2,172 |
| boxes | 2,819 | 675 | 611 | 4,105 |
| groups (`group` column) | 773 | 97 | 14 | |
| car-cabin videos | 20 | 11 | 8 | 39 |
| fingerprint (label files) | `c3157a189e0cf989` | `ce3c850de8bf9164` | `4a2b68f5cbb3f2d9` | |

Per source (images; boxes), train / val / test: 6ckvk 401;660 / 49;77 / 0. baby_adult 162;267 / 19;29 / 0. ihdwp 358;834 / 29;75 / 0. doll negatives 40;0 / 0 / 0. car_cabin_v1 (old frames) 134;260 / 85;192 / 92;364. car_cabin_v2 (new frames) 247;383 / 142;271 / 121;202. coc2_phone 226;403 / 26;26 / 20;40. coc2_stock 11;12 / 5;5 / 5;5.
Test now: 8 car videos (213 frames, about 20 distinct cabin people per 08: 11 men, 6 women, 3 children) + 25 child-set images, so test is no longer "92 frames from 2 videos" and **the benchmark changed**: scores on person_det_v1 test are not comparable.

Leakage guards (asserted in the build, all passed): no group spans two splits (773/97/14 groups); the 8 car test videos and 11 val videos have no frame in train, and no train video in val/test; no non-car, non-child-set image in test; no byte-identical file in two splits; no old car frame changed split; train vs val/test 32x32 correlation max 0.936 (> 0.95: none dropped); val vs test max 0.678. Group check beyond 32x32: the child set's groups were built by viewing and F4 inherited train from the held clip; **limits**: the held 6ckvk/baby_adult/ihdwp groups in v1 are single-image or hash groups, not clips, so v1's own 10% val slice can still share a clip with train (inherited from v1, not changed); same-person leakage in the families is as well as I could see (the siblings may also appear in `fam_dark_lying`, `fam_tropical_boy`, `fam_pair_belt`, all train).

Ultralytics dry check (no training): `check_det_dataset` parses `data.yaml` (nc 1, names {0: person}, three split paths); `YOLODataset` scan of train/val/test found 1,579 / 355 / 238 images, 40 backgrounds, **0 corrupt**, boxes 2,819 / 675 / 611 (matches provenance). The label caches it wrote were deleted again (no `.cache` files left in the export).

## 7. `crops_mfc_v4`
`crops_mfc_v3` copied unchanged (v3 untouched) + child crops from the same 293 images by the v3 rule (top 60% of the box, square, edge-replicate pad, 112 px, min side 24 px), unique names `coc2_<image>_<box index>_<hash>.jpg`, each crop in its group's split, no PA-100K or augmented crop in val/test. Per crop provenance row: source `coc2_phone` / `coc2_stock`, group, `label_origin` dataset child box. Train crops near-duplicate (corr > 0.95) of a val/test crop: none (new train vs new val/test max 0.786; the guard against the old crops dropped 0).
| | male / female / child (v3) | added child | v4 male / female / child |
|---|---|---|---|
| train | 595 / 277 / 771 | +414 (402 phone, 12 stock) | 595 / 277 / **1,185** |
| val | 254 / 175 / 149 | +31 (26 phone, 5 stock) | 254 / 175 / **180** |
| test | 242 / 144 / 180 | +45 (40 phone, 5 stock) | 242 / 144 / **225** |
Caution for the crop trainer: child is now 58% of train crops (was 47% in v3) and 414 of the new ones are two siblings; use the per-source tag and `balanced_order` and do not read the child recall as 20 children's worth. Child val crops: 31 new from about 8 children; test: 45 from about 8 children.

## 8. Caveats
- **Viewpoint**: rear-seat, eye-level, mostly close-up phone video or stock photos. Not the dash/ceiling camera of the deployment; expect a gap. Report the `coc2_*` rows separately from `car_cabin_*`.
- **Two families dominate** (siblings 208 of 293 images, 71%): two (maybe three) children make up 71% of the new images. Few distinct children still: train about 18, not the 25-35 of the raw set (more were dropped as duplicates of held sets).
- **Aspect ratio**: the source images were resized to 640x640 by Roboflow; children look slightly squeezed, and the exported 416x416 keeps that. A detector trained on it sees that distortion on every image of this set.
- **Watermarks** (iStock, "Car Seats for the Little", Khojochi, "BABY GIFT") on several stock photos in train and some in val/test; the detector may use them.
- **Boxes**: head-only/hair-only boxes of children leaning out of frame in the sibling clips (counted as person), boxes cut by the frame edge. Single labeller (the dataset's), no second review.
- **Rights unverified, CC BY 4.0 as stated by the uploader, internal use only** (owner decision). Attribution: `SOURCE.json` and `raw/.../README.roboflow.txt`.
- **Unlabelled adults**: see section 5; residual risk 0-3 images; the detector's `person` class is now correct for the excluded cases only because they were left out.
- **Child vs teen**: the dataset's `child` label is not age-checked; a few older children (stock girl in uniform, the 7-8 year olds in the test group) are borderline for a "child <= 12" decision and are labelled child as supplied.
- Test and val counts for the child set are small (25 / 31 images from 8 children each): a score on them is per-child noise.

## 9. Checks and QA
- `python -m passenger.data_manager verify`: raw data intact (after registration and again after the exports). `validate`: `labels OK` (it checks the registered labels, not the export folders; the exports were checked by the asserts and the Ultralytics scan above).
- QA gate (`python scripts/qa_gate.py --data passenger_data`, run once, after confirming with Get-CimInstance that no python/training process was running; nothing under `passenger/` or `tests/` was edited): tests 452 passed x2 (168 s, 158 s); evaluation on the 630 test frames identical to the previous row (det P 0.672 R 0.788 AP50 0.668, count MAE 0.54, classification macro-F1 0.790, coverage 0.85); **overhead 17.6 ms vs budget 15 ms = FAIL, so the gate exited 1 and is not green**. I did not retry it. A diagnostic `bench_passenger_overhead.py` x3 straight afterwards read 3.3 / 3.6 / 3.8 ms (CPU 8-19% busy with Chrome/VS Code, the suite took 158-168 s vs 90-125 s on quieter runs), so load is the likely reading as with the 15.2 and 19.6 ms readings of 2026-10-28, but the cause is not proven. Logged in WORKFLOW.md (entry and history row). No project code was added, so no new tests.
