# 03 Clean report (2026-10-28)

Baseline: `data_manager verify` = raw data intact (before and after, 6 min each); `validate` = labels OK; manifest.json not touched by this run. Nothing was written to raw/. Builder scripts (not project code) are in the session scratchpad `.../scratchpad/clean/` (`sb_cluster.py`, `sb_build.py`, `child_build.py`, `seat_build.py`, `face_build.py`, `sheet.py`); every export folder has `provenance.csv` (stem, split, source, origin, original split/name, group, boxes, md5).

## Read this first: there is NO real cabin test set
The owner has no cabin footage. None of the four exports contains an image from a camera the model will run on, and none of their "test"/"val" scores can say anything about a real car. `seatbelt_v1` has a test split that is leak-free but made of web photos, one fleet camera and one driver-monitoring set. `child_seat_synth_v1` and `faces_utk_v1` have no test split on purpose. A real test set must still be made (free-licence car-interior video hand-labelled, as 02_audit.md says) before any car metric is quoted.

## Exports (all new names; no older export touched). Folder: `passenger_data/annotations/<name>/`

| Export | Classes (data.yaml) | Images | Boxes | Splits |
|---|---|---|---|---|
| seatbelt_v1 | 0 seatbelt | 1,753 | 1,949 | train 1,264 / val 219 / test 270 (boxes 1,412 / 245 / 292) |
| child_v1 | 0 child | 906 | 1,274 | train only (no val/test, per the rule) |
| child_seat_synth_v1 | 0 child_seat (SYNTHETIC) | 360 (290 with a box, 70 background) | 440 | train 250 / val 110 (boxes 290 / 150); NO test, by design |
| faces_utk_v1 | classify: 0 male, 1 female, 2 child (folder per class) | 22,522 | n/a | train 19,291 / val 3,231 (male/female/child counts in section 4); NO test |

Fingerprints (DataManager recipe: sha256 over label-file stems and label hashes, first 16 hex):
- seatbelt_v1: train `4914136c25fe358b`, val `1542a45d39a2b084`, test `c740b9bfb0430626`
- child_v1: train `043645b9b23a9d64`
- child_seat_synth_v1: train `7ae6e1864e7a78de`, val `2bbd900c33ead71b` (final build, after the y-flip fix)
- faces_utk_v1 (no label files; sha256 over split+stem+class+md5 of the images): `6277c5d21cee68d3`

These exports are separate from the manifest: they are not registered in `manifest.json` (told not to ingest), so `DataManager.export_yolo` / `stats` do not see them; the trainer must be pointed at each `data.yaml`. I re-checked all 3,663 label rows of the three YOLO exports: class 0 only, coordinates inside 0-1, positive size.

Split rule used everywhere: stable hash of the group key (smallest member name of a union group), so a group never moves. Groups: seatbelt = union of near-duplicate cluster (32x32 grey correlation > 0.97 over habbas + firasmahdi + sachinx + ramankamran together), the `_mp4` clip name, and for fleet frames the 14-digit timestamp; child = video id; child_seat = scene id; faces = near-identical-UTK group. Checked on the written exports: 0 groups and 0 identical files span two splits.

## 1. seatbelt_v1 (habbas11 class 4 + firasmahdi class 5)
Rules applied (audit): class 4/5 only, boxes < 0.3% or > 60% of the image dropped, same-class duplicate boxes (IoU > 0.9) dropped, caps per cluster (fleet and `_mp4` frames 3, web photos 1, firasmahdi 2), supplied splits discarded, cluster-level re-split 70/15/15, grey-copy web group capped at 50% of the final set (not binding: 725 of 1,753 = 41%).

| Step | habbas11 | firasmahdi |
|---|---|---|
| images in raw | 3,939 | 1,321 |
| no belt-class box (not used as negative) | 0 | 1,032 |
| belt boxes < 0.3% | 17 boxes | 14 boxes |
| belt boxes > 60% | 62 boxes | 0 |
| duplicate boxes IoU > 0.9 | 5 boxes | 0 |
| images left with no usable box | 66 | 13 |
| rotated variants with a black border (>8% of an edge) | n/a | 50 |
| duplicate of a habbas image (corr > 0.97), habbas copy kept | n/a | 17 |
| removed by the per-cluster cap | 2,277 | 22 |
| train images removed for corr > 0.95 / identical bytes to a val/test image | 30 (both sets together) | |
| polygon rows converted | 0 (no class-4/5 polygons exist; the audit's "8 polygons" are other classes) | 0 |

Kept per split and origin (images): train 1,264 = fleet 287, mp4 130, web grey copy 494, web other 240, firasmahdi 113; val 219 = 24, 3, 107, 60, 25; test 270 = 29, 36, 124, 52, 29. Origin "web_grey_copy" = habbas greyscale copies of sachinx web photos (close-up selfies).
Not done: the sachinx colour-copy substitution (habbas boxes are in habbas pixel space; sachinx is squashed to 256x256, no box transfer attempted).
Estimate vs audit: 1,753 images / 1,949 boxes, above the audit's 1,200-1,600 / 1,400-1,900 (the clip-union made groups, not images, the limit; per-cluster caps left more frames).

Visual check (contact sheets with boxes drawn, about 60 kept boxes: 24 firasmahdi, 24 web, 12 fleet): most sit on a strap or belt; doubtful ones: boxes on the belt of an empty seat (web tile with leather seats), belt-only boxes without a person, long thin boxes at the image edge, loose chest boxes in firasmahdi; about 3 of 24 firasmahdi images are upside-down (180-degree variants have no black border, so my filter cannot see them).

## 2. child_v1 (hf_slownite_child_detection)
Raw 1,928 images (train 1,102 + test 826, videos disjoint). Rules: boxes 0.5%-60% of image kept, image dropped without a box, <= 3 keyframes per video (first, middle, last), all train.
Removed: 274 boxes > 60% (close-ups), 1 box < 0.5%, 339 images with no usable box left, 682 images by the 3-keyframe cap, 1 exact duplicate. Kept 906 images, 1,274 boxes, 321 videos (519 from slownite's train, 387 from its test; slownite's own split and the manifest's hash split are ignored).
The ~20% cap: this export holds only slownite, so the cap cannot be applied inside it. Whoever mixes it must limit slownite child boxes to <= 20% of the child boxes in the training mix (with the 204 Open Images train child boxes that means very few; use it as a separate pretraining stage as the audit suggested). Because it is train-only, `data.yaml` has no `val`: any trainer needing one must use a different set. Boxes of children missing a label (audit saw one) are unfixed. Review list: none produced (no label changes were made; the audit's missing-label observation stands).

## 3. child_seat_synth_v1 (sviro_intercar_subset, SYNTHETIC)
Method: seat codes from the scene folder name (`GT_a_b_c` = left, middle, right seat; code 1 infant seat, 2 child seat); keypoints files `*_2d_<car>.txt` per person, matched to occupied seats by sorting persons left-to-right (median keypoint x; the count of keypoint files equalled the occupied-seat count in all 420 scene-car images, 0 dropped). Box = bounding box of the person's keypoints, widened and heightened by 20%, clipped.
Defect found and fixed: SVIRO 2D keypoints are y-from-bottom. The first build (y as given) put every box above its seat; I drew 24 images, saw it, flipped y (`y = H - y`), rebuilt, redrew 24 images. Final check (24 train images drawn, ~35 boxes): all but about 3 sit on a child or infant occupant; 2-3 on dark images sit on a barely visible occupant. Limit of the method: the box is the occupant skeleton plus 20%, so it covers the child and the upper seat, but not the whole seat shell (the seat sticks out below and behind). For a detector that is meant to find the installed seat, these boxes are tight; a larger expansion would fit better but is not what was asked.
Counts: 42 scenes, 29 with a child/infant seat (230 infant, 210 child-seat boxes); train 25 scenes, val 11 scenes (the hash gave val 34% of boxes: bigger than 15-20% but it can never move). 70 background images (7 scenes with only adults/empty seats, <= 25% of the positives); 60 further negative images left out for that cap. Never in a test split. SVIRO must stay <= 30% of child_seat boxes in any training mix, but the real share is 100% today: there are no real child_seat boxes anywhere in the repo. Review list `03_clean_review_child_seat_synth_v1.csv`: 81 box-carrying images with mean brightness < 25 (occupant barely visible).

## 4. faces_utk_v1 (hf_py97_utkface_cropped)
`passenger/datasets/utkface.py` could not give child <= 12 (it only returned FairFace age groups, 3-9 / 10-19). Small change: `parse_age_gender` (exact age; `parse_name` now calls it, same results) and `face_class` (child <= 12, male/female >= 18, else None); 2 new tests (plus the 20 existing) pass. The file is a single `UTKFace.tar.gz` (not part*.tar.gz); I read the tar without extracting.
Removed: 3 unparsable names, 820 aged 13-17, 363 exact duplicates (md5), 0 near-duplicates of FairFace, 0 class caps hit.
Dedupe against FairFace (97,698 faces, train + validation): the audit's dHash <= 6 matched 8,614 of 22,885 UTK faces and 40% of a 1,500-face probe at chance level; pairs viewed at distance <= 2 were different people (aligned faces all look alike to an 8x8 hash). It was NOT used. Replacement: 32x32 grey correlation of the UTK face against the FairFace face with its 0.25 padding cropped away, > 0.95: 0 matches, highest 0.927 (top pairs viewed: different people). This only catches the same crop; a same-photo different-crop duplicate would need face embeddings, not available. Caps (FairFace train counts: child 0-9 = 12,200, male 20+ = 35,011, female 20+ = 30,430) never bound. Split: 85/15 train/val by near-identical group (UTK dHash <= 2, 1,834 images in such groups); no test split; FairFace's own validation file remains the face benchmark.
Counts (face crops): train male 8,721 / female 7,674 / child 2,896; val 1,466 / 1,277 / 488. Total 22,522.
Note the child cut differs from FairFace's (FairFace child = 0-9 only; UTK child = 0-12), so UTK children aged 10-12 are labelled child where FairFace calls the group 10-19 "ambiguous". Mixed training sees the two definitions.

## Class imbalance for the trainer
- seatbelt_v1: one class, so imbalance is by origin: web photos 61% (close-ups; 725 are grey copies), fleet+mp4 29%, firasmahdi 10%. One fleet camera shows essentially one man (red shirt/uniform) in IR: expect over-fit to that camera. About 90% of belt boxes sit on a real belt (audit estimate, and the 60 boxes I viewed agree roughly); all images have positives only: unboxed people are unlabelled belts (missing-label risk).
- child_v1: close-ups; median box was 26% of the image; mixing limit 20%.
- child_seat_synth_v1: 290 train boxes (below the 200-box warning only after the mix; 42 layouts, 10 renders each of the same scene), 100% synthetic.
- faces_utk_v1: child is 15% of train (2,896 vs 8,721 male); UTK is adult-skewed and frontal; no cabin look.
- no_seatbelt and male/female whole-body car boxes: no data produced (not in scope and no source).

## Rejected / skipped (per audit, not touched)
c3rl, bus set (for the car model), car_cabin_videos_stock (no rights, unlabelled), sachinx/lavdeep/youcef/mariam/roboflow_seatbelttraining/alexandresintes/nexdata (no training), ramankamran (not requested; only used to check overlap: 0), sviro_intercar_people_subset.

## Honest limits
- Label accuracy was judged by eye on about 60 belt boxes and 24 SVIRO images; no gender/age ground truth exists for the car sets.
- Test splits in `seatbelt_v1` are leak-free by cluster but domain-biased; no real in-car test set exists.
- Licences: slownite CC BY-NC-SA, UTKFace NC, SVIRO (believed) NC, habbas web photos and fleet frames (rights unverified, real drivers), firasmahdi (stock watermarks): internal training only; do not redistribute these exports or publish weights trained on them without the owner's decision.
- The exports are not in the manifest and are not covered by `DataManager.stats`; fingerprints above are the identity to cite.
- QA gate (`python scripts/qa_gate.py --data passenger_data`): 425 passed x2, overhead 9.8 ms [OK]; evaluation numbers ran on the manifest's old test frames (417), unrelated to these exports, and are not comparable to the previous row (slownite test frames are now in the manifest set).
