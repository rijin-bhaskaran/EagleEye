# 05 Clean report, round 3 (2026-10-28)

Rules: `02_audit.md` "Round 3: real-car sets". Baseline and after: `python -m passenger.data_manager verify` = raw data intact (before and after), `validate` = labels OK. raw/ and manifest.json were not written. Builder scripts (not project code, so no new tests) are in the session scratchpad `.../scratchpad/r3c/`. Every export folder has `provenance.csv`. No training was done.

## Exports (new names; nothing older touched)
| Export | Content | Images / crops | Boxes |
|---|---|---|---|
| `person_det_v1` | YOLO, 1 class `person`, long side 416 px | train 1,095 / val 182 / test 92 = 1,369 | 2,021 / 373 / 364 |
| `crops_mfc_v1` (fine-tune) | folders {train,val,test}/{male,female,child}, 112 px | train 1,260 / val 307 / test 364 | train male 286, female 203, child 771; val 100 / 96 / 111; test 92 / 92 / 180 |
| `crops_mfc_v1/pretrain_pa100k` | PA-100K adult crops, male/female only | train 10,000 / holdout 2,000 (+2,000 `check_official_test`) | 5,000+5,000 / 1,000+1,000 / 1,000+1,000 |
| `crops_mfc_v1/review_pending` | 0.7-0.9 gender crops, NOT in any split | 178 | see review CSV |

Fingerprints. person_det_v1 (DataManager recipe, label files): train `86c379540f760d80`, val `79b3a08a865b8154`, test `01e917af0beb1b08`. crops_mfc_v1 (no label files; sha256 over relative path + image md5): train `c3465bbfed64535d`, val `5aca7f1cb94eedb9`, test `9a19fea1736bd57c`; pretrain train `57716bf996b8b043`, holdout `d29ee1c19260e811`, check_official_test `e064b2c179d8a8e5`.

## A. person_det_v1: per source, before -> after
| Source | In raw | After rules | train / val / test (images; boxes) |
|---|---|---|---|
| rf_seat-belt-ihdwp_v1 | 1,035 | 387 images (146 groups, <= 3 frames each) | 358 / 29 / 0; 834 / 75 boxes |
| rf_baby-adult-in-car_v1 | 213 | 181 (-29 duplicates of 6ckvk, -3 near-duplicate clusters) | 162 / 19 / 0; 267 / 29 |
| rf_child-detection-6ckvk_v4 | 2,829 | 450 drawn from 1,283 child groups | 401 / 49 / 0; 660 / 77 |
| 6ckvk doll/teddy negatives | 1,113 doll/teddy-only photos | 40, empty labels | 40 / 0 / 0 |
| car_cabin_v1 (all boxes -> person) | 311 | 311 | 134 / 85 / 92; 260 / 192 / 364 |

Removed and why:
- ihdwp: 647 frames beyond 3 per group (first/middle/last), 1 image without labels. Supplied train/valid/test discarded.
- 6ckvk: 184 supplied-test images, plus 95 more images whose near-duplicate group touches a supplied-test image (group unused); 29 baby-adult images that duplicate 6ckvk photos; 4 child groups dropped by the re-box rule (label unmatched and < 48 px); 833 child groups not drawn (cap 450); 972 doll-only groups unused except 40 negatives; 49 doll/teddy label boxes deleted.
- Train images that near-duplicate (corr > 0.95) a val/test image: **0** (max train-vs-val/test corr 0.936). Guards in the build assert: no group spans two splits; car-vid-5 and car-vid-8 (test) and the car val videos never in train; no non-car image in test.
- Box filters (zero area / out of range, < 8 px at 416, duplicates IoU > 0.9): none triggered in the export.

Splits. ihdwp: grouped by clip id (146 groups from clip id + corr > 0.97), 12 whole groups to val by hash. baby-adult/6ckvk: 10% of groups to val by hash (joint groups, so duplicated photos share one split). Validation = car val frames (85) + these slices (ihdwp 29, child sets 68). **Test = car_cabin_v1 test only: 92 frames from 2 videos, about 7 distinct people** (4 adults, 3 children); a score there is per-person noise, not a car accuracy.
Deviation: ihdwp groups use corr > 0.97, not 0.9: at 0.9 one component swallowed 584 of 1,035 images. 146 groups give 387 images (target was 440; the group count is the limit).

### Re-box (child/adult labels -> person boxes) on the 1,579 candidate images
Label matched to a yolov8n (conf 0.3) detection: 2,173; original box kept (min side >= 48 px): 89; images dropped: 4; unmatched detections kept as `person`: 292; unmatched detections dropped for overlapping a kept box (IoU >= 0.45): 277; doll-overlapping detections removed: 12; duplicate boxes dropped: 30.
Deviation from the audit rule: a label also matches if >= 80% of it lies inside the detection. With IoU >= 0.2 alone, face-sized child labels (about half of 6ckvk) fell back to a face box while the same child's body became an extra `person`; I saw this on the first sheet and changed the rule.
Review sheet `05_rebox_sheet.jpg` (60 random images: 42 6ckvk + 18 baby-adult; white = original, magenta = doll (deleted), green = child, blue = adult, yellow = original box kept, red = unmatched detection). My reading of the final sheet: about 7 of 60 images (12%) hold a doubtful box: spurious or partial red boxes (about 5 of roughly 15 red boxes), 1-2 duplicates on one child, face-sized boxes kept on two babies in seats. No child box had the wrong class or missed its child in what I saw. Treat the error as 10-15% (60 images, one reviewer).
Drawn check: 24 images of the final export (7 ihdwp, 6 6ckvk, 4 baby-adult, 5 car, 2 negatives) viewed; boxes sit on people and ihdwp rear passengers are tight.
Doll sheet `05_doll_negatives_sheet.jpg` (40): 18 in-car teddies on seats or belts, 22 dolls. **No doll in a car was found among the 423 doll clusters** (all off-car, mostly one baby doll in a box), so the doll negatives are largely off-domain. I picked them myself; veto or replace on the sheet. Candidates had no yolov8n person at conf 0.25.

## B. crops_mfc_v1
Crop = top 60% of the box, padded to square with edge colour, 112 px. Group split identical to part A (a crop's image is in the same split as its detector group). Train crops with corr > 0.95 to a val/test crop: 0 (max 0.920).
| Source | train male / female / child | val | test |
|---|---|---|---|
| car_cabin_v1 (own labels, split of the video) | 101 / 120 / 39 | 85 / 85 / 22 | 92 / 92 / 180 |
| rf_child-detection-6ckvk (child = dataset class; adults gender-gated) | 61 / 74 / 534 | 3 / 6 / 66 | 0 |
| rf_baby-adult | 3 / 3 / 198 | 0 / 3 / 23 | 0 |
| rf_seat-belt-ihdwp (face-classifier gender) | 121 / 6 / 0 | 12 / 2 / 0 | 0 |
| **total** | **286 / 203 / 771** | **100 / 96 / 111** | **92 / 92 / 180** |

Audit targets were child ~950, male ~870, female ~755 (2,500 total); train reaches 1,260. Reasons: only car sources were used (not the gender set, not UTK/FairFace ages 0-9, not 100+100 PA-100K), plus the face gate. Caps applied: 6ckvk child 600 (534 train + 66 val), baby-adult 250 (221 available).
Gender gate (face classifier v2 on the top 55% of the box, YuNet face >= 24 px, person >= 60 px high, live crop margin 0.1): >= 0.9 accepted, 0.7-0.9 review, < 0.7 dropped. Counts: no usable face 1,338 boxes; confidence < 0.7: 390; > 2 per group (ihdwp): 89; over per-source cap: 23; review crops dropped because the group already had 2 crops: 381. Accepted: ihdwp 141 (male 133, female 8), 6ckvk 144 (64 / 80), baby-adult 9. **ihdwp is almost all men (masked, 2020 taxi): women stay scarce.**
Review: `05_gender_review.csv` + `05_gender_review_sheet_1..3.jpg` = **178 crops** (not ~330: the 2-per-group cap removed most; ihdwp 95, 6ckvk 79, baby-adult 4). Fill the last column (male/female/skip). Crops are in `crops_mfc_v1/review_pending/` and in no split. Sheets show ids only, not the prediction.
Spot check `05_autoaccept_spotcheck.jpg`: 29 of 294 face-accepted crops (10%) viewed by me: 1 clear gender error (bald man with mask and sunglasses read female 0.91) and 2 possible teens or children among "adults". About 3% visible error, no tight estimate. 6ckvk adult labels are partial and may include teens; nothing filters age except the dataset's own adult label.
Many crops show replicate-padding streaks (tall thin boxes padded to square).

PA-100K pretraining (`pretrain_pa100k/`): train + val csvs only; Age18-60 + AgeOver60 (AgeLess18 excluded entirely; teen-looking people inside Age18-60 cannot be detected and are not removed); Front/Side 80%, Back 20%; top 45% of image height; near-twins (corr > 0.97) dropped during selection (male 55, female 66); AgeOver60 kept up to 12.5% per stratum (male 728, female 388, because the pools hold only 822 / 389, so the audit's "about 1,500" is not reachable); holdout (1,000 + 1,000) chosen by cluster at corr > 0.9. `check_official_test`: 2,000 adult crops from the official test split without a twin of a chosen crop (4 dropped). PA-100K is not in train/val/test of the fine-tune set; no synthetic data used.

## Imbalance and limits for the trainer
- Detector train: ihdwp 358 (one taxi fleet, masks), stock photos 563 (watermarked, eye level or close-up), car_cabin 134 (3 videos), 40 negatives. The gap to a ceiling camera is large; car val/test are the only in-domain data.
- Crops train: male 286 / female 203 / child 771 (child 61%); female is the weak class. Val child (111) is 80% stock photos while test child is 3 children in one video, so val child does not mirror test. PA-100K pretrain is balanced 5,000 / 5,000 but outdoor and adult only (mask the child output).
- Licences: CC BY as stated by uploaders, content rights unverified; internal use. Stock watermarks (Envato / iStock) are in car_cabin and in the web photos.
- The leakage check is 32x32 correlation (max 0.936 detector, 0.920 crops); another frame of the same stock clip would not be caught.
- QA gate: the evaluation numbers did not change in this run (same 509 frames as the previous row; I registered no frames). The earlier change came from the car frames registered in the previous task. A first gate run under heavy load failed (see WORKFLOW.md); the idle rerun is green.
