# 02 Dataset audit (2026-10-28)

Target: one car-cabin YOLO, classes male, female, child, seatbelt, no_seatbelt, child_seat (docs/CAR_MODEL.md, config passenger.car).
Method: contact sheets with boxes drawn (about 12-36 images per set, plus label statistics over every label file), near-duplicate check by 8x8 dHash (Hamming <= 6), tar listing for UTKFace. Nothing was extracted into raw/, nothing modified. Scripts live in the scratchpad.
Limits: no gender or age ground truth exists for the car sets, so gender/child label accuracy could not be measured. Belt-label correctness was only judged by eye on 12 images. The existing champion (models/passenger/registry.json v0/v1) is a person detector on Open Images; it has no per-class car scores, so every car class is unmeasured today.

## Verdict table

| Dataset | Verdict | One line |
|---|---|---|
| hf_ramankamran_seatbelt_yolov11 | PARTIAL (weak; belt-state head start only) | Roadside photos of cars from outside, driver only, about 240 distinct scenes |
| hf_c3rl_seatbelt_detection | REJECT | Procedurally drawn flat cartoons, no boxes |
| hf_slownite_child_detection | PARTIAL (child appearance only, NC licence blocks commercial use) | Real children, but not in cars |
| hf_py97_utkface_cropped | PARTIAL (face-attribute stage only, NC licence) | Faces, not cabin; useful only if a face stage is added |
| roboflow_inside_bus_view_v1 | REJECT for the car model (keep for the bus pipeline) | Bus interior, seat occupied/unoccupied, no gender/child/belt |
| car_cabin_videos_stock | PARTIAL (unlabelled; qualitative test clips only) | Real car cabins, watermarked stock, rights unclear |
| sviro_intercar_subset | PARTIAL (child_seat only, after pseudo-boxing; synthetic) | Rear bench renders, seat codes, keypoints, no boxes |
| sviro_intercar_people_subset | PARTIAL (person/child on rear bench, after pseudo-boxing; synthetic) | Same, no keypoints, adults only mostly |
| fairface_hf_025 (context) | already ingested, face classifier only | Not rechecked here |
| open_images_v7_* (context) | already ingested | Man/Woman/Boy/Girl in general scenes; weak for cabins |

## hf_ramankamran_seatbelt_yolov11
Evidence
- Roboflow export, 640x640, classes person-noseatbelt (0) / person-seatbelt (1). train 2548 / valid 246 / test 310 images. Boxes: train 1437 seatbelt + 1089 noseatbelt, valid 102 / 144, test 186 / 124. 22 train images empty. Exactly one box per image.
- Domain: NOT a cabin camera. The camera stands at a roadside and films cars driving past from outside; the box is the driver seen through the windshield or side window (median box 1% of the image, p10 0.4%, so roughly 60x60 px or smaller; many only 20-40 px). Daylight only, no passengers, no rear seats, no IR. Camera angle is unlike a dashboard/cabin camera, so a model trained on it learns "belt state from outside", which does not transfer to an inside camera.
- Diversity is poor: the sheets show the same handful of cars (red Clio, white sedan, dark hatchback, black Fiesta) and two street backgrounds. Distinct scenes by dHash clustering in train: about 240 of 2548. Roboflow augmentation produced the rest (3 rf.* variants per photo; md5 shows 38 exact duplicates in train, 7 valid, 6 test).
- Leakage inside the set: source filename prefixes do not overlap across splits, but 86 of 310 test images and 19 of 246 valid images are near-duplicates of train images (same car, same frame sequence). Its own test split is therefore inflated.
- Label noise: whether a belt is worn is hardly visible from outside through glass. In the 12 images viewed, the belt is visible in none of them at this scale; the labels look like per-clip annotator decisions. Treat as noisy. The scout's CC BY 4.0 is the Roboflow author's statement; the HF re-upload has no licence (unverified).
- Marginal value: seatbelt / no_seatbelt, but as "person with belt", not a belt box. Our schema wants seatbelt as a belt box across the occupant. The data has no belt boxes.
Selection rule for the cleaner (PARTIAL, low priority, ship only if the user wants an early belt signal)
1. Group by dHash cluster (Hamming <= 6) across all three splits; discard the original Roboflow splits. Keep at most 5 images per cluster, preferring the largest box.
2. Drop images with box area < 0.2% of image (17 in train) and all empty images.
3. Class-map for a car model is NOT direct: person-seatbelt -> mark "belted driver", person-noseatbelt -> "unbelted driver". Do not convert into seatbelt/no_seatbelt box classes (the box is the person, not the belt). Preferred use: a separate image-level belt-state classifier on driver crops, or relabel after the user's own footage exists.
4. Expect about 240 clusters x up to 5 = at most about 1,000 images; realistic yield about 700 (about 400 seatbelt, 300 noseatbelt). Hold the whole dataset OUT of the test split of any car model (domain mismatch would mask real performance).
Licence: CC BY 4.0 (attribution to Roboflow user class-oohmp required; HF copy licence-unverified).

## hf_c3rl_seatbelt_detection
Evidence: 5000 jpg, 2500 seatbelt / 2500 no_seatbelt folders, image-level only. 36 sampled images: flat coloured rectangle torso, circle head, diagonal bar for belt on a coloured background. Not photographic, generated by a script; the "no belt" images carry random distractor shapes. A model learns bar-detection on cartoons.
Verdict: REJECT. Zero transfer to cabin video, negative value if it dilutes real data. The user may delete it (176 MB) to free budget.
Licence: MIT per metadata, card empty (unverified). Moot.

## hf_slownite_child_detection
Evidence: train 1928-826=1102 images / 204 videos, test 826 images / 155 videos, no video in both (train/test are already video-disjoint). 2,534 child boxes (all class 0), median box 26% of the image, only 3 boxes < 2%: close-ups, not cabin-distance. Scenes: kitchen, bed, play, outdoors, baby close-ups, medical; Pexels stock keyframes. None in a car or a car seat. 36 of 1928 images have no box. Sample showed missed children in some multi-person frames (one baby with an unlabelled box in a crowd frame), so recall of the labels is imperfect. Infants through adolescents (to age 19) are in the same class; the "child" cut-off is therefore not our project cut-off (see docs/ANNOTATION.md).
Value: only for the appearance of children (proportions) in a general detector; it will not teach seat context, belts, or the camera angle. Risk: pure close-ups bias toward large boxes.
Selection rule for the cleaner (PARTIAL; training only, non-commercial)
1. Use as `child` boxes only. Keep boxes with area between 0.5% and 60% of the image; drop images with no box.
2. Group by video_<id> (already in filename); cap at 3 keyframes per video; do not mix videos across splits. Put all into train; never use as test for the car model.
3. Mix at no more than about 20% of the child boxes in a training set so the model does not learn "close-up = child".
4. If the model will be released commercially, exclude this dataset (CC BY-NC-SA 4.0, share-alike; derived weights arguably inherit the condition). Flag as a blocker for release, fine for internal research.
Yield: about 1,100 train + 800 test images, roughly 600 images after the video cap.

## hf_py97_utkface_cropped
Evidence: tar.gz listing only (not extracted). 23,708 aligned face crops, 3 names unparsable. Gender 12,391 male / 11,314 female (balanced). Age bins: 0-2: 1,605; 3-12: 1,808; 13-17: 820; 18-29: 7,704; 30-49: 6,781; 50-69: 3,615; 70+: 1,372. Child (<=12): 3,413; (<18): 4,233. Race counts: White 10,078, Black 4,526, Asian 3,434, Indian 3,975, Other 1,692. Face crops, frontal, adult-heavy, tight crop with no hair or clothing context. Not cabin, not whole body. The original is research-only/non-commercial.
Overlap with FairFace: not checked by hash (different sources: UTK web crawl, FairFace YFCC); expect small overlap, but run a dHash dedup in the cleaner.
Value: only useful if the design adds a face-attribute stage (the docs name this as the fix when body-level male/female is weak) or to rebalance child faces for the bus face classifier. Not for the car detector.
Selection rule (PARTIAL; face classifier only, never detector)
1. Extract with passenger/datasets/utkface.py into a new empty directory; parse <age>_<gender>_<race>; drop files that do not parse.
2. child = age <= 12 (match docs/ANNOTATION.md; adjust here if the project cut-off differs); male/female only for age >= 18; drop 13-17 (ambiguous) from the gender label.
3. Cap each class to match FairFace-balanced counts; dedup against fairface_hf_025 by dHash <= 6.
4. Keep out of every held-out test split of the face classifier unless the test is taken from the user's own frames.
Licence: NC research, HF "MIT" tag unreliable. Do not ship commercially.

## roboflow_inside_bus_view_v1
Evidence: 3,303 images (train 2888 / valid 274 / test 141), classes Occupied (0) and Unoccupied (1). Boxes train 69,052 Occupied + 39,205 Unoccupied (about 24 and 14 per image), valid 6,546 / 3,706, test 3,113 / 2,171; 193+13+6 polygon rows (not plain boxes). Median box 0.2% of the image, 57,420 of the train boxes under 0.2%; the "Occupied" box is a person (face/torso) and "Unoccupied" is a seat. Real bus interior, daylight, tall overhead camera, mostly Philippine passengers, clean timestamp overlay. Train 2888 images are 963 source photos, 3x augmented; valid and test have unique images (no source overlap).
Fit: great for the bus person/seat counter (it is the "Inside Bus View" candidate in DATASETS.md); no gender, child, belt or child seat; buses do not use belts here. For the car model, a person is only "Occupied", not male/female/child, so none of the 6 classes are labelled. Using it as `person -> male` would poison the gender classes.
Verdict: REJECT for the car model. For the bus detector: ADOPT separately (split by source id in train, keep valid/test as given), but that is outside this task.
Licence: Public Domain per its own data.yaml (verify on Roboflow page).

## car_cabin_videos_stock
Evidence: 12 mp4, total about 4.5 minutes (12-32 s each; 960x540 except one 4096x2160 and one 768x432; 25-50 fps). 4 frames viewed: front-seat drivers and passengers in real cars, one dark night/IR-like clip with Envato watermark, family with two children and a child in the back (belted driver and passenger visible). Envato and iStock watermarks in the frames. Real in-car footage with day and low-light, visible belts and children, which is exactly the missing domain. But: no labels, tiny volume (about 6,700 frames, only a few scenes, near-identical frames), watermarks (a model learns the watermark) and no usage rights.
Verdict: PARTIAL, not for training. Use as qualitative smoke-test clips for the car pipeline and as a source of frames to pseudo-label with the future model for the user to review. Do not add to train/test unless the licence is cleared.
Licence: none; stock content, assume all rights reserved.

## sviro_intercar_subset (and sviro_intercar_people_subset)
Evidence
- 42 scenes x 10 cars = 420 images each (960x640 png), rear bench, three seats, ceiling-ish front view of the back seat. Subset folder codes (0 empty, 1 infant seat, 2 child seat, 3 adult). InterCar: 42 scenes (seat code counts: a handful each of 1, 2 and 3). 2D keypoints for the persons (InterCar only).
- Quality: renders are synthetic with quite dark and sometimes almost black cabins (the dark-skin or dark-clothing cases are nearly invisible), humans are CGI mannequins with fixed outfits (striped shirt repeated), the same scene is repeated across 10 car models (so 420 images are 42 distinct layouts). Child seats and infant seats are clearly visible in the renders, so `child_seat` is the one class that the real world has no other source for.
- No boxes, no gender, no belts. Seat code alone gives image-level presence and seat position, not a box. Belts are visible on the CGI (seen on the tiger-striped adults) but not labelled.
- Domain: rear bench only, nothing like the front-camera of a typical EagleEye install; synthetic-to-real transfer for child seats is unproven (docs/CAR_MODEL.md already says synthetic alone will not transfer).
- People subset: same layout, adult codes dominate (3 counts), almost no child codes, so no child boxes; value is `person` occupancy only.
Selection rule (PARTIAL)
1. Derive boxes with the keypoints (InterCar) or a one-off pseudo-labeller: a seat-code of 1 or 2 gives a child_seat box covering that seat's keypoints bounding box expanded by 20%; discard when the keypoints are missing. Count the people subset only for person boxes (class not assignable: drop it unless the user confirms a pseudo-label step).
2. Split by scene id (imageID), never by image; the same scene in 10 cars all go to one split. Never put SVIRO in the test split of the car model (synthetic test numbers are meaningless for the real camera).
3. Cap SVIRO at 30% of the child_seat boxes; the rest must be real.
4. Yield after rule 1: about 14 infant-seat + 14 child-seat scenes x 10 cars = at most about 280-420 child_seat boxes, 42 distinct layouts only (low diversity). Reaches the 200-box warning threshold in stats but is mostly the same seats.
Licence: none stated, believed CC BY-NC-SA 4.0 (SVIRO paper); unverified. Treat as non-commercial.

## Per-class summary (target classes)

| Class | Best source in hand | Real data? | Adequate? |
|---|---|---|---|
| male / female | Open Images (Man/Woman, general scene), FairFace/UTKFace (faces, NC for UTK) | no real in-car body boxes | No |
| child | slownite (NC, not in car), OI Boy/Girl, UTK faces | no in-car | No |
| seatbelt | none usable (ramankamran is person-level, driver from outside) | no belt boxes anywhere | No |
| no_seatbelt | ramankamran (person-level) | partial | No, and optional per CAR_MODEL.md |
| child_seat | SVIRO subset (synthetic) | no | No |

## Ranked list (most useful for the car model first)
1. sviro_intercar_subset (only child_seat source; after pseudo-boxing, synthetic, NC-likely)
2. hf_slownite_child_detection (child appearance, NC)
3. hf_ramankamran_seatbelt_yolov11 (driver belt state from outside, noisy, 240 scenes)
4. car_cabin_videos_stock (unlabelled test clips only, no training rights)
5. hf_py97_utkface_cropped (face stage only, NC)
6. sviro_intercar_people_subset (little beyond subset 1)
7. roboflow_inside_bus_view_v1 (bus, not car; adopt for the bus model only)
8. hf_c3rl_seatbelt_detection (reject, delete)

## Remaining data gaps (what the scout or the user must find)
- Real in-cabin images with male/female/child on whole bodies from a cabin-mounted camera: none. Only the user's own footage or a licensed set (e.g. TICaM after login, DMD) can fill it.
- Real child_seat boxes: zero real images. Search Roboflow Universe "car seat child" / "baby car seat" (CC BY) and Kaggle; check that the camera looks inside the car.
- Belt boxes (not person boxes) on a worn belt, with passengers, from inside the car: none. Look for Roboflow "seatbelt" sets with box labels of the strap; the ramankamran one is outside-in.
- No IR/night in-cabin labelled data (only one unlabelled dark stock clip).
- Rear-seat adult/child bodies in daylight: only synthetic.
- Licence blockers: slownite (CC BY-NC-SA), UTKFace (NC), SVIRO (probably NC), stock videos (no rights). A commercially shipped car model cannot use any of these without clearance; Roboflow ramankamran (CC BY) and Open Images (CC BY) are the only permissively licensed bodies of data here.
- The practical path: a small labelled set (a few hundred frames, all six classes) from the user's real camera, using car_cabin_videos_stock-style clips as the labelling pilot.

## Hand-off note for the cleaner
Do not add anything to the car training set before the user decides on the licence question. Safe to prepare (no commercial claim): (a) slownite child boxes per the rule above, (b) SVIRO child_seat boxes via keypoints, (c) ramankamran clusters as a separate belt-state classifier set. Skip c3rl, bus set (for car), and the stock videos. The test split of any car model must come from real, in-car frames of the user's own footage; none exists yet.


# Round 2: Kaggle and Roboflow sets (2026-10-28)

Owner decisions applied: not a commercial release, so NC licences are recorded but do not cause rejection; no own cabin footage exists, so public data is the only source for now. Licence-unverified web-scrape data is flagged "internal training only, do not redistribute the data or publish weights trained on it"; the owner should confirm that is acceptable.

Method (all read-only, `python -I`, scripts in scratchpad/r2): contact sheets with boxes drawn (36 images per set from distinct near-duplicate clusters, stratified by class; more for habbas11 and firasmahdi, see below; all 5 + 8 images of nexdata and alexandresintes), label statistics over every label file, near-duplicate detection by 32x32 grey normalised correlation > 0.97 (the 8x8 dHash of round 1 chained dark IR frames into giant clusters here, so it was replaced; cluster counts below use the new measure; ramankamran re-measured the same way gives 134 chained clusters at this threshold and 455 at dHash Hamming <= 2, so round 1's "about 240 scenes" is the right order of magnitude), within-set clusters and cross-set matches including against hf_ramankamran_seatbelt_yolov11. Pairs of cross-set matches were viewed to confirm they are the same photos.
Limits: no gender/age truth; belt correctness judged by eye only; no model trained, so every "usable count" is an estimate.

## Verdict table

| Dataset | Verdict | One line |
|---|---|---|
| kaggle_habbas11_dms_seatbelt_subset | PARTIAL (best belt-box source, adopt class 4 only) | 3,939 images with class 4 Seatbelt boxes; includes rear passengers, children in seats, IR fleet-cam frames; heavily clustered |
| kaggle_firasmahdi_driver_seatbelt_ir | PARTIAL (belt boxes only, 289 images) | Driver cabin, strap boxes of mixed tightness; no no-belt class; an absent box is NOT a negative |
| kaggle_sachinx_seatbelt_positives | PARTIAL (via habbas boxes, plus child_seat hand-labelling pool) | 1,300 unique web photos of belted people, children, infant seats; no labels; squashed to 256x256 |
| kaggle_youcef_seatbelt2_subset | REJECT for training (manual review queue only) | Class is confounded with image size/source; labels conflict with other sets |
| kaggle_lavdeep_driver_seatbelt | REJECT | 1,177 of 1,264 are 128x128 copies of sachinx; the 87 negatives are unverifiable |
| kaggle_mariam_noseatbelt_truckcab | REJECT | One truck fleet, overhead fisheye cab camera, one class, person boxes, rotated duplicates |
| roboflow_seatbelttraining_seatbelt_v1 | REJECT | 15 YawDD subjects, heavy augmentation, person boxes; "no-seatbelt" is 2 people |
| kaggle_nexdata_passenger_behavior_sample | PARTIAL (smoke-test views only, no training) | Right viewpoint (rear passengers, RGB+IR) but 5 PNG montages, unlabelled, CC BY-NC |
| kaggle_alexandresintes_seatbelt_real_car | REJECT for training (smoke test only) | 8 images, one person, one car |

## kaggle_habbas11_dms_seatbelt_subset
Evidence
- 3,939 images (train 2,040 / valid 1,461 / test 438), mostly 640x384, 640x360, 640x480. Label rows: class 4 Seatbelt 4,149 (every image has at least one), Open Eye 4,912, Closed Eye 825, Phone 29, Cigarette 1. No polygons, no empty images. Belt boxes per image: 1 in 3,772, 2 in 139, 3 in 21, 4+ in 7. Belt box area median 10.7% of the image (p5 1.8%, p95 37.6%), aspect w/h median 0.95; 14 boxes < 0.2%.
- Origin mix (file name plus content match against sachinx): 1,893 fleet-cam frames (`CHANNEL_05_2022...`, a cab camera with a GPS overlay, about 90% grey/IR, curtains-and-bus scenes, one man in a red shirt over many clips), 904 frames from mp4 clips (`_mp4` in name), 785 greyscale copies of the images.cv web photos that are in sachinx (verified by pair view), 357 other web photos.
- Duplication: 1,434 clusters over 3,939 images; the largest hold 332, 215, 141, 107 images (fleet cam, same driver, same minute). Fleet cam: 188 clusters for 1,893 images; mp4 frames: 105 clusters for 904. The supplied splits are unusable for evaluation: 94 clusters span more than one split (2,488 images).
- Viewed 96 images (36 random from distinct clusters, 24 with 2+ belt boxes, 36 non-fleet) and about 130 belt boxes. Boxes cover the strap or strap plus torso; tightness ranges from good to loose (long shoulder-to-waist boxes); a few boxes cover a belt on an empty seat or the harness of an empty child seat (belt present, not worn); rear passengers appear (three-person back seats, two adults in the rear, children in child/booster seats with harness boxes: about 100 such images among the 167 with 2+ belt boxes); dogs and a screwdriver appear as one-offs. Roughly 85-90% of the boxes checked sit on a real belt, the rest are loose or off.
- Domain: real cabins; IR/night via the fleet cam; close-up phone selfies via the web part; passengers and children present but not labelled as such. No no_seatbelt class, and an unboxed person is not proven unbelted: the set gives positives only.
Fit per target class: seatbelt YES (detector boxes including passengers, rear seats, grey/IR); no_seatbelt NO; child_seat NO labels (child-seat photos exist, see the sachinx hand-labelling pool); male/female/child NO.
Selection rule for the cleaner (PARTIAL)
1. Keep class 4 only, map to `seatbelt`; drop classes 0-3. Drop belt boxes < 0.3% or > 60% of the image.
2. Dedupe with the normalised 32x32 grey correlation > 0.97 (`scratchpad/r2/a4.py` logic), within habbas and against sachinx, firasmahdi, ramankamran. Group id = cluster id; for fleet-cam frames (`CHANNEL_05_<timestamp>`) also group by the 14-digit timestamp (take the union).
3. Per cluster keep at most 3 frames for fleet-cam and `_mp4` frames, 1 for the others. Expected: about 188x3 + 105x3 capped to about 600 fleet/mp4 frames, plus up to 1,142 web frames (785 grey copies of sachinx + 357 others). Cap the grey-copy web group at about 50% of the final selection because those are close-up selfies, not a cabin camera. Realistic yield 1,200-1,600 images, 1,400-1,900 belt boxes.
4. Discard the supplied train/valid/test; re-split by group. Never use it as the only test for the car model (90% grey and one fleet camera dominate).
5. Spot-check 100 kept boxes before training; treat unboxed people in these images as unlabelled (missing-label risk), see the belt plan.
Licence: Apache 2.0 per Kaggle; the web photos inside are scraped (rights unverified); the fleet frames show real drivers and GPS stamps (privacy: internal only).

## kaggle_firasmahdi_driver_seatbelt_ir
Evidence
- 1,321 images (train 1,189 / test 132), 640x480 (862) and 640x640 (459); label rows 2,391: Hands off Wheel 740, Phone 651, Hands on Wheel 291, Seatbelt-On (5) 297, Drinking 262, Eating 82, Smoking 67, Sleeping 1; 8 polygon rows; no no-belt class. 289 images carry a class-5 box (208 clusters); 1,032 images have none.
- Belt box area median 1.9% (p5 0.3%, p95 10.4%), aspect median 0.64 (strap-shaped); 7 boxes < 0.2%.
- 601 clusters over 1,321 images (max 104); 11 belt clusters span train/test. 37% grey in a 200-image sample. Augmentation visible: 90/180 degree rotations with black borders, brightness changes. About 20 images are the same web photos as in sachinx/habbas, 8-12 match youcef.
- Viewed 36 random + 30 belt-box images (about 45 belt boxes): about 70% tight on the strap or the belt across the torso, 20% loose (a chest box not on a visible strap), 10% doubtful (a belt on a skeleton prop, boxes on faces). Second persons with a visible belt are often unboxed.
- Domain: dashboard/side/face-on cameras, stock-watermarked frames, mostly the driver doing a behaviour; 90% of the rows are behaviour classes irrelevant to us.
Fit: seatbelt YES (driver-centric, little passenger data); no_seatbelt NO; an image with no class-5 box must NOT be used as a negative.
Selection rule (PARTIAL)
1. Keep only images with a class-5 box (289), map to `seatbelt`, convert the 8 polygon rows to boxes; drop all other classes.
2. Dedupe as for habbas11; at most 2 per cluster; drop boxes < 0.3%; prefer unrotated variants (exclude images with large black borders).
3. Expected yield about 180-200 images / about 200 boxes; discard the supplied split and re-split by group.
4. Do not use the other 1,032 images as negatives, nor the behaviour classes (out of scope).
Licence: none stated (Kaggle: Unknown); stock watermarks present (rights reserved). Internal training only.

## kaggle_sachinx_seatbelt_positives
Evidence
- 1,300 images, all 256x256 (originals squashed to a square, bodies look stretched), no near-duplicates inside the set, supplied split train 767 / val 133 / test 400. One folder `seat_belt`: image-level positives only. Source images.cv (web scrape).
- Viewed 36: selfies and close-ups of a belt across a person (about 60%), pets (about 8%), a belt without a person, children and infants in car seats or boosters (6 of 36, so roughly 150-200 in the whole set), a few belts in buses/planes. Not a cabin-camera viewpoint; mostly handheld photos from the seat.
- It is the same photo pool as the 1,177 `seat_belt` images in lavdeep (128x128 copies, verified by correlation) and the 785 greyscale copies inside habbas11, which already carry belt boxes.
Fit: seatbelt image-level only; child_seat: real photos of children in harness seats and infant carriers, but unlabelled; no no_seatbelt, no male/female/child labels.
Selection rule (PARTIAL)
1. Do not train on the folder label. Use it (a) as the colour source for the 785 images habbas11 already boxed (keep one copy only; keep the habbas box; use the colour copy only if the box can be transferred); (b) as a hand-labelling pool of about 150-200 images with a child seat or a child in a harness to get real `child_seat` (and `child`) boxes; (c) the remaining images for a pseudo-label pass by the belt detector with human review.
2. Drop pet-only and belt-only images from the child pool; keep 256x256 images out of any test split unless relabelled.
Licence: none stated; scraped web photos, rights reserved. Internal training only.

## kaggle_youcef_seatbelt2_subset
Evidence
- 1,370 images (every 6th/8th of 8,617): Seat_Belt 717, WithoutSeat_Belt 653, image-level; train 1,213 / test 157.
- Class confound: all 717 Seat_Belt images are 640x480 (one shoot: a side camera on a driver in the same grey interior; 265 clusters, largest 155 frames); only 49 of the 653 WithoutSeat_Belt images are 640x480, 604 have other sizes (256 files `train_witout_seatbelt_ori` at 612x408/848x464, stock/movie frames; 125 `withoutseatbelt...`). A classifier would learn the source (frame size, camera, interior), not the belt.
- Label conflict: 5 youcef WithoutSeat_Belt images (pair view) are the same YawDD driver frames that roboflow_seatbelttraining labels as seatbelt; the strap is visible in at least two. 3 exact md5 copies are shared with that set.
- Viewed 36: the WithoutSeat_Belt examples are mostly drivers whose belt is just not visible (smiling driver, hijab covering the torso, a driver seen from outside); several are watermarked stock frames. 735 clusters at corr > 0.97; 39 clusters (448 images) span train/test.
Fit: seatbelt (image-level, driver side view) weak; no_seatbelt unreliable.
Verdict REJECT for training. Review-queue option: the 470 distinct-cluster WithoutSeat_Belt images could be hand-checked (about 1 hour) to extract a verified "driver, belt clearly absent" set; expect well under 100 to pass (rule: torso fully visible, no belt visible).
Licence: Apache 2.0 per Kaggle; the content is stock/movie frames, rights unverified.

## kaggle_lavdeep_driver_seatbelt
Evidence: 1,264 images; seat_belt 1,177 (all 128x128; train 811 / test 366) and no seatbelt 87 (train 54 / test 33; 390x280 to 1600x1155). All 1,177 seat_belt images match a sachinx image at corr > 0.97 (same pool, lower resolution), 710 also appear as grey copies in habbas11, 34 exact md5 duplicates inside the set, 33 clusters span train/test. Of 36 viewed, the seat_belt class includes non-car images (a toy buggy, a blanket, an aircraft cockpit) and the "no seatbelt" images are stock photos of drivers through the window from outside with the belt not visible. Class ratio 13.5:1.
Verdict REJECT: zero unique positives, negatives unverifiable. The 87 negatives could be reviewed by hand in about 15 minutes if the owner wants hard negatives.
Licence: none stated (Unknown), scraped.

## kaggle_mariam_noseatbelt_truckcab
Evidence: 1,587 images, all 512x512 (train 1,387 / valid 200), 1,643 rows, a single class `no_seatblet_detected`, every row a polygon (converted to boxes: median 7.7% of the image, p10 5.7%, p90 9.5%). 1,000 source images with up to 2 variants each, 919 clusters at corr > 0.97, 144 clusters span train/valid. Fisheye ceiling cab camera in lorries from one fleet: the same cab, same background, few drivers; about half the images are rotated 180 or 90 degrees so people hang upside down. The box is the whole person (driver or sleeping co-driver), not a belt; there is no belted class.
Viewed 36: no belt discernible at this scale; the viewpoint and seat geometry are unlike a passenger car.
Verdict REJECT: wrong vehicle, one fleet, no positives, person-level label.
Licence: Roboflow README says CC BY 4.0 (Kaggle says Apache 2.0); either is fine.

## roboflow_seatbelttraining_seatbelt_v1
Evidence: 1,629 images 416x416 grey (train 1,425 / valid 135 / test 69), class 1 seatbelt 1,488 boxes, class 0 no-seatbelt 141, always one box per image, median 35-38% of the image, covering head and torso (person-level, not a belt). File names (`2-FemaleNoGlasses`, `8-FemaleGlasses_avi`, ...) show YawDD driver videos: only 15 distinct subjects, 278 sources, up to 16 augmented variants per source (90/180 rotation, salt noise), 461 clusters, 57 span splits. The 141 no-seatbelt boxes come from two subjects (`4-FemaleNoGlasses` 98, `7-MaleGlasses` 43); in the 24 class-0 images viewed no belt is visible, and the same woman appears in about 20 of them, rotated and noisy. The class is a subject identity, not a belt state.
Verdict REJECT: person-level boxes, 15 subjects, label conflict with youcef.
Licence: CC BY 4.0 (YawDD asks for a citation).

## kaggle_nexdata_passenger_behavior_sample
Evidence: 5 PNG files of about 1033x499 RGBA, each a montage of RGB+IR pairs (about 8 tiles per PNG, about 20 pairs in total) from a camera looking back over the front seats: drivers and rear passengers, day and IR, belts visible on drivers and some rear occupants, standing people at a rear door, a woman reading in the dark. No labels. CC BY-NC 4.0, a sample of a commercial set.
Right viewpoint, includes rear seats, IR and RGB, but tiny.
Verdict PARTIAL: smoke-test views only (visual check of person detection, belt visibility, IR behaviour), no training, no metrics. Crop the tiles by hand if used.

## kaggle_alexandresintes_seatbelt_real_car
Evidence: 8 images 640x640 (one black-padded), one man in one car with the belt worn, 5 clusters. No labels. CC BY 4.0.
Verdict REJECT for training; keep as a smoke test for the belted-driver case.

## Cross-set overlap (corr > 0.97; images in A with a match in B)
- lavdeep to sachinx 1,177 of 1,264; lavdeep to habbas 710; sachinx to habbas 785; lavdeep/sachinx to firasmahdi 19/20; habbas to firasmahdi 17; youcef to firasmahdi 12 (8 on the firasmahdi side); youcef to roboflow_seatbelttraining 5 (49 on the roboflow side, plus 3 md5 copies).
- No overlap with hf_ramankamran_seatbelt_yolov11 from any of the nine sets, so it neither contaminates nor duplicates them.
- Net unique photos in the image-level web belt pools (lavdeep + sachinx + habbas web) is about 1,300 + 357 = 1,650, not 1,264 + 1,300 + 785.

## Effective belt data after the rules (estimates)
| Source | belt boxes | no-belt | passengers/rear | child in seat | IR/grey |
|---|---|---|---|---|---|
| habbas11 (rule above) | 1,400-1,900 | none | some (100+ images) | about 100 images, unlabelled as seat | about 90% grey |
| firasmahdi class 5 | about 200 | none | rare | none | 37% grey |
| sachinx (hand-label pool) | 0 until labelled | 0 | few | 150-200 photos for child_seat | no |
| ramankamran (round 1) | 0 belt boxes (about 700 person images) | outside-in only | none | none | no |

## Re-evaluation of round-1 PARTIAL verdicts with non-commercial licences allowed
- hf_slownite_child_detection (CC BY-NC-SA): the licence blocker goes away; usefulness is unchanged because the problem was domain, not licence. Still PARTIAL: child boxes only, at most 3 keyframes per video, about 20% cap on child boxes, train only. Best use is a pretraining stage for child appearance before fine-tuning on cabin images; it gives no belts or seat context.
- hf_py97_utkface_cropped (NC research): the licence blocker goes away and it becomes the largest labelled source of child faces (3,413 aged 0-12). Promote to ADOPT for a face-attribute stage (male/female/child on face crops of detected occupants) if the owner takes the two-stage design that CAR_MODEL.md names as the fix for weak body-level gender. Rule as in round 1 (child <= 12, gender for >= 18 only, drop 13-17 from gender, dedupe against FairFace). Stays PARTIAL if no face stage is built. Frontal studio faces do not match a ceiling-mounted camera, so fine-tune the face stage on cabin crops later.
- sviro_intercar_subset and sviro_intercar_people_subset (believed CC BY-NC-SA 4.0): the licence no longer blocks. This makes SVIRO the only child_seat source that can be scaled now: the 42-scene limit was a size and caution choice, not a data limit. Recommendation: fetch more scenes (the train split has 990 InterCar scenes; 42 scenes took about 470 MB, so 150-200 scenes is roughly 2 GB, within the budget) and apply the same rules (keypoint-derived child_seat boxes, group by scene id, never in test, SVIRO at most 30% of child_seat boxes). That lifts layouts from 42 to hundreds but does not fix the sim-to-real gap, so real child_seat boxes from the sachinx/habbas hand-labelling pool are still needed. The people subset stays low value (mostly adults).
- hf_ramankamran_seatbelt_yolov11: unchanged (licence was never the issue; outside-in, driver-only, about 240 distinct scenes).
- car_cabin_videos_stock: licence concern reduced for internal use (watermarks remain). Upgrade to "usable as unlabelled frames to pseudo-label and hand-review for internal training"; never as the final test set and never redistributed. It is the only real passenger-and-child cabin footage in hand.

## Belt plan for a mostly driver-only, image-level label world
Facts: no source has person boxes plus belt boxes in the same image; no_seatbelt has no trustworthy labelled source (ramankamran outside-in, mariam trucks, youcef confounded, roboflow_seatbelttraining two people); belt boxes (habbas, firasmahdi) exist, but the images carry no person/child/seat labels.
Recommendation: do NOT drop these images unchanged into the six-class car YOLO. Mixing sets that label only some classes teaches the detector that unlabelled people are background (missing-label problem).
1. Train a separate single-class `seatbelt` detector (YOLO n, 640) on habbas11 + firasmahdi after the rules above (about 1,400-2,100 boxes). Link belts to occupants at inference: a belt is "on" an occupant when at least 30% of the belt box lies inside the occupant box expanded downwards by 20%; `belt_on_ratio` over recent frames (CAR_MODEL.md) gives belted / not seen. `no_seatbelt` is derived (occupant torso visible for N frames, no belt linked), not trained, which matches the doc's own note that a never-seen belt reads NO BELT. Drop `no_seatbelt` from the trained class list until verified negatives exist; its test needs real labelled negatives either way.
2. Later merge: auto-label persons in the belt images with the existing person detector and the face-attribute stage (UTKFace/FairFace), review about 100-150 frames by hand, then fold belts into the single car model with complete labels. Merging matters for edge cost: a second YOLO-n at 640 may roughly double belt-stage latency, but the Pi speed is unmeasured, so measure first.
3. Add-on: a belt-state classifier on occupant crops (belted / not belted / unsure) for rear passengers whose thin belts the detector may miss. Its training data does not exist yet: positives can be cut from the habbas/firasmahdi boxes; verified negatives only from hand review of youcef/lavdeep negatives (probably under 150) and from reviewed pseudo-labels on stock cabin video. Build it only if step 1 shows a rear-seat recall problem.
4. Evaluation: group-split by cluster; test frames must come from a source the training set never saw. All public sets look different from each other, so hand-label about 150-300 frames of free-licence car-interior video (see gaps) with belt, child and child-seat boxes as the test set. Do not report the car metric from habbas11's supplied test or the grey fleet-cam frames.
5. Child seat: hand-label about 150-200 real photos (sachinx and habbas "other" web photos) plus SVIRO-derived boxes (cap 30%); the real test set from step 4.

## Remaining gaps and candidate public sources
- Real male/female/child person boxes in cabins: nothing labelled. Public routes: (1) free-licence stock car-interior videos (Pexels, Pixabay, Mixkit, Wikimedia Commons CC clips; explicit licences, no watermark) plus the stock clips already in hand, sampled at about 1 fps and hand-labelled to the six classes (needs no own filming); (2) TICaM and TIMo (DFKI, account required, in-cabin ToF/IR with child seats) and DMD (registration); (3) a face-attribute stage fed by UTKFace/FairFace avoids body-level gender labels.
- Child seat boxes: SVIRO (synthetic, scalable now) and the hand-labelling pool above; Roboflow `occupant_classification_model1/2` (adult_occupant, child_occupant, empty_seat, lapbelt) were not located by slug and need a manual Universe search; Wikimedia Commons "child safety seat" photos (CC) for installed-seat appearance.
- Belt boxes with passengers from a ceiling/dash camera: Roboflow `seatbelt_ilg` (datascince/seatbelt_ilg, listed as CC BY in docs/CAR_MODEL.md, never fetched or viewed: fetch by slug with the existing API script and check overlap with habbas11); Roboflow `jui/driver-behaviors` (9,901 images with seatbelt boxes, probably a duplicate of the DMS set: test overlap on a 200-image sample before a full download).
- IR/night in cabin: habbas11 fleet cam (single camera), firasmahdi, the 20 nexdata pairs. abdoreldeeb/driver-monitoring-system-dataset (Kaggle, about 2 GB, CC BY 4.0, unviewed) is probably a driver-monitoring set of the same family as habbas11 (the GPS-stamped fleet cam may be the same source). I have not seen it: fetch a 200-image sample first, run the correlation check against habbas11, view 36 images, and pull the rest only if it adds new cameras, rear seats or distinct drivers. Other leads: `smartcity12/adms-dataset` (928 MB, unviewed), `huynhtan/seatbelt001` (1.4 GB, unknown licence), `yehiahassanain/seat-belt2` (probably the same data as youcef, skip).
- Verified no_seatbelt: no usable source; hand-verify the youcef/lavdeep negatives (an estimated under 150 images) or derive it from absence (plan above).
- Rear-seat adults/children in daylight real photos: SVIRO is synthetic; habbas11 "other" web photos give a few dozen; the stock-video route is the main option.

## Updated ranked list (belt and child-seat relevance for the car model)
1. kaggle_habbas11_dms_seatbelt_subset (PARTIAL: belt boxes, rear seats, IR)
2. sviro_intercar_subset (child_seat; worth scaling up now that NC is fine)
3. kaggle_firasmahdi_driver_seatbelt_ir (PARTIAL: about 200 belt boxes)
4. kaggle_sachinx_seatbelt_positives (PARTIAL: hand-label pool for child_seat, colour source)
5. hf_py97_utkface_cropped (ADOPT if a face stage is built)
6. hf_slownite_child_detection (PARTIAL: child appearance pretraining)
7. car_cabin_videos_stock (unlabelled cabin footage to pseudo-label and test)
8. kaggle_nexdata_passenger_behavior_sample (about 20 smoke-test views)
9. hf_ramankamran_seatbelt_yolov11 (weak)
10. REJECT: kaggle_youcef_seatbelt2_subset, kaggle_lavdeep_driver_seatbelt, kaggle_mariam_noseatbelt_truckcab, roboflow_seatbelttraining_seatbelt_v1, kaggle_alexandresintes_seatbelt_real_car


## Round 3: real-car sets (2026-10-28)

Scope: rf_baby-adult-in-car_v1, rf_seat-belt-ihdwp_v1, rf_passenger-gender-dataset_v3, rf_child-detection-6ckvk_v4, kaggle_pa100k, judged for the two-stage design (Stage A person/occupant detector; Stage B male/female/child crop classifier), internal non-commercial training, CPU-only laptop. All licences CC BY 4.0 as stated by the uploaders, content rights unverified (recorded; acceptable for internal use).
Method (read-only, `python -I`, scripts in scratchpad `r3/`): contact sheets with boxes drawn (36 images per YOLO set, plus 36 child-only and 36 doll-only for 6ckvk; PA-100K: 36 stratified by gender x age x view, plus 36 AgeLess18 Front and 36 AgeLess18 Side), label statistics over every label file, 32x32 grey normalised correlation > 0.97 within each set (all 100,000 PA-100K images too), against the 311 car_cabin_v1 frames (most similar val/test pairs viewed), against all 494 extracted car frames, against the 14 earlier sets (21,582 images) and between the new sets. Source photo = file name before `.rf.`.

### Leakage and duplicates (headline)
- **No leakage into car_cabin_v1.** The highest correlation of any image of the four YOLO sets against any car_cabin_v1 frame is 0.46-0.64 (0.78 against all 494 raw frames); nothing above 0.9. The four most similar pairs per set against val/test frames were viewed: different scenes. Caveat: this measure only catches the same framing; a different frame of the same stock clip would be missed (none of the viewed images showed a car-vid person or the Envato watermark). PA-100K max 0.685.
- No match (> 0.97) against any of the 14 earlier sets, for any of the five sets.
- Between new sets: **29 of the 213 baby-adult images are the same photos as 30 images in 6ckvk** (viewed: identical stock photos, partly in different splits). Rule: drop these 29 from baby-adult, keep the 6ckvk copy, and use one split for both.
- Within-set at corr > 0.97: baby-adult 210 clusters / 213 images; ihdwp 1,001 / 1,035 (but see below); gender 379 clusters from 368 source photos / 1,110 images (Roboflow augmentation, about 3 copies per photo in train, 14 exact md5 duplicates, 2 source photos span splits); 6ckvk 2,704 clusters / 2,829 images, 2,684 source stems (about 5% augmented; 26 source stems span splits, 4 md5 copies).
- **ihdwp is video frames split by frame.** File names are `frameN-M-`; M (148 distinct values, up to 24 frames each) is the source clip; 89 of 148 clips have frames in more than one of train/valid/test, so its own val/test are not independent of train. Loose clustering (corr > 0.8) gives 293 groups; a handful of cars and about a dozen drivers repeat across clips. Group by M for every split and never use its supplied split.
- PA-100K: 4.4% of train images have a > 0.97 twin inside train (11% above 0.9: the same person in successive frames); **no val/test match to train** at > 0.97 or > 0.9. The supplied test has only 92 AgeLess18 crops, val 1,443: splits are not stratified by age, so make our own split by twin cluster.

### rf_baby-adult-in-car_v1: PARTIAL (child boxes only, re-boxed)
Evidence: 213 images 512x512 (train 171 / valid 42); boxes child 289 (train 235, valid 54), adult 23. Viewed 36: web/stock photos (iStock, Getty, brand ads) of infants and toddlers in child seats and rear seats, a few family shots, all inside cars in daylight; it gives real rear-seat child-seat appearance that no other set has. Child boxes are large (median 35% of the image, p10 17%) and usually include the child seat, not tight person boxes. Adults are mostly unlabelled (23 adult boxes in 213 images; about one viewed image in six shows an adult without a box): used as is for a person detector it teaches that adults are background. Child = infant to about 10 years; no teens seen. Viewpoint: close-ups from the front to the rear seat, much closer than a ceiling camera.
Rule: drop the 29 images duplicated in 6ckvk; keep the other 184 (about 250 child boxes); use only through the re-box rule below (a) and as child crops (b).

### rf_seat-belt-ihdwp_v1: ADOPT for the detector (subsampled); PARTIAL for Stage B
Evidence: 1,035 images 640x640 (850/124/61), 2,333 `passenger` boxes (2.2 per image, max 5), one class. Viewed 72 (two sheets): a taxi cabin camera (fisheye, dash level, looking back across the front seats to the rear seat) with timestamp and speed overlay, daylight and IR/grey frames (about 8% grey). Boxes are tight person boxes for the driver (labelled "passenger") and rear passengers; rear passengers are small (box area p10 3.5%). In 30+ spot-checked boxes no wrong box; the only miss is an almost fully hidden rear person. Almost everyone wears a surgical mask (2020 footage), the angle is oblique, mostly men; about 12 drivers and 4-5 cars over 148 clips.
Viewpoint fit: the best of all sets for the EagleEye cabin camera (wide view across the cabin, seated people, rear seats). Weakness: one fleet; 1,035 images are really about 150 clips of near-identical scenes.
(a) Rule: group by clip id M (union with corr > 0.9 clusters); at most 3 frames per group (first, middle, last), prefer groups with a rear passenger; target 440 images (about 1,000 boxes); our own split: about 12 whole groups for val; map to class `person`; keep greyscale frames. Do not use the supplied valid/test.
(b) Gender: auto-labelling is poor. Of 500 boxes where YuNet found a face (a face was found on about 45% of boxes; 1,114 probed), face classifier v2 confidence was >= 0.9 on 118 (24%), 0.7-0.9 on 230, < 0.7 on 152. No children (all adults).

### rf_passenger-gender-dataset_v3: REJECT for the detector; PARTIAL (small, review-gated) for Stage B
Evidence: 1,110 images 640x640 (1053/39/18) from 368 source photos; head boxes man 4,271, woman 2,794 (after de-duplication by source photo: man 1,328, woman 838). Viewed 36: Indonesian buses and minibuses, ceiling or front camera, many dark/IR frames and overexposed ceilings, up to 22 heads per image; head boxes are tiny (median 0.7% of the image; about 1/3 of boxes below 0.5%), some drawn on dark blobs or seat parts (roughly 1 of 8 viewed images had doubtful boxes), woman is largely hijab in this population. No child class. Head-only boxes are not person boxes, so the detector gets nothing. As crops (after dedupe by source photo, min box side >= 48 px at 640): man 754, woman 383 (train 681 / 345).
Rule (b only): unique source photo, min side >= 48 px, train split, then hand review of the crops (about 1,100) keeping at most 250 men and 250 women (women first). It adds hijab, bus and IR diversity but cannot be tested (18 photos in test). If hand review is not affordable, skip it.

### rf_child-detection-6ckvk_v4: PARTIAL (child images from cars; dolls and teddies only as negatives)
Evidence: 2,829 images (2,293/352/184); boxes child 1,923, adult 410, doll 1,116, Teddy Bear 126. By unique source photo: child-containing 1,432, adult 317, **doll- or teddy-only 1,113 (39%)**. Child = infant to about 10-12 (viewed 72 child/doll images: no teens; a teen driver is labelled adult), so the label follows apparent age by eye. Quality: boxes are loose and **inconsistent in extent**: tight faces for small children in rear seats, upper body for others, a child plus seat for babies; child boxes median 6% of the image area, p10 56 px min side, 1,656 of 1,919 at least 64 px. Domain: stock photos (Getty, iStock) mostly of children in cars (about 90% of 36 viewed child images: rear seats and child seats; the rest portraits); adults appear in 22% of child images and are only partly labelled. Viewpoint: eye level or close-up to the rear seat, not a ceiling camera. Dolls: the 1,116 doll boxes are mostly the same 3-4 baby dolls on a box, rug or floor, not in cars (18 viewed: 16 off-car, 1 in a child seat, 1 stock cut-out). Teddies sit in car seats next to children in 48 images: plausible cabin objects.
Rule:
1. Drop the 29 images that baby-adult duplicates (keep here) and keep one image per near-duplicate cluster.
2. Take child-containing images only (1,432 unique; cap 450 for the detector; for (b) cap 600 child crops and 200 adult crops), train plus valid; the supplied test stays unused. Use the re-box step below because box extents are mixed.
3. Doll and Teddy Bear: delete these boxes inside kept child/adult images (an unlabelled doll is background for the detector, as wanted). Keep at most 40 doll-only images with an empty label file as hard negatives: one per cluster at corr < 0.9, preferring in-car and in-seat ones, chosen from a 40-image review sheet. Do not use the other 1,070 doll/teddy-only images. Doll/teddy classes are never trained.
4. The adult label (410 boxes) is partial: use adults for Stage B crops only, not as detector labels.

### kaggle_pa100k: PARTIAL, Stage B male/female pretraining only; REJECT for the detector; it cannot supply a child class
Evidence: 100,000 pedestrian crops from 598 outdoor surveillance scenes; median 201 x 74 px (p10 148 x 54); the head is about 25 px, the face about 18 px wide. Female 36,492 / 4,948 / 3,896 (train/val/test); train Age18-60 74,721, AgeOver60 1,127, AgeLess18 4,152 (female 1,695, male 2,457); val AgeLess18 1,443 (female 841); test 92. Views (train) Front 28,718, Side 22,706, Back 28,576. Adults (train): female 34,439, male 40,282. Viewed 36 stratified (female/male x <18, 18-60, 60+) plus 72 AgeLess18: gender labels look right (30+ checked, a few ambiguous back views). Outdoors, daytime, walking, full body: not a seated cabin passenger; an upper-body crop is the nearest match.
**AgeLess18 is mostly teenagers.** In the 72 viewed crops about 80% are school-uniform students (white polo with blue collar, school bags) who look 12-17 and about 15% are clear young children. So the class would teach "school uniform = child" and conflicts with the owner's view that teens are not children. Two machine attempts to separate them failed: face classifier v2 on YuNet faces of 5,595 AgeLess18 crops (face found on 63% after 3x upscaling, median 18 px) gives P(age 0-9) > 0.5 for 8% of them vs 4% of 2,500 adults (no signal); head-size ratio (face width / crop height) is 0.090 vs 0.092 (no signal). **There is no age threshold in the file that fixes this: do not use AgeLess18 automatically, and drop 13-17-looking people from gender as well.** Project definition: child = apparent age up to about 12. Optional hand route: 1,200 random Front/Side AgeLess18 crops from train+val on 12 sheets of 100, keep the clear young children (expect 150-250); low priority.
Rule (b pretraining): train+val csvs; Age18-60 and AgeOver60 only; Front and Side 80%, Back 20%; crop = top 45% of the image height (head and shoulders to chest), pad to square with edge colour, resize 112; drop near-twins (corr > 0.97, keep one per cluster); 6,000 female + 6,000 male (12,000 crops; 10,000 train / 2,000 holdout split by cluster; AgeOver60 included in these counts, about 1,500); the official test adults (9,350) only as an extra check on a 2,000 sample.

### Re-box rule (child/person boxes of mixed extent -> person boxes) for the cleaner
For every kept in-car image of baby-adult and 6ckvk: run the repo person detector (yolov8n, conf 0.3). For each labelled child/adult box take the detection whose box contains the label centre (largest IoU, at least 0.2) and give it that class. If none matches, keep the original box only if its min side >= 48 px, else drop the image. Unmatched detections stay as `person` for the detector set (class unknown for crops). Make a 60-image review sheet (random order) to estimate the error before bulk use. For crops use the re-boxed box top 60% (head and torso).

### Answers to the design questions
**(a) Stage A person/occupant detector (single class `person`; belts and child seats stay separate single-class detectors from the earlier rounds):**
- ihdwp 440 images (about 1,000 boxes).
- baby-adult 184 + 6ckvk child images 450, after the re-box step (about 630 images; complete-label check on the review sheet).
- car_cabin_v1 train frames (134) as given; val and test frames never in training.
- Total about 1,200 images at 416 px. Not for the detector: the gender set, PA-100K, doll/teddy classes, supplied val/test of ihdwp and 6ckvk.
Verdicts: ihdwp ADOPT (subsampled); baby-adult and 6ckvk PARTIAL; gender REJECT; PA-100K REJECT.

**(b) Stage B male/female/child crop classifier:**
- Pretrain: PA-100K adult crops 12,000 (male/female; mask the child output in this stage).
- Fine-tune set about 2,500 crops, split by cluster, at most 2 crops per cluster or per clip:
  - child about 950: 6ckvk 600 + baby-adult 250 + car_cabin_v1 train/val 60 (all of them) + UTKFace/FairFace ages 0-9 about 100 (low weight);
  - male about 870: car_cabin_v1 train/val 186 + ihdwp 250 + 6ckvk adults 80 + gender-set 250 + PA-100K 100;
  - female about 755: car_cabin_v1 train/val 205 + ihdwp 120 (women are scarce) + 6ckvk adults 80 + gender-set 250 + PA-100K 100.
- Gender for crops of sets without it (ihdwp passengers, 6ckvk adults, baby-adult adults): run YuNet then face classifier v2 on the top 55% of the box; accept automatically if confidence >= 0.9 (ihdwp 118 of 500 probes, about 11% of all boxes, about 250 crops; 6ckvk adults 118 of 282 = 42%, about 170 crops; baby-adult almost none); put the 0.7-0.9 band (ihdwp 230 of 500; 6ckvk 102 of 282) on a hand-review sheet; drop the rest. Cabin faces are misread about 1 time in 6 (04_car_labels): spot-check 10% of the auto-accepted. Child is taken only from the dataset label (6ckvk child, baby-adult child), never from the face model.
- Never in training: car_cabin_v1 test videos (car-vid-5, car-vid-8). Car val videos choose thresholds only.

### CPU measurements (this laptop, 12 threads, torch 2.14 CPU, ultralytics 8.4.174; measured)
- YOLOv8n fine-tune from yolov8n.pt, batch 16, workers 4, mosaic on, 160 ihdwp images: **416 px: 2.6-2.8 s per batch (0.17 s per image, 6 img/s)**, 77 s for 2 epochs incl. validation; **640 px: 7.0 s per batch (0.44 s per image, 2.3 img/s)**, 164 s for 2 epochs. 416 is 2.6x cheaper.
- Classifier training step, batch 64, fp32: MobileNetV3-small 96 px 117 img/s; 128 px 89 img/s; MobileNetV3-large 128 px 35 img/s; ResNet-18 128 px 26 img/s; EfficientNet-B0 128 px 18 img/s. Inference at batch 1: MobileNetV3-small 16-17 ms, MobileNetV3-large 35 ms, ResNet-18 29 ms, EfficientNet-B0 53 ms.

### Recommended CPU training recipe
1. Stage A: YOLOv8n from yolov8n.pt, class `person`, **imgsz 416**, batch 16, 40 epochs, close_mosaic 5, patience 10. 1,200 images = about 200 s/epoch + 20 s validation = **about 3.7 min per epoch, about 2.5 h for 40 epochs** (overnight). Optionally the last 10 epochs at 640 (about 9 min per epoch, 1.5 h) if small rear passengers are missed; not from the start (about 6.5 h). Belt and child-seat detectors (single class each, 416 px, about 1,000 images, 40 epochs) cost about 2.5 h each.
2. Stage B: MobileNetV3-small (ImageNet weights), **112 px input** (about 100 img/s train), AdamW 1e-3 cosine, label smoothing 0.1, augmentation: flip, small scale/crop, colour jitter, random greyscale (IR), random erasing. Pretrain 12,000 crops x 8 epochs = 96,000 samples = **about 16 min**. Fine-tune 2,500 crops x 25 epochs (backbone lr 3e-4, head 1e-3) = 62,500 samples = **about 10 min**. Total about 30 min. Larger nets cost 3-5x and the data is too small to use them.
3. Evaluate on car_cabin_v1 val (thresholds) and test (once). The test is about 7 distinct people: report per person with intervals, not per frame. Run `scripts/qa_gate.py` after the cleaned exports (project rule).

### Remaining gaps
- No real child_seat boxes and no belt-with-passenger boxes here (child boxes that include the seat only teach seat appearance implicitly).
- Women and teens/pre-teens in a real cabin: ihdwp is mostly masked men; 6ckvk/baby-adult are watermarked stock photos at eye level; PA-100K is outdoors. A ceiling or dash camera with women, teens and children is missing (only the owner's footage; car_cabin_v1 has about 14 people).
- IR/night with passengers: ihdwp's 8% grey frames only.
- No source defines children vs teens reliably; keep teens out of all labels.
- Hand work needed: 60-image re-box sheet; 0.7-0.9 gender review (about 330 crops); 40 doll sheet; optional gender-set review (about 1,100 crops) and 12 PA-100K child sheets.
- The 32x32 correlation cannot detect other frames of the same stock clip; absence of leakage into val/test rests on the correlation and 16 viewed pairs.

### Ranked list for this round
1. rf_seat-belt-ihdwp_v1 (ADOPT subsampled, grouped by clip).
2. rf_child-detection-6ckvk_v4 (PARTIAL, child images only, dolls removed).
3. kaggle_pa100k (PARTIAL, adult gender pretraining; no child class).
4. rf_baby-adult-in-car_v1 (PARTIAL, 184 images, child crops).
5. rf_passenger-gender-dataset_v3 (PARTIAL for hand-reviewed crops; REJECT for the detector).
