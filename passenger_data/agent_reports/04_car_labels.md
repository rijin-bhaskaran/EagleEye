# 04 Car cabin labels (2026-10-28)

First real male / female / child cabin set, from the owner's 12 car videos. Export: `passenger_data/annotations/car_cabin_v1/` (images, labels, data.yaml; whole-video splits; no other export touched). Review material: `car_label_previews/`, `car_label_review.csv`, `car_label_dropped.csv`, `car_label_how_to_review.md`; my original labels in `car_labels_original/`.

## What was done
- Registered all 12 files with `data_manager register --vehicle car`, extracted at 2 fps with near-duplicate dropping (494 frames). raw/ untouched. The 4096x2160 clip was kept at original size; no resized copy was made (labels are normalised; previews are 960 px wide derived copies). Its 64 exported frames are large (the export folder is about 106 MB).
- Boxes: the repo person detector (yolov8n, conf 0.25), followed through each clip by box overlap; for the small children in the back seat, a box derived from the YuNet face box (head and upper body, approximate). Classes: my own visual judgement of every person (I looked at the frames of every video), cross-checked against the repo face classifier.
- Frame rule (ANNOTATION rule 4): a frame is dropped if a person's box is missing, a person is leaving the frame, or an unlabelled person is visible.

## Splits (whole video, fixed)
| split | videos | labelled frames |
|---|---|---|
| test | car-vid-5 (family, sunny road, 2 adults + 3 children), car-vid-8 (man + woman in sunglasses, green road) | 92 |
| val | car-vid-6 (windscreen view, couple + baby), car-vid-11 (man + woman in sunglasses, sunny road) | 85 |
| train | 4607377 (4K taxi: driver + rear passenger), car-vid-2 (woman), car-vid-3 (man, woman, boy); plus unlabelled car-vid-1, 4, 7, 9, 12 | 134 |

Test and val videos hold different people and settings from the labelled train videos; car-vid-5 went to test because it has the children. Limit: all clips are stock footage, and the windscreen filming style of car-vid-6 (val) also appears in car-vid-9 and 12 (train, unlabelled), so val is not fully independent in look.

## Boxes per class per split
| split | male | female | child |
|---|---|---|---|
| train | 101 | 120 | 39 |
| val | 85 | 85 | 22 |
| test | 92 | 92 | 180 |

Total 816 boxes in 311 frames.
- **The test set is two videos (92 frames) with 4 adults and 3 children: tiny and noisy.** Frames of one video are near-identical, so the effective sample is about 7 people, not 364 boxes. By box count test is near the ~100 target (male 92, female 92 slightly below, child 180 above), but 92 boxes of 2 adults of each sex are not 92 independent examples. Val child is 22 boxes (one baby); train child is 39 boxes (one boy). Any score here swings on single people; do not quote it as a car accuracy.

## Frames dropped (11) and videos not used
Dropped (`car_label_dropped.csv`): car-vid-3 9 frames (woman or boy box lost 6; a 4th person visible 3); 4K clip 2 frames (rear passenger half out of the frame).
Videos left unlabelled (frames stay in `processed/frames/`, split train, ready to label later):
- car-vid-1 (15 frames): night, a child asleep behind the driver, dark and partly hidden by the watermark; age and boxes too uncertain.
- car-vid-4 (51), car-vid-9 (32): extra small people in the back seat (heads about 30-40 px, partly hidden) that I could not box reliably; labelling only the front two would break rule 1.
- car-vid-7 (25): crowded back seat, 4-5 overlapping adults and children; needs hand annotation.
- car-vid-12 (49): the infant in the rear seat is hidden in most frames (face found in 4 of 49), so omitted rather than guessed.

## Known gaps inside the labelled videos
- car-vid-3: a toddler behind the driver is mostly hidden. Only the 3 frames where its face was found were dropped; in the other 39 it may be partly visible and is NOT labelled. Every car-vid-3 frame is flagged `least_sure`.
- 202 child boxes (all in car-vid-5 and car-vid-6) are face-derived, so their extents are approximate (flag `approx_box`, 82 frames); the class is certain by eye (small children).

## Visual judgement vs model-assisted
- Boxes from the detector: 614 of 816 (75%); from the face detector: 202 (25%).
- Classes: 100% my visual judgement, one decision per person per clip. For the 614 detector boxes the repo classifier agreed on 494 (80%), gave no answer on 23 (4%) and disagreed on 97 (16%). In every disagreement I kept my judgement (long hair, clothing and sunglasses or profile faces are what the model misreads; mostly the women in car-vid-11, 2 and 3). So the classifier misreads about one cabin face in six here, and a score on this set will look worse than on FairFace.
- Flags: least_sure 53 frames (train 43, val 7, test 3), model_disagrees 91 frames. I am the only labeller: a human spot-check of at least the flagged rows is needed before any score is trusted.

## Caveats
- Watermarks: every clip carries a large Envato or iStock watermark across the cabin, covering people. A model trained here can learn the watermark; a real camera has none. Licence is unverified (`raw/external/car_cabin_videos_stock/SOURCE.json`): internal experiment only.
- Domain: staged stock footage, good light, people mostly facing the camera, a handful of people in total (about 14 across the 7 labelled videos); no night, no real in-car camera angle.
- Frames at 2 fps from 7-30 s clips are still highly similar to each other.
- These frames are now in the manifest, so `evaluate --source videos` and the QA gate use them (92 test frames); gate evaluation numbers before and after are not comparable.

## Identity (DataManager fingerprints, source videos)
train `5f308afb893029f6`, val `7a42eb4ff5c335f5`, test `8c1bfff8e2e7ad7a`. Scripts (not project code) are in the session scratchpad: `prelabel.py`, `track.py`, `assemble.py`, `finalize.py`.
