# 08 New in-car video collection: cleaned, grouped and labelled (2026-10-28)

Source: the owner's folder (33 new clips; `4607377-uhd...` was already in `data/car`). Nothing in that folder was modified. Originals are in `passenger_data/raw/videos/car/` (`verify`: raw data intact). Rights recorded per video: `none stated / unverified (stock watermark or screen capture of third-party video); internal training only`.
Exports (new names; nothing older touched): `car_cabin_v2` (153 MB), `car_cabin_v2_aug` (221 MB), `crops_mfc_v3` (24 MB). Review material: `car_label_previews_v2/` (44 MB), `car_label_review_v2.csv`, `car_label_dropped_v2.csv`, `car_labels_original_v2/`, `car_label_how_to_review_v2.md`. No model was trained, registered or promoted; `config.yaml` untouched.

## 1. Groups (splits are by whole group) and per-video table
22 groups. Evidence = frames viewed (contact sheets of every clip). The three 1011/1015/1033 clips show the same two men in the same fisheye car; the vlogger (man with glasses) drives in all four 2046/2048/2056/2057 screen recordings; the cap-wearing woman and the man in the navy tee appear in four recordings (210532 is a compilation of several rides including them); the two suit-driver clips share driver and car; 2995105 and 2997616 look like the same actor (merged on purpose, not certain). **1103517607 shows the same man in the same car as the existing val clip car-vid-11, so it inherited val** (the boy in it is the only new child). Existing clips were compared to the new ones side by side: no other overlap found.

| video id | source file | group | split | sampled 2fps | selected | labelled | M/F/C boxes |
|---|---|---|---|---|---|---|---|
| 06bd1a0d0e | 1006939087-preview.mp4 | g_pair_glasses_woman | test | 51 | 15 | 15 | 15/15/0 |
| 736364971b | 1009847780-preview.mp4 | g_pair_dark_hair | val | 40 | 10 | 9 | 9/9/0 |
| 673229eeb4 | 1011852128-preview.mp4 | g_plaid_men | train | 52 | 23 | 23 | 46/0/0 |
| dd7d1deb15 | 1015987696-preview.mp4 | g_plaid_men | train | 48 | 7 | 7 | 14/0/0 |
| caa031c735 | 1033372688-preview.mp4 | g_plaid_men | train | 43 | 6 | 6 | 12/0/0 |
| 9f9f62890c | 1028305148-preview.mp4 | g_bald_hoodie | train | 39 | 17 | 17 | 17/0/0 |
| e51cc3370e | 1039413257-preview.mp4 | g_bearded_hoodie | train | 18 | 7 | 7 | 7/0/0 |
| 2a619dce68 | 1103517607-preview.mp4 | g_car11_family | val | 119 | 38 | 38 | 38/0/38 |
| 524b28be41 | 1106470831-preview.mp4 | g_sunglasses_tee | test | 61 | 20 | 20 | 20/0/0 |
| 8bd665fb8b | 13070387-preview.mp4 | g_dark_car_man | test | 60 | 24 | 23 | 23/0/0 |
| 875490a61a | 20991304-preview.mp4 | g_bald_wheel | train | 53 | 20 | 20 | 20/0/0 |
| d0665489c8 | 2995105-preview.mp4 | g_curly_beard | train | 39 | 15 | 14 | 14/14/0 |
| b49ab11dd4 | 2997616-preview.mp4 | g_curly_beard | train | 16 | 5 | 2 | 2/0/0 |
| ed4fd14aca | 4657409-preview.mp4 | g_teen_denim | test | 59 | 23 | 16 | 16/16/0 |
| 961ca9395a | 4657775-preview.mp4 | g_striped_polo | val | 53 | 18 | 18 | 18/18/0 |
| c703962698 | 7586458-preview.mp4 | g_goatee_barefoot | train | 60 | 20 | 20 | 20/0/0 |
| 8d7a84f9cc | 8479435-preview.mp4 | g_scarf_woman | test | 60 | 23 | 18 | 18/10/0 |
| 35ffb9660c | 9519369-uhd_4096_2160_25fps.mp4 | g_suit_driver | val | 38 | 12 | 11 | 11/7/0 |
| 37dba31f41 | 9520003-uhd_4096_2160_25fps.mp4 | g_suit_driver | val | 75 | 27 | 21 | 33/0/0 |
| 59b1378a1d | 9530762-uhd_4096_2160_25fps.mp4 | g_night_beard | train | 70 | 32 | 32 | 32/0/0 |
| 551f23ec7f | istockphoto-2234158942-640_adpp_is.mp4 | g_istock_family | train | 47 | 23 | 0 | 0/0/0 |
| 7a808a5444 | SR 204624.mp4 | g_vlogger_glasses | train | 40 | 16 | 16 | 16/32/0 |
| 3788def01f | SR 204830.mp4 | g_vlogger_glasses | train | 32 | 15 | 15 | 15/0/0 |
| 9e7a8cfc9f | SR 205614.mp4 | g_vlogger_glasses | train | 39 | 12 | 12 | 12/12/0 |
| ba04c71e03 | SR 205726.mp4 | g_vlogger_glasses | train | 40 | 18 | 16 | 16/16/0 |
| b2d6faf0f2 | SR 210135.mp4 | g_redpolo_dashcam | train | 32 | 14 | 14 | 14/0/0 |
| 08f4136bb1 | SR 210532.mp4 | g_cap_couple | val | 43 | 17 | 2 | 2/2/0 |
| 86e79feb74 | SR 210611.mp4 | g_cap_couple | val | 8 | 2 | 2 | 2/2/0 |
| 3f5e0d5855 | SR 211209.mp4 | g_cap_couple | val | 60 | 24 | 24 | 24/24/0 |
| a6daefd93b | SR 211319.mp4 | g_cap_couple | val | 53 | 20 | 17 | 17/17/0 |
| e91216632f | SR 210759.mp4 | g_tattoo_pair | test | 116 | 40 | 29 | 58/11/0 |
| ad93cfa2ef | SR 210836.mp4 | g_hoodie_old | train | 16 | 1 | 1 | 2/0/0 |
| df5bf56ed2 | SR 211013.mp4 | g_hoodie_old | train | 72 | 25 | 25 | 50/0/0 |

Splits chosen (new groups): **val** = g_car11_family (forced), g_striped_polo, g_suit_driver, g_pair_dark_hair, g_cap_couple. **test** = g_scarf_woman, g_teen_denim, g_tattoo_pair, g_pair_glasses_woman, g_sunglasses_tee, g_dark_car_man. Everything else **train**. Existing splits unchanged and asserted in code before saving: car-vid-5 and -8 test, car-vid-6 and -11 val.
Adding to test starts a new benchmark: the test split grew from 2 videos (92 frames) to 8 videos (213 frames) and its fingerprint changed (videos source now train 275fd4c4d3c84e22, val c9bb9d3119be5b9a, test bc6b09b9c28f1fca; before 5f308afb893029f6 / 7a42eb4ff5c335f5 / 8c1bfff8e2e7ad7a). Scores on the old 2-video test set are no longer the only reference; any champion must be re-scored on the new test before comparison.

## 2. Cleaning pipeline and survivors after each filter (2 fps, one clip decoded at a time, 2 threads)
| step | frames | note |
|---|---|---|
| sampled at 2 fps | 1,652 | 33 clips, 14 min |
| manual time exclusions | -15 | black frame, unrelated white-room clip (SR210836), dashcam product shots (SR210611), a street cut (SR211209), a snow road (SR210532); chosen by eye |
| exposure (mean grey < 25 or > 235) | -2 | |
| blur (Laplacian variance at <= 640 px < max(30, 0.5 x clip median)) | -6 | |
| no person (YOLOv8n conf 0.25, box >= 2% of the frame) | -7 | leaves 1,622 |
| near-duplicate: whole-frame 32x32 grey correlation > 0.97 vs the last kept frame AND person-region correlation > 0.95 | 1,622 -> 1,161 | the plain 0.97 rule alone would keep only 874: static cabins (1011852128: 52 -> 2) lose all pose variety, so a frame counts as a duplicate only if the people region did not change either |
| diversity cap (farthest-point sampling on 16x16 grey + person boxes + person count, <= 40 per clip, >= 1 s apart) | **589** | |

Border auto-crop (temporal-std strips ending in a sharp edge): only SR204624 had one (48 px Windows taskbar at the bottom); the other 11 recordings have none. Full-frame originals stay in raw/; derived frames are <= 1280 px long side (4K clips downscaled).
Not removed (caveat): the screen recordings still show transient player UI (title, pause icon, progress bar, stop-recording toast) and burned-in captions in some frames; stock watermarks cover the cabin in all stock clips. Reusable code: passenger/frame_quality.py (+9 tests). The per-clip driver script is scratch (scratchpad/nv/proto.py); the frames came from an identical prototype of the module, not from a re-run through the final module (same functions; selections not re-verified).

## 3. Labels
Method as for the first 12 videos: repo person detector (YOLOv8n) boxes tracked through each clip from one anchor per person; class by my own visual judgement, once per person per clip; the repo face classifier only as a cross-check. 510 frames labelled; 79 selected frames not labelled (car_label_dropped_v2.csv): rear passengers of the iStock family are early teens (23 frames: child vs teen undecidable, clip omitted), SR210532 compilation (15: night ride with 4 overlapping dim people, a puppy, a small child behind a seat), driver-side man partly out of frame or hiding the rear woman in SR210759 (11), rear passenger moving out of view (6), people entering or leaving (8), tracking lost / hand over a person / sun flare / black frame (10). No face-derived boxes in this batch (no approx_box flags). Model cross-check on 856 person boxes: agrees 753, no face read 11, disagrees 92 (11%); I kept my judgement in every case. Flags: least_sure 144 of 510 frames (clip-level reasons: SR210759 two segments, vlogger head turned away, 2995105 hands over people, 13070387 dark stretch, the boy age borderline), model_disagrees 89 frames.

## 4. Counts
Boxes per split (all videos, old + new, from stats): train M 410 / F 194 / C 39 (381 frames, 26 videos); val M 239 / F 164 / C 60 (227 frames, 11 videos); test M 242 / F 144 / C 180 (213 frames, 8 videos). New frames only: train 247, val 142, test 121 (510).
Distinct people (by eye, per group; the same person across clips counted once):

| split | male | female | child |
|---|---|---|---|
| train, old | 2 | 3 | 1 |
| train, new | 12 | 5 | 0 |
| val, old | 2 | 2 | 1 (baby) |
| val, new | 5 (+ the family man = existing car-vid-11 man) | 4 | 1 (boy, about 10-12) |
| test, old | 2 | 2 | 3 |
| test, new | 9 | 4 | 0 |
| **total** | train 14 / val 7 / test 11 | train 8 / val 6 / test 6 | train 1 / val 2 / test 3 |

Balance: new labelled boxes are 613 male / 205 female / 38 child (72% / 24% / 4%). **Children are the gap**: the only new child is one boy in a val clip; train still has a single child person; test children are still the 3 from car-vid-5. Female is about half of male in train boxes. Frames of one person are highly correlated: 510 frames are about 33 people, not 510 independent samples.

## 5. Augmentation (car_cabin_v2_aug, separate export)
Live trainers already augment online (YOLO mosaic/HSV/flip; cabin_augment for crops): not duplicated. Offline, train only: child frames get 3 variants, female-only frames 1, male-only frames none; 256 variants from the 381 train frames (child 117 boxes from 39, female 272 from 194, male 223 carried along from 410). Transforms (2-3 per variant, seeded): gamma/brightness, IR-like grayscale + CLAHE, motion blur, noise, JPEG q 18-45, rotation +-7 deg and scale 0.92-1.1 (boxes recomputed; variant rejected if a person ends < 70% inside), horizontal flip (boxes flipped), low-res upscale, random erasing covering <= 25% of any box. Names source_augK; no variant in val or test (checked: 0); val/test and all originals are identical to car_cabin_v2; augmentation_log.json lists source, variant and operations. Share of augmented boxes in train: child 75%, female 58%, male 35%. **Benefit is NOT validated**: the 117 child variants are copies of the same few children, so they add appearance variety, not new people. Recommendation: do not use it as the default; train on car_cabin_v2 first and run one ablation (v2 vs v2_aug, same seeds, same val/test) before adopting it.

## 6. Crops (crops_mfc_v3)
Existing script logic reused (top 60% of the person box, square, 112 px, edge-replicate pad, min side 24 px; no crop rejected). v3 = v2 (unchanged) + new cabin crops named car2_*: new train M 309 / F 74, val M 154 / F 79 / C 38, test M 150 / F 52. Totals (male / female / child): train 595 / 277 / 771, val 254 / 175 / 149, test 242 / 144 / 180. Checks in the script: unique file names, each group in a single split, no train crop that is a near-duplicate (corr > 0.95) of a val/test crop (0 found), no PA-100K or augmented crop in val/test. The v2 recommendation (neutral pad instead of edge replication) was not applied, to stay comparable with v2.

## 7. Remaining gaps and caveats
- Children: 1 new child in total; none new in train or test. Real children aged 2-10 are needed before any child score means anything.
- Women: 14 new people but only 5 in train (4 are passengers of one vlogger); men are 72% of new boxes.
- Same-actor risk: 2995105/2997616 merged by judgement; stock clips from one series may share people I did not recognise.
- Domain: staged stock footage and YouTube screen captures (selfie-style close-ups, boxes often cut by the frame edge), watermarks and captions burned in, no real in-car camera angle, little night; a model can learn the watermark.
- Rights unverified, internal use only. Single labeller; no human spot-check yet.
- 596x336 previews: small faces; 4K clips shown at <= 1280 px.
- The gate evaluation set grew (630 frames incl. 213 video frames); rows before and after are not comparable.

## 8. What the owner should spot-check first
1. All LEAST_SURE rows of car_label_review_v2.csv, starting with video e91216632f (SR210759: Wolverine-tee man, driver, small rear woman) and 2a619dce68 (the boy: is he 12 or younger?).
2. The 89 MODEL_DISAGREES rows, especially women in caps/hoods (SR211209, SR211319) and the vlogger passengers.
3. Frames of the six new test groups (121): a wrong label there changes the benchmark.
4. That g_curly_beard (2995105 + 2997616) is one actor, and that the iStock clip should stay unlabelled.

## 9. QA
Tests first: 15 new (437 -> 452). Gate (WORKFLOW.md): 452 passed x2 (all agree), overhead 3.2 ms, evaluation on the new 630-frame set det P 0.672 R 0.788 AP50 0.668, classification macro-F1 0.790 (not comparable with 0.773 on 509 frames). The gate was started while two of my own verify runs were still reading raw/ (noticed afterwards; killing them was refused), so it was not on a fully idle machine; it was green and not repeated.
