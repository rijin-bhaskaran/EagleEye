# 07 Crop classifier v2: data fix, balance, regularisation, bias/variance (2026-10-28)

Nothing registered or promoted; config.yaml and the registry untouched. Model: `models/passenger/crops/v2/` (model.pt, metrics.json, train.log). Scratch scripts and run logs: session scratchpad `.../scratchpad/v2/`.

## 1. Data fix: `crops_mfc_v2`
Cause of the 47 overwritten crops: in `build_crops2.py` the file name was `<src>_<origin[:30]>_<box>_<sha1(origin+box+class)[:5]>` and `box` was a constant 0 for every child and face-classifier crop. Two crops from one image (the "2 per group" rounds) therefore got the same name and the second write replaced the first (baby_adult child 36+4, ihdwp male 4, 6ckvk male 2 + female 1). Fix: a running index in the name. Built from the saved stage-1 crops (same rules, same near-duplicate filter: 0 drops, max corr 0.920). Asserts: no group spans two splits, no non-car crop in test, test/val videos never in train, no duplicate (split, class, file), no byte-identical crops. review_pending is not included; PA-100K stays in `crops_mfc_v1/pretrain_pa100k` (not copied).

| | v1 on disk | v2 |
|---|---|---|
| train male / female / child | 280 / 202 / 735 (1,217) | 286 / 203 / 771 (1,260) |
| val | 100 / 96 / 107 (303) | 100 / 96 / 111 (307) |
| test | 92 / 92 / 180 | 92 / 92 / 180 |

Per source (v2 train male/female/child): car 101/120/39; 6ckvk 61/74/534; baby_adult 3/3/198; ihdwp 121/6/0. Val: car 85/85/22; 6ckvk 3/6/66; baby_adult 0/3/23; ihdwp 12/2/0. Test: car only.

Distinct-people proxies (car = distinct videos, about 1-3 people each; other = distinct source images, not people):
| split / class | car videos | other source groups |
|---|---|---|
| train male | 2 (0d5c, cb4b) | 138 |
| train female | 3 (0d5c, cb4b, e2d6) | 81 |
| train child | 1 (0d5c) | 696 |
| val male / female / child | 2 / 2 / 1 | 11 / 11 / 85 |
| test male / female / child | 2 / 2 / 1 | 0 |
So train has about 3 cabin women, 2 cabin men and 1 cabin child video; every other child is a stock close-up.

## 2-3. Code and ablation (all selection runs used `--no-test`; test was never read)
Added to `passenger/train_crops.py` (tests first; 8 new tests, 12 in the file): `balanced_order` (class-balanced epochs: each class drawn equally, rare classes with replacement, child sub-sampled each epoch, so child is capped dynamically and no crop is lost), `cabin_augment` (colour/saturation jitter + random erasing on top of the existing zoom/shift/tilt/low-res/blur/brightness), `regularise` (label smoothing 0.1, weight decay 0.02, backbone frozen 3 epochs then 0.1x LR, early stop patience 8), `adult_mix`, `pretrain_cache`, `score_test=False`, and balanced accuracy, recall spread, clean train score and per-epoch val loss in the metrics. Per-source caps were not built.

Val (307; 2 cabin videos, about 3 people). Same 8-epoch PA-100K pretrain for all (holdout macro-F1 0.87), one seed each unless stated.
| Run | val macro-F1 | val recall M / F / C | recall spread | bal. acc | train macro-F1 | train-val gap | best epoch |
|---|---|---|---|---|---|---|---|
| r1 control (v1 recipe on v2) | **0.679** | 0.92 / 0.27 / 0.94 | 0.67 | 0.709 | 0.984 | 0.31 | 14 |
| r6 control, seed 1 (a different pretrain seed too, by accident) | 0.535 | 0.76 / 0.09 / 0.96 | 0.86 | 0.603 | 0.990 | 0.46 | 14 |
| r2 + balanced sampling | 0.588 | 0.92 / 0.09 / 0.97 | 0.88 | 0.662 | 0.964 | 0.38 | 7 |
| r5 + regularisation only | 0.537 | 0.88 / 0.09 / 0.78 | 0.79 | 0.586 | 0.903 | 0.37 | 26 |
| r3 + balanced + regularisation | 0.549 | 0.84 / 0.10 / 0.84 | 0.74 | 0.594 | 0.924 | 0.38 | 29 |
| r4 = r3 + 300+300 PA-100K adults | 0.425 | 0.42 / 0.07 / 0.93 | 0.86 | 0.474 | 0.900 | 0.47 | 17 |
| final: control recipe re-run (seed 0) to score test | 0.606 | 0.86 / 0.18 / 0.94 | 0.76 | 0.658 | 0.985 | 0.38 | 14 |

Reading: nothing added beat the control on val. The two control seeds differ by 0.14 and a re-run of the same seed by 0.07 (thread non-determinism), so every gap between recipes (0.04-0.14) is inside noise except the PA-100K mix, which is clearly worse (domain gap; male recall fell to 0.42). Balanced sampling shows the female error is not a class-weight problem: female val recall is 0.09-0.27 under weighted loss and balanced sampling alike; balanced sampling shows each of the ~3 training women about 2x per epoch, which if anything memorises them more.

Learning curve (control recipe, stratified random share of train crops, val unchanged):
| train crops | val macro-F1 | val female recall | train-val gap |
|---|---|---|---|
| 25% (316) | 0.373 | 0.21 | 0.62 |
| 50% (631) | 0.542 | 0.08 | 0.45 |
| 100% (1,260) | 0.679 / 0.535 / 0.606 (3 runs) | 0.27 / 0.09 / 0.18 | 0.31-0.46 |
Rising with data; train stays near 1.0.

Leave-one-video-out on the three cabin train videos (control recipe; model picked on the real val; scored on the held-out video's train crops): 0d5c (woman + man + child) male 2/39, female 35/39, child 14/39; cb4b male 51/62, female 26/48; e2d6 female 9/33. An unseen woman is recognised 58% of the time (70/120) vs 18-27% for the val women; the only cabin child video fails (14/39) once held out.
Zero-shot PA-100K net on cabin adults: val car female 22/85, male 25/85; train cabins male 4/39 (0d5c) to 56/62 (cb4b). Outdoor gender cues do not transfer to these cabin crops.
Per-video val of the final model: car 2af461cf70 male 52/63, female 10/63; d3d258db0c male 22/22, female 0/22, child 19/22; 6ckvk child 63/66; baby_adult child 22/23; ihdwp male 11/12. I viewed a sheet of the val women: labels are right (one wears sunglasses with hair tucked in, one smiles in a light shirt); the train women are all long dark hair, and many train crops carry replicate-padding streaks.

## Bias vs variance verdict
**Variance dominates.** Train macro-F1 0.98-0.99 vs val 0.54-0.68 (gap 0.3-0.46), val loss 3-20x train loss, val F1 plateau after epoch 7-14, learning curve still rising, seed noise 0.07-0.14, and recall depends on which woman is in the held-out video. Regularisation lowered train F1 to 0.90-0.92 without raising val: regularising 3 women does not create a 4th. A bias component exists at the domain level (PA-100K features are near chance zero-shot on cabins), which more cabin adults fix, not a bigger network (pretrain holdout 0.87, train 0.99).

## 4. Selection and the final test score
Recipe chosen by val and the per-video check: the control (v1 recipe) on v2 data; the balanced, regularised and PA-mix variants were all lower on val and none raised per-video female recall. Caveat: val is noisy; the control's mean over its two seeds is 0.607 vs 0.54-0.59 for single seeds of the others. Test was scored once, by the final run, after the choice; no tuning followed.

Final model, TEST (364 crops, 2 videos, about 7 people), scored once: accuracy 0.577, macro-F1 0.556, balanced accuracy 0.566, recall male 0.62 / female 0.47 / child 0.61, recall spread 0.15 (confusion rows male 57/27/8, female 37/43/12, child 22/48/110). v1 on the same test: acc 0.615, macro-F1 0.574, recalls 0.57 / 0.43 / 0.73. Not better; the 0.02-0.04 difference is noise. Test spread is 0.15 but val spread is 0.76: the bias metric swings with who is in frame, so neither is a stable property of the model.

Baseline (repo classifier: YuNet + face net) and offline fusion on the same crops (nothing wired):
| | val (307) | test (364) |
|---|---|---|
| Baseline coverage | 65.1% | 78.6% |
| Baseline, answered crops: acc / macro-F1 | 0.560 / 0.549 | 0.654 / 0.625 |
| Baseline answered recall M / F / C | 0.45 / 0.42 / 0.78 | 0.45 / 0.82 / 0.71 |
| Baseline, unknown = miss: acc / macro-F1 | 0.365 / 0.367 | 0.514 / 0.491 |
| New model, same answered crops: acc / macro-F1 | 0.625 / 0.574 | 0.539 / 0.506 |
| New model, all crops: acc / macro-F1 | 0.674 / 0.606 | 0.577 / 0.556 |
| Fusion (face model if face found, else new): acc / macro-F1 | 0.632 / 0.613 | 0.668 / 0.632 |
| Fusion recall M / F / C | 0.66 / 0.38 / 0.83 | 0.43 / 0.76 / 0.74 (spread 0.33) |
| New model where no face found: acc | 0.766 (n 107) | 0.718 (n 78; male 0.17 on 6) |
Differences of 0.1-0.2 here are within noise at about 7 test people (about 3 val people). The two models err in opposite directions (this one reads women as male, the baseline over-calls female), which is why fusion looks best on test and ties on val. No evidence supports promoting either as a standalone classifier.

## Remaining data gaps
Distinct real cabin people in the set: train about 3 women, 2 men, 1 child video; val about 2 women, 2 men, 1 child; test about 2 women, 2 men, 3 children. Needed: 15-20 distinct women and men in varied cars, light, camera angles, hair and clothing, 5+ distinct children aged 2-10, person-level splits (leave-one-person-out), and a neutral pad instead of edge replication. The 178 review_pending crops are still unreviewed. 262 train crops carry face-classifier labels (about 3% visible error in a 29-crop check).

## What I could not verify
- Anything on the owner's own ceiling/dash camera; Pi/edge speed of this model.
- Whether more seeds would order r2/r3/r5: one seed each plus one extra control seed and one re-run.
- Source-diversity caps in the sampler (not built); PA-100K mix tested at one size only (300 per class).
- model.pt is not bit-reproducible on this CPU/thread setup (same seed gave val 0.679 and 0.606).
- Budget: about 2 h of training, including about 25 min lost to one mis-pathed scratch run.

## Process and QA
Gate (idle machine, nothing training): 437 passed x2, overhead 3.3 ms (budget 15), evaluation on 509 frames unchanged (det P 0.632 R 0.747 AP50 0.612, count MAE 0.62, classification macro-F1 0.773). The earlier overhead failures (15.2, 19.6 ms) did not recur on an idle machine; their cause is still unproven. No gate ran while I edited passenger/ or tests/.
Mess left: an MSYS-style path inside a `python -c` string made the r6 scratch run write to `C:\c\Users\RIJINB~1\...` (a pre.pt and the r6 model). The delete was blocked by the safety check; the owner can delete the folder `C:\c`.
