# 06 Training analysis: crops_mfc_v1 male/female/child classifier (run `models/passenger/crops/v1`)

Analysis only. Nothing retrained, promoted or edited; config.yaml and the registry untouched. Scratch scripts (read-only) are in the session scratchpad: `an06.py`, `an06b.py`, `an06c.py`, `an06d.py`, plus the earlier `baseline_vs_crops.py`.
Run: MobileNetV3-small, 112 px, ImageNet init, PA-100K pretrain 8 epochs (holdout acc 0.8725, macro-F1 0.8725), fine-tune 25 epochs lr x0.3, class-weighted loss, best epoch by val macro-F1 (epoch 14, 0.6239). Test scored once.

## 1. Verdict
Worse than or at best equal to the production baseline where the baseline can answer; better only where it cannot (no usable face). Not a replacement: on the crops both can score, the baseline wins on test (macro-F1 0.625 vs 0.521) and loses on val (acc 0.561 vs 0.626), so the evidence is a wash, and neither is trustworthy at 2 test videos.
Worth keeping as a research artefact and as the no-face fallback candidate, not as a classifier on its own: its female class is broken in real cabins (val recall 0.19, test 0.43). The user decides on promotion; I do not recommend it.

## 2. Metrics (same crops, same fingerprints; baseline = YuNet + champion face net, "unknown" shown separately)
| | val (303) | test (364) |
|---|---|---|
| New model acc / macro-F1 | 0.690 / 0.624 | 0.615 / 0.574 |
| Baseline coverage (answers) | 65.3% | 78.6% |
| Baseline, unknown = miss, acc | 0.366 | 0.514 |
| Baseline, answered crops only, acc | 0.561 (198) | 0.654 (286), macro-F1 0.625 |
| New model on those answered crops | 0.626 | 0.584, macro-F1 0.521 |
| New model on the crops the baseline cannot answer | acc 0.811 (n 106; child 1.00, male 0.96, female 0.10) | acc 0.731 (n 78; child 0.89, female 0.67, male 0.17 on only 6) |

Per-class recall (precision in brackets for the new model):
| Class | New val | New test | Baseline val (answered) | Baseline test (answered) | n val / test |
|---|---|---|---|---|---|
| male | 0.88 (0.60) | 0.57 (0.46) | 0.45 | 0.45 | 100 / 92 |
| female | 0.19 (0.82) | 0.43 (0.47) | 0.42 | 0.82 | 96 / 92 |
| child | 0.96 (0.76) | 0.73 (0.79) | 0.79 | 0.71 | 107 / 180 |

Confusions (rows true, cols male/female/child), from metrics.json. Val: male 88/1/11, female 57/18/21, child 1/3/103. Test: male 52/20/20, female 37/40/15, child 23/25/132. The dominant error is female read as male (val 57 of 78 female errors), then female/male read as child. Baseline on answered test crops: male 39/27/20 (male read as female), female 0/46/10, child 11/31/102 (child read as female), so the baseline has the mirror-image bias (over-calls female); the two models err in opposite directions.
Counting: not applicable here (crop classifier); count error comes from the detector, evaluated elsewhere.

Fusion (face model answer when YuNet finds a face, else this model), computed on the same crops, offline, nothing wired:
| | val acc / macro-F1 | test acc / macro-F1 |
|---|---|---|
| New alone | 0.690 / 0.624 | 0.615 / 0.574 |
| Baseline alone (unknown = miss) | 0.366 / n/a | 0.514 / n/a |
| Fusion, all faces | 0.648 / 0.626 | 0.670 / 0.636 |
| Fusion, face used only if its confidence >= 0.8 | 0.651 / 0.621 | 0.679 / 0.640 |
Fusion recalls on test: male 0.43, female 0.76, child 0.74; on val: male 0.70, female 0.35, child 0.86.

## 3. Findings
1. **Overfit to a handful of people; plateau is real.** Train loss 0.93 -> 0.06, train accuracy 0.994 (car sources 1.000), while val macro-F1 reached 0.62 at epoch 14 and then drifted between 0.585 and 0.604 until epoch 25. The model memorises the 3 training cabin videos. Train car female crops (120) come from 3 videos, roughly 3 women; train car male (101) from 2 videos. The pretrain holdout is 0.87, so the network can do gender on outdoor PA-100K adult crops; the collapse is the domain shift to cabin crops, not model capacity.
2. **Female errors are a car-domain problem, not a taxi/mask or label-circularity artefact.** Val per source (own labels vs. face-classifier labels):
   - car (own visual labels, 2 videos): female 10/85 correct (55 read male, 20 read child); male 76/85; child 21/22.
   - 6ckvk female 5/6, baby_adult female 2/3, ihdwp male 11/12, ihdwp female 1/2 (these are the 26 val crops labelled by the face classifier at >= 0.9, 8.6% of val; they score fine).
   So every large val error is on own-labelled car crops; the face-derived labels are a small share of val and are not where the model fails. Circularity exists in train (262 of 1,260 crops, ihdwp 127 men and 6ckvk 135, are face-classifier labels, so the new model inherits its biases) but cannot explain the val female collapse.
3. **Val and test disagree because each is a few people.** Val female are 2 videos (about 2 women); test female are 2 videos. In test, video 67287def25 (3 children + 1 woman + 1 man, 300 crops): female 18/60 (37 read male), male 20/60 (20 each to male, female, child), child 132/180. Video d769e9f0d3 (1 man + 1 woman, 64 crops): male 32/32, female 22/32. A single woman who looks male-ish to the model drops female recall by 30 points. Female recall swings 0.19 -> 0.43 across val/test purely by who is in the frame.
4. **Overconfident when wrong in the target domain.** Test accuracy by confidence: 0.5-0.7 -> 0.54, 0.7-0.9 -> 0.56, 0.9-0.99 -> 0.57, > 0.99 -> 0.98 (60 crops). 27% of the 140 test errors have confidence >= 0.9. Val is better calibrated (0.9-0.99 -> 0.90, wrong predictions >= 0.9 only 7%) because val is 70% easy stock-photo child crops. A confidence threshold will not filter cabin errors on test.
5. **The baseline's face confidence is no better.** Baseline accuracy at face confidence 0.9-1.0 is 0.60 (val) / 0.70 (test); thresholding at 0.8 changes fusion macro-F1 by only +0.004 on test and -0.005 on val. Fusion beats the new model alone on test (0.636 vs 0.574) and ties on val (0.626 vs 0.624); fusion is driven by the baseline's good test-woman recall (0.82) and loses male recall (0.43).
6. **Where the new model is clearly useful: no-face crops.** Where YuNet finds no usable face (35% of val, 21% of test) the new model gets 0.81 and 0.73 accuracy, i.e. it adds an answer where the baseline gives none. Child is near-solved in that slice (val 1.00, test 0.89). Male there is untested in test (n 6) and female is 0.10 on val and 0.67 on test: unreliable.
7. **Child is right for the wrong reason risk.** Val child = 107 crops, 89 from stock photos (6ckvk 66, baby_adult 23) at 0.96-1.00 accuracy; the only car child in val (22 crops, 1 video) scores 21/22. Test child = 3 children in one video, recall 0.73, with 25 read female and 23 read male. Stock-photo children are close-ups and easy; the high val child score does not predict cabin children.
8. **Brightness / sharpness is confounded with video.** By brightness quartile test accuracy falls 0.79, 0.87, 0.48, 0.32 (lowest to highest brightness); by Laplacian sharpness it is 0.78, 0.49, 0.63, 0.56. This is mostly the two videos having different exposure; with 2 videos I cannot separate brightness from identity, so I treat it as a lead, not a cause. Crop size cannot be tested because crops are stored resized to 112 px with no original box size in provenance.csv.
9. **Class mix.** Train 286 / 203 / 771 (male / female / child), 61% child; val 100 / 96 / 111; test 92 / 92 / 180. Source mix of train: 6ckvk 669, baby_adult 204, car 260, ihdwp 127. Adult cabin crops in train are only 221 (car male 101 + female 120) from 3 videos; the other adults are from stock photos and one masked-taxi fleet. ihdwp is 121 men / 6 women, so it adds essentially no female information.
10. **The fine-tune start is hard.** Val F1 after epoch 1 is 0.29 and needs 12 epochs to reach 0.62, i.e. the 2-way pretrained head/backbone plus the new 3-way head and lr x0.3 leave most of the 25 epochs unproductive; peak at epoch 14 and flat after.

### Discrepancy 1,217 on disk vs 1,260 in 05_clean_round3.md (and val 303 vs 307)
Explained. `provenance.csv` has 1,260 train rows and 307 val rows, but 47 rows are exact duplicates of another row on (split, class, file): the same crop file name written twice from the same origin image and group, so the second write overwrote the first. Train 43 (baby_adult child 36, ihdwp male 4, 6ckvk male 2 + female 1) and val 4 (baby_adult child 4) = 47 duplicate rows, which equals 1,260 - 1,217 = 43 and 307 - 303 = 4 exactly. Test has none (364 = 364). Per class: train male 286 -> 280, female 203 -> 202, child 771 -> 735 (matches); val child 111 -> 107. Cause: the file name embeds a truncated copy of the source file name plus a short hash; two boxes in one image (or the same box registered twice) produce the same name. I did not open pixels, so I cannot say whether the two overwritten crops were identical or two different people; names matching and being in the same image suggests duplicates or near-duplicates. Consequence: the trainer saw 1,217 / 303 (correct, no leakage since the same group), but 05_clean_round3.md overstates child by 36 in train and the table is 3.4% too high. The builder should de-duplicate or add a counter to the name. Proposed fix is for the cleaner stage; I did not edit anything.

## 4. Recommendations (ranked)
| # | Type | Action | Expected effect | Effort |
|---|---|---|---|---|
| 1 | More data (exact gap) | Real cabin crops of adult women and men from many different people and cabins. Current: 3 train videos, about 3 women. Needs at least 15-20 distinct women and 15-20 distinct men in different cars, lighting, camera angle, seat positions; at least 5 distinct children aged 2-10. dataset-scout re-run: "adult women in car passenger seat (front and rear), ceiling or dash camera, varied clothing incl. hijab / short hair", "men with long hair or beard", and children of 3 age bands | Biggest lever: female cabin recall (0.19 / 0.43) is a people-diversity problem; every other fix is secondary | High (collection + labelling) |
| 2 | Validation redesign | Use leave-one-video-out or group-by-person CV across all car videos, and report per-person results; add 5+ people to val/test before trusting any number | Honest ranking of models; removes the 0.19 vs 0.43 swing | Medium |
| 3 | Hybrid decision | Offline fusion: face model when YuNet finds a face, this model only when no face. Test macro-F1 0.636 vs 0.574 alone and 0.625 baseline-answered; val tie 0.626. Face threshold >= 0.8 adds 0.004 / -0.005, so not worth tuning. Worth prototyping because it fills in the 21-35% no-face crops with a classifier that is accurate on child and (on val) male, but the male fallback and female fallback are unreliable | Coverage 100% instead of 65-79%; accuracy +0.05 on test, about 0 on val; do not ship on this evidence | Low-medium (wire after more data) |
| 4 | Relabel / audit | Hand-review the 178 `review_pending` crops, then spot-check the 262 face-classifier-labelled train crops, mainly ihdwp women (6) and 6ckvk adults (teens possible). Remove the replicate-padding streak crops or pad with a neutral colour | Cleaner adult labels; modest | Low |
| 5 | Augmentation | Strong colour/brightness/contrast jitter, random erasing, blur, and random partial crops of the top 60%; train with mixup | Reduces memorisation of the 3 train cabins (train 1.00 vs test 0.6) | Low |
| 6 | Hyperparameter | Freeze the backbone for the first 3-5 fine-tune epochs, then unfreeze at a lower lr; early stop at about epoch 14; try label smoothing 0.1 | Faster start (epoch 1 F1 is 0.29), less over-confidence; calibration would be better | Low |
| 7 | Threshold tuning | Add a "not sure" output below a confidence tuned on a cabin validation set; do not use it until a cabin val set of 10+ people exists (current calibration at 0.9-0.99 on test is 0.57) | Fewer wrong labels shown; small until recommendation 1 is done | Low |
| 8 | Data | Add UTK/FairFace ages 0-9, PA-100K AgeLess18 child crops, and teens, in a controlled share, so the child class is not 80% stock babies | Child generalisation; checks that child is not "small crop = child" | Medium |
| 9 | Model size | Do not enlarge for accuracy; MobileNetV3-small is not the limit (pretrain holdout 0.87, train 0.99) | None | n/a |

Edge: no latency was measured for this model. MobileNetV3-small at 112 px is cheap and a fallback-only use (about 20-35% of crops) would cost little; Pi speed remains unmeasured.

## 5. Caveats
- Test is 364 crops from **2 videos, about 7 distinct people** (video 67287def25: 3 children, 1 woman, 1 man; d769e9f0d3: 1 man, 1 woman). Each class-recall figure is effectively 1 to 3 people. All differences of 0.1 to 0.2 (new vs baseline, fusion vs alone) are within what one person changes. Val is also tiny for cars: 2 videos (2af461cf70, d3d258db0c), about 3 people.
- Crops of one person in one video are highly correlated, so the 364 / 303 counts overstate the information (60 frames per person in video 67287def25).
- Val child is 80% stock photos; baseline val numbers for answered subsets (198 / 286) are subsets chosen by the baseline's own face finder and are biased toward easy, frontal faces.
- 262 train labels came from the face classifier (>= 0.9) and 3% of a 29-crop spot check was visibly wrong (05 report); some 6ckvk adults may be teens.
- Baseline was scored on 112 px top-60% crops with the full crop as the box, not on full frames with the detector; production face finding on full frames might behave differently. The scratch baseline unknown rate (35% / 21%) reflects the 112 px crop size.
- The brightness/sharpness analysis is confounded with video; no original resolution is recorded in the crops, so a size-versus-error check was not possible.
- No real cabin footage beyond these 7 car videos (3 train, 2 val, 2 test), and no edge-device timing for this model.
