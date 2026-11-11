# Passenger models: metrics and pipeline recommendation

Date: 2026-10-29. Nothing described here is wired into the pipeline or `config.yaml`. Every number comes from files in this repo; "estimate" marks anything that is not measured.

## 1. What was built

| Part | What | Where |
|---|---|---|
| Person detector | YOLOv8n fine-tuned from COCO, one class `person`, 416 px, 37 epochs (best = epoch 37) | `models/passenger/person_det/v2/run/weights/best.pt` |
| Attribute classifier | MobileNetV3-small on a 112 px person crop (top 60%), male / female / child; PA-100K pretraining, then 25 epochs on cabin crops; 3 seeds averaged | `models/passenger/crops/v4_seed0..2/model.pt` |
| Calibration | log-probability offsets (male 0, female +2.0, child +0.5), chosen on validation | `models/passenger/crops/v4_blend.json` |
| Data | `person_det_v2` (train 1,579 / val 355 / test 238 images), `crops_mfc_v4` (train 2,057 / val 609 / test 611 crops) | `passenger_data/annotations/` |
| Charts and tables | per-epoch train / validation curves with the final test point | `passenger_data/agent_reports/training_curves/` |

Not built: seatbelt, no-seatbelt and child-seat models. Their exports exist (`seatbelt_v1`, `child_seat_synth_v1`) but nothing was trained on them, so the `car:` mode in `config.yaml` (one 6-class model) has no model behind it.

## 2. Person detector

Test = 8 cabin videos (213 frames, about 20 people) + 25 child photos. Validation = same kinds of sources, different people.

| | Fine-tuned (epoch 37) | COCO YOLOv8n (what the pipeline's detector is built on) |
|---|---|---|
| Validation P / R | 0.947 / 0.906 | 0.945 / 0.883 |
| Validation mAP50 / mAP50-95 | 0.964 / 0.777 | 0.954 / 0.806 |
| Test P / R | 0.881 / 0.764 | 0.922 / 0.678 |
| Test mAP50 / mAP50-95 | 0.835 / 0.624 | 0.814 / 0.641 |
| Cabin count error on test, conf 0.35 (MAE / exact) | 0.63 / 53% | 0.47 / 64% |
| Count bias on test cabin frames | -0.16 (undercounts) | +0.32 (overcounts) |

Findings
- Fine-tuning raised recall (test +0.09) and mAP50, and lowered precision and box tightness. It counts worse than COCO at the 0.35 threshold. It is not a clear win.
- Count error is concentrated in crowded frames. Test frames with 5 people (60 of 213, one family video): fine-tuned MAE 1.33, exact 17%; COCO MAE 0.92, exact 35%. No training or validation cabin frame has more than 3 people.
- A confidence threshold chosen on validation did not transfer (fine-tuned 0.60 gave test MAE 0.71; COCO 0.45 gave 0.49). Validation never contains crowded frames, so it cannot choose a threshold for them.
- Training-run notes: the laptop restarted overnight (about 04:00) and stopped the run after epoch 31; it was resumed. A bug in my stop rule (mixed fitness definitions after a resume) ended the run at epoch 37 instead of continuing to the cap of 40. Converged epochs, small loss.

## 3. Male / female / child classifier

Test = 611 crops (242 male, 144 female, 225 child) from 14 person groups (about 20 people, 6 women). Validation = 609 crops, 102 groups. The face model was left out of the final design (section 3.3).

### 3.1 Results (test, scored once)

| | Accuracy | Macro-F1 | Balanced acc. | Male / female / child recall | Spread* | Worst class |
|---|---|---|---|---|---|---|
| Seed average, no calibration | 0.787 | 0.758 | 0.754 | 0.91 / 0.53 / 0.81 | 0.38 | 0.535 |
| Face model when a face is found, else crop model | 0.671 | 0.662 | 0.678 | 0.48 / 0.69 / 0.86 | 0.38 | 0.483 |
| **Seed average + calibration (chosen)** | 0.771 | 0.764 | 0.779 | 0.80 / 0.85 / 0.69 | 0.16 | 0.689 |

*Spread = best class recall minus worst class recall; lower is more even.

Validation, for reference (optimistic, the calibration was chosen here): macro-F1 0.834, spread 0.03, recall 0.82 / 0.84 / 0.85.

### 3.2 How sure we can be (cluster bootstrap over person groups, 95% interval)

| | Macro-F1 | Spread | Worst class | Accuracy |
|---|---|---|---|---|
| Test, uncalibrated | 0.758 [0.62, 0.93] | 0.38 [0.11, 0.62] | 0.54 [0.30, 0.88] | 0.79 [0.70, 0.96] |
| Test, calibrated | 0.764 [0.65, 0.95] | 0.16 [0.03, 0.40] | 0.69 [0.54, 0.93] | 0.77 [0.68, 0.96] |
| Validation, uncalibrated | 0.754 [0.65, 0.89] | 0.53 [0.20, 0.73] | 0.42 [0.23, 0.79] | 0.79 [0.70, 0.92] |
| Validation, calibrated | 0.834 [0.75, 0.89] | 0.03 [0.02, 0.24] | 0.82 [0.65, 0.87] | 0.84 [0.77, 0.90] |

- On validation (102 groups) the evenness gain is clear: the uncalibrated and calibrated spread intervals barely overlap.
- On test (14 groups) the intervals overlap. The direction is the same but the test set is too small to prove it.
- The accuracy cost of calibration on test (-0.016) is well inside the noise.

### 3.3 Sensitivity of the calibration (validation only)

| Female offset | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| Macro-F1 | 0.743 | 0.782 | 0.833 | 0.834 | 0.834 | 0.814 | 0.771 |
| Spread | 0.59 | 0.44 | 0.25 | 0.12 | **0.03** | 0.13 | 0.24 |

- Macro-F1 is flat from 1.0 to 2.0 and falls beyond 2.5, so the setting is not at the edge of a cliff.
- The very low spread at 2.0 is a sharp optimum. A female offset of 1.5 to 2.0 gives almost the same macro-F1 with a spread of 0.03 to 0.12. I would expect real-world evenness nearer the test value (0.16) than the validation value (0.03).
- The face model was given weight 0 by the search, and the "face when found" rule is clearly worse. My earlier suggestion to combine the face model with the crop model does not hold for this stronger classifier.

### 3.4 Other measurements
- Seed noise (validation macro-F1): 0.74, 0.77, 0.78 for the three seeds; test 0.71 to 0.76. Averaging gave 0.758 on test.
- Per-frame cost on this laptop CPU: 3-seed ensemble, one crop, 24 ms (8 ms per network). Raspberry Pi: not measured.
- Training curves: train loss falls to 0.05 while validation loss rises from 0.65 (epoch 14) to 0.83 (epoch 25): overfitting to the few people in the training set. Charts: `training_curves/classifier_crops_v4_per_epoch.png`.

## 4. What these numbers do not tell us
- The test set is about 20 people, 6 women and 3 test children from one family video plus 7 or 8 children in photos taken from a child's eye level. One person changes a class recall by about 0.1 to 0.3.
- No data comes from your own camera. Everything is stock footage with watermarks, public photos and synthetic images. Rights are unverified: internal training only.
- One labeller (me) made the cabin labels; the face model disagreed on about 11% of boxes and I kept my own reading.
- Children in training: about 18 distinct real in-car children, mostly eye-level rear-seat photos.
- Per-frame scores only. The pipeline smooths each tracked person over about 15 frames, which should help, but that has not been measured.

## 5. Pipeline recommendation (proposal only, nothing applied)

Principle: add only what the evidence supports, behind an option, with the current behaviour as the default.

1. Detector: keep the COCO YOLOv8n person detector at conf 0.35. Do not switch to the fine-tuned model. Revisit when there is labelled data with crowded cabins (4 to 6 people) and one consistent box style (whole upper body).
2. Classifier: add a new classifier option next to `levi_hassner` and `torch` in `passenger.classifier`, for example `model: crops`. It takes the person box rather than a face, crops the top 60% to a square, resizes to 112 px, averages the 3 networks, adds the log offsets (0, +2.0, +0.5) and returns the label. Keep the face-based path as the fallback if the weights are missing. Drop the face model from the crop path (weight 0).
3. Keep the existing per-person smoothing and the background thread. Classify at most a few people per frame; 3 networks at 8 ms each means 5 people is about 120 ms if done every frame, which the existing bounded-calls design avoids.
4. Add an "unknown" output: below a probability threshold the label is `unknown`. Choose the threshold on validation, never on test, and report coverage next to accuracy as the current code does.
5. Tests first, as in `WORKFLOW.md`: loading the 3 weights, offsets applied in the right order (male, female, child), box-to-crop geometry, unknown handling, and a missing-weights fallback test. Then the QA gate.
6. Measure on the Raspberry Pi before enabling: latency per crop, and whether one network (or an ONNX/OpenVINO export) is enough. If speed forces a single network, expect macro-F1 about 0.71 to 0.76 and re-check evenness.
7. Do not enable `vehicle_type: car` with the 6-class model: no such model exists. Seatbelt and child-seat detection remain separate, untrained workstreams.
8. Release gate (suggested): macro-F1 at or above 0.75 and worst-class recall at or above 0.65 on a fresh test set recorded with your own camera, with at least 15 distinct women, 15 distinct men and 5 children (ages 2 to 10), including front and rear seats, night/IR and crowded cabins. This is the single most valuable thing that could be added.

Proposed config shape (not applied):

```yaml
passenger:
  classifier:
    model: crops                    # levi_hassner | torch | crops
    crops:
      weights: [models/passenger/crops/v4_seed0/model.pt, models/passenger/crops/v4_seed1/model.pt, models/passenger/crops/v4_seed2/model.pt]
      log_offsets: {male: 0.0, female: 2.0, child: 0.5}
      min_prob: null                # to be chosen on validation
```

## 6. Known open items
- QA gate overhead check: failed at 15.2, 19.6 and 17.6 ms (budget 15) three times right after long test runs; an isolated benchmark reads 3.3 to 5.8 ms. Cause not proven (CPU load or heat is my guess). Re-check on an idle machine.
- Unreviewed material: 53 least-sure frames of `car_cabin_v1`, 144 least-sure frames of `car_cabin_v2`, 178 gender crops in `crops_mfc_v1/review_pending`. All labels are one labeller's judgement.
- Stray folder `C:\c` (7.3 MB) from a path slip; not deleted.
- Overnight run risk: a Windows restart stopped the detector run; a startup task to resume after reboots has not been added.
