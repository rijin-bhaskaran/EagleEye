# 01 Scout report (2026-10-28)

Budget: 4 GB. Fetched this run: about 0.54 GB (176 + 102 + 224 + 41 MB). raw/external total incl. earlier Open Images and FairFace: 1.5 GB. Nothing trained or cleaned.

| dataset_id | url | licence | size | labels | use case | fetched | raw path |
|---|---|---|---|---|---|---|---|
| hf_ramankamran_seatbelt_yolov11 | https://universe.roboflow.com/class-oohmp/seatbelt-detection-esut6-p8g6m/dataset/2 (via https://huggingface.co/datasets/ramankamran/seatbelt-detection-v2i-yolov11-lt) | CC BY 4.0 per Roboflow export; HF card none (licence-unverified for the re-upload) | 224 MB (zip 110 + extracted export 110) | YOLO boxes: person-noseatbelt, person-seatbelt; 3,104 images (train 2548, valid 246, test 310), Roboflow-augmented | seatbelt worn / not worn (real photos) | yes; raw files registered only (class set does not cover it); zip's ultralytics code not extracted | passenger_data/raw/external/hf_ramankamran_seatbelt_yolov11/ |
| hf_c3rl_seatbelt_detection | https://huggingface.co/datasets/c3rl/seatbelt-detection | MIT per HF metadata, empty card (licence-unverified) | 176 MB | folder labels seatbelt / no_seatbelt, 2500 each, no boxes | seatbelt | yes; raw registered only. WARNING: sampled images are procedurally generated flat cartoon figures, not cabin photos; likely near-useless | .../hf_c3rl_seatbelt_detection/ |
| hf_slownite_child_detection | https://huggingface.co/datasets/Slownite/ChildDetectionDataset | CC BY-NC-SA 4.0 (non-commercial) | 41 MB | child boxes, 1,928 images (train+test), mapped to child=2 | child appearance (stock-video frames, medical/home/play scenes, NOT cars) | yes; registered via add_external_dataset (items split by hash of name; frames share video_<id> so split by video before use) | .../hf_slownite_child_detection/ |
| hf_py97_utkface_cropped | https://huggingface.co/datasets/py97/UTKFace-Cropped (orig https://susanqq.github.io/UTKFace/) | HF tag MIT, original non-commercial research (licence-unverified) | 102 MB | 23,708 face crops, age and gender in filename | male/female/child face classifier (read by passenger/datasets/utkface.py, left as tar.gz) | yes; raw registered | .../hf_py97_utkface_cropped/ |
| c3rl/seatbelt-detection-v2 | https://huggingface.co/datasets/c3rl/seatbelt-detection-v2 | Apache-2.0 | 3.2 GB | seatbelt (1500) / noseatbelt (1000) | seatbelt | NO: sampled one image, it is an AI-generated studio portrait of a man in a control-room chair, not a car; would eat 80% of budget | - |
| huajianhuakai/seatbelt-detection | HF | MIT | 170 MB | same layout and byte size as c3rl v1 | seatbelt | NO: apparent duplicate of c3rl v1 | - |
| anywaylabs/synthetic-driver-monitoring-detection | https://huggingface.co/datasets/anywaylabs/synthetic-driver-monitoring-detection | CC BY 4.0 | 2.1 GB | driver distraction classes, synthetic | none of the 4 targets | NO: wrong labels, big | - |
| TICaM (DFKI) | https://vizta-tof.kl.dfki.de/download/ | not stated | - | front cabin ToF, 2D/3D boxes, activities, child seats with dolls | cabin person count, child seat | BLOCKED: login/account required | - |
| Open Images "Seat belt" boxes | Open Images V7 val/test | CC BY | few | only 21 boxes in val+test (train bbox CSV is >1 GB) | seatbelt | not fetched: negligible | - |
| SVIRO / Car-cabin-videos / Cabin multi-modal zip | already in C:\Users\RijinBhaskaran\Downloads | see memory car-model-datasets | - | seat-level classes | child seat, count | not re-fetched; never registered under raw/external; still to be copied in | - |

## Blocked: manual steps for the user
- Roboflow Universe sets (Seatbelt Detection, "car-seatbelt", "seatbeltdet", child-seat sets, Inside Bus View 1,378 images): free Roboflow account, open dataset page, Download Dataset, YOLOv8 format, "download zip". Place in raw/external/<id>/ with a SOURCE.json. Only CC BY ones should be preferred when picking.
- Kaggle "lavdeep1234/driver-seat-belt-dectection" (binary, windshield view, ~15:1 imbalance) and other Kaggle seatbelt sets: Kaggle login and API token (`kaggle datasets download -d lavdeep1234/driver-seat-belt-dectection`).
- TICaM and TIMo (DFKI ToF): create account at https://vizta-tof.kl.dfki.de/ then download page.
- DMD / DriveAct-style sets: registration forms (not attempted).

## Gaps remaining
- No real in-car dataset with male/female/child person boxes. All gender/child data are faces (FairFace, UTKFace) or generic photos (Open Images, stock-video children).
- Child seat present: nothing fetched. Only SVIRO (already local, synthetic) and TICaM (login) cover it.
- Seatbelt: one usable real-photo set (Roboflow export); the Roboflow export is augmented, so split by source photo; the other seatbelt set is cartoon.
- Rear-seat/IR cabin person-count data still only from own footage.
- Roughly 3.4 GB of budget unused.

## Ready for the auditor
hf_ramankamran_seatbelt_yolov11 (check label quality and augmented duplicates), hf_slownite_child_detection (domain gap, NC licence), hf_py97_utkface_cropped (dedupe vs FairFace), hf_c3rl_seatbelt_detection (likely discard; user may delete 176 MB). `DataManager.verify_raw()` returns no errors.

## Addendum: Downloads datasets (2026-10-28)

Copied (originals untouched) from C:\Users\RijinBhaskaran\Downloads into raw/external/, each with SOURCE.json, registered raw-only via `register_raw_files` (labels are seats / child seats / none, not male/female/child, so no conversion). `python -m passenger.data_manager verify`: raw data intact. raw/external total now about 3.74 GB (3,741 MiB), under the 4 GB cap.

| dataset_id | origin | licence | size | labels | notes |
|---|---|---|---|---|---|
| roboflow_inside_bus_view_v1 | Roboflow seat-occupancy/inside-bus-view v1 (download folder "busCabin dataset") | Public Domain (own data.yaml/README) | 1,240 MB, 6,609 files, 3,303 images (train 2888 / valid 274 / test 141) | YOLO boxes: 0 Occupied, 1 Unoccupied seats | full copy. Augmented 3x per source photo (split by source); this is the "Inside Bus View" set listed as a candidate in DATASETS.md. |
| car_cabin_videos_stock | Car-cabin-videos | none stated / licence-unverified (stock, watermarked) | 110 MB, 12 mp4 | none | full copy. One clip is 4096x2160. |
| sviro_intercar_subset | SVIRO-InterCar | none stated / licence-unverified (believed CC BY-NC-SA, verify) | 468 MB | seat codes in scene name (0 empty, 1 infant seat, 2 child seat, 3 adult; 3 rear seats) + 2D keypoints txt | SUBSET: 42 of 990 train scenes (every 24th in name order), all 10 car renders each. Full set 10.9 GB. Train split only; no boxes/gender/belts. |
| sviro_intercar_people_subset | SVIRO-InterCar-People | same as above | 453 MB | seat codes in scene name; no keypoints | SUBSET: 42 of 1000 train scenes, same rule. Full set 10.9 GB. |

Not found: a "cabin multi-modal zip". Downloads has no such zip. The only zip there is Cyphera_v03.0.19_Evaluation_Setup.zip (168 MB, apparently unrelated software; not copied or opened) and a CYPHERA prompt .md. The multi-view in-cabin bus dataset (3D boxes, no licence) remains not obtained.

## Addendum: Roboflow and Kaggle round (2026-10-28)

Step 1 (owner decision): `hf_c3rl_seatbelt_detection` (cartoons) removed: raw/external folder deleted, entry dropped from passenger_data/manifest.json (raw_files) via DataManager (`dm.raw_files.pop` + `_save`; no remove API exists). `python -m passenger.data_manager verify` afterwards: raw data intact.

Method: Kaggle REST API and Roboflow API with the owner's credentials from `.env` (no key written anywhere). Candidates were downloaded to a staging folder in the scratchpad, contact sheets viewed, and only passing sets copied into raw/external, each with SOURCE.json, registered with `register_raw_files` (labels are belt state / behaviour, not male/female/child, so `add_external_dataset` does not apply). Nothing cleaned or trained. raw/external total after this round: 3.89 GB (3.49 GB after step 1, +0.40 GB). Cap now ~8 GB, so budget is not the constraint; supply of real in-car data is.

| dataset_id | url | licence | size | labels | real cabin? | notes |
|---|---|---|---|---|---|---|
| kaggle_lavdeep_driver_seatbelt | kaggle.com/datasets/lavdeep1234/driver-seat-belt-dectection | none stated / licence-unverified | 17.6 MB, 1,264 img | image-level folders seat_belt 1177 / no seatbelt 87 | yes, mixed web photos (adults, children, some non-car) | heavy imbalance; no boxes |
| kaggle_sachinx_seatbelt_positives | .../sachinxshrivastav/seatbelt-dataset | none stated / licence-unverified | 32.5 MB, 1,300 img | seat_belt positives only | yes, web photos, children and infant car seats visible | best child-in-seat appearance found, unlabelled for it |
| kaggle_youcef_seatbelt2_subset | .../youcefbahgat/seat-belt | Apache 2.0 (Kaggle) | 77.4 MB, 1,370 img (SUBSET of 8,617 / 486 MB) | Seat_Belt / WithoutSeat_Belt, image-level | yes, side view of drivers, video-still style | every 6th train / 8th test file; near duplicates likely |
| kaggle_firasmahdi_driver_seatbelt_ir | .../firasmahdi/seat-belt-detection-data-science | none stated / licence-unverified | 52.5 MB, 1,321 img | YOLO 8 classes, class 5 Seatbelt-On (no no-belt class) | yes, driver cabin cam, IR and RGB | some rows are polygons |
| kaggle_habbas11_dms_seatbelt_subset | .../habbas11/dms-driver-monitoring-system | Apache 2.0 (Kaggle) | 100.4 MB, 3,939 img (SUBSET of 9,884 / 244 MB) | YOLO, class 4 Seatbelt boxes (+ eyes, cigarette, phone) | mostly yes (IR/RGB driver cabin), some web photos | only images containing a Seatbelt box kept |
| kaggle_mariam_noseatbelt_truckcab | .../mariamabdelghanie/no-seatbelt-v3i-yolov11 | Apache 2.0 (Kaggle); Roboflow README may say CC BY 4.0 | 51.2 MB, 1,587 img | YOLO 0 no_seatblet_detected | yes but TRUCK cabins, driver only | only unbelted-boxes source; domain is trucks |
| kaggle_nexdata_passenger_behavior_sample | .../nexdatafrank/passenger-behavior-recognition-data | CC BY-NC 4.0 | 5.3 MB, 5 img | none | yes, RGB+IR pairs, rear passengers | sample of a commercial set, tiny |
| kaggle_alexandresintes_seatbelt_real_car | .../alexandresintes/seatbelt-detection-dataset-real-car-photos | CC BY 4.0 | 0.4 MB, 8 img | none | yes, one driver | tiny |
| roboflow_seatbelttraining_seatbelt_v1 | universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec (v1, YOLOv8) | CC BY 4.0 | 72.3 MB, 1,629 img | YOLO 0 no-seatbelt, 1 seatbelt (person-in-cabin belt boxes) | yes, grayscale driver webcam views | Roboflow-augmented (rotation, noise): split by source |

Skipped after viewing samples (not fetched into raw; staged copies stay only in the scratchpad):
- Kaggle uzairahmad1434/seatbelt-detection (13 MB) and mariamabdelghanie/seatbelt-detection-v1i (172 MB), Roboflow akaike/seatbelt-detection-vvyjz: roadside or overhead traffic cameras, outside view.
- Roboflow yolov7obj/seat-belt-b93ls (driver and passenger belt classes, 2,205 img): toll-booth front views of cars from outside through the windshield, not a cabin view.
- Roboflow sotukenn/child-in-car (3,263 img): mostly children in homes, parks, plus exterior car photos; no cabin.
- Roboflow carseat-project/car_seat-eqpkt: factory racks of wrapped car seats, not child seats.
- Roboflow yolov8-training-nwfyz/adult-kid and just-for-testing/gender-and-age (Man/Woman/Kid/Baby): generic stock people photos, not cars (a few seat shots).
- Kaggle moh3we5/driver-monitoring (52 MB): one or two people heavily augmented, behaviour labels only. sachinmlwala/seatbelt3: only a .pt model and 2 videos (not data; model file not opened).
- Not fetched, not viewed: huynhtan/seatbelt001 (1.4 GB, unknown licence), smartcity12/adms-dataset (928 MB), abdoreldeeb/driver-monitoring-system-dataset (2 GB, CC BY 4.0, Ala Mahlak DMS: possible cabin value, worth a look if the owner wants more driver/IR data), yehiahassanain/seat-belt2 (same size as youcef, likely same data). Roboflow jui/driver-behaviors (9,901 img, seatbelt boxes) likely duplicates the DMS set above. SVIRO-seg on Roboflow (133 img, infant/child seat classes) is synthetic and a re-upload of SVIRO already held; occupant_classification_model1/2 (adult_occupant, child_occupant, empty_seat, lapbelt) could not be located by workspace slug without the Universe site search (no API for search).

Key finding: no real in-car dataset with male/female/child person labels exists in the Roboflow/Kaggle results I could reach. Everything new is seatbelt state (driver mostly), image-level or person-level, plus IR/night driver frames.

Remaining gaps: real in-cabin male/female/child on whole bodies; real child_seat boxes (only photos, no boxes: sachinx, lavdeep contain child seats unlabelled; SVIRO synthetic); belt boxes from inside with passengers (all new sets are driver-centric); rear-seat passengers (nexdata sample only, 5 images); IR cabin with labels is driver-only (firasmahdi, habbas11). The user's own footage remains the only route for the male/female/child classes, or a licensed set (TICaM, a full Nexdata purchase).

## Addendum: real-car round (2026-10-28)

Owner decisions: not commercial (any licence OK, recorded), internal training only, soft cap ~8 GB for raw/external. Sets 1-4 copied from the scratchpad stage3 (file counts matched staged copies exactly); PA-100K downloaded from Kaggle with the owner's `.env` credentials (zip 453 MB, extracted with a path-traversal check, never executed). Each has a full SOURCE.json (url, licence as stated by uploader, content_rights "unverified", retrieval date, selection rule, classes, counts) and is registered with `register_raw_files` (raw-only; no label conversion invented). `python -m passenger.data_manager verify`: raw data intact.

| dataset_id | url | licence (uploader) | size / counts | labels | notes |
|---|---|---|---|---|---|
| rf_baby-adult-in-car_v1 | universe.roboflow.com/jhonatans-workspace-c7rlt/baby-adult-in-car | CC BY 4.0 | 7.6 MB, 213 img (train 171 / valid 42) | YOLO boxes: 0 adult, 1 child | in-car; could map adult/child later but no male/female, so raw-only |
| rf_seat-belt-ihdwp_v1 | universe.roboflow.com/ali-matar-tscn0/seat-belt-ihdwp | CC BY 4.0 | 57 MB, 1,035 img (850/124/61) | YOLO boxes: 0 passenger | taxi cabin person boxes, no gender/age |
| rf_passenger-gender-dataset_v3 | universe.roboflow.com/passenger-head-detection/passenger-gender-dataset | CC BY 4.0 | 48 MB, 1,110 img (1053/39/18) | YOLO head boxes: 0 man, 1 woman | bus; augmented, split by source photo; no child class |
| rf_child-detection-6ckvk_v4 | universe.roboflow.com/child-detection-0zzkz/child-detection-6ckvk | CC BY 4.0 | 308 MB, 2,829 img (2293/352/184) | YOLO boxes: 0 Teddy Bear, 1 adult, 2 child, 3 doll | dolls/teddies are hard negatives; filter in cleaning |
| kaggle_pa100k | kaggle.com/datasets/yuulind/pa-100k (Liu et al. 2017) | CC BY 4.0 per Kaggle card | 457 MB, 100,000 img (80k/10k/10k) | image-level, 26 attributes in train/val/test.csv + annotation.mat (Female, AgeOver60, Age18-60, AgeLess18, pose, clothing) | full-body pedestrian crops, not in-car. Female: 36,492 / 4,948 / 3,896. AgeLess18: 4,152 / 1,443 / 92 (test has few children). Child is an age flag, not a third gender class, so no male/female/child conversion done |

raw/external total after this round: 4.71 GB (153,552 files). Free disk about 90 GB.
Scratchpad housekeeping: staging folders for candidate sets not copied (all other rf_*, hf_opensparx_*, kaggle_pa100k_sample) were deleted from stage3; they can be re-fetched via the Roboflow API/HF if wanted.
