# How to spot-check the car cabin labels

1. Open `passenger_data/agent_reports/car_label_previews/<video_id>/_contact_sheet.jpg` for an overview, then single frames `<video_id>_<frame>.jpg` (boxes: blue = male, magenta = female, orange = child; the word is the label).
2. `car_label_review.csv` has one row per labelled frame: split, boxes per class, a note per person, and three flags: `least_sure` (my doubt: low detector score, overlapping boxes, hidden toddler in car-vid-3), `model_disagrees` (the repo face classifier said something else; I judged by eye), `approx_box` (child box derived from the face: class certain, box rough). Start with the flagged rows, then sample about 10% of the rest.
3. Mark a wrong frame: type `wrong` in the last column `owner_verdict` (add a note if you like). Frames I dropped are in `car_label_dropped.csv` with the reason. Whole videos left out are listed in `04_car_labels.md`.
4. Fix and re-import:
   - Drop a frame: delete `passenger_data/annotations/labels/<video_id>_<frame>.txt`.
   - Change a class or box: edit that txt (one line per person: `class cx cy w h`, values 0-1, class 0 male, 1 female, 2 child), or upload `passenger_data/processed/frames/<video_id>/` to CVAT, correct, export YOLO 1.1 and run `python -m passenger.data_manager import-labels <export folder>` (overwrites the same frame names).
   - My original labels are kept in `car_labels_original/` so you can diff your edits.
   - Then run `validate`, `stats`, and rebuild under a NEW name (`export car_cabin_v2 --source videos`); v1 stays as it is. Any edit changes the split fingerprint, so scores before and after are not comparable.
