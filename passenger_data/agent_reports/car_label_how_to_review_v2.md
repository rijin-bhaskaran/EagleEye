# How to spot-check the new cabin labels (v2)

1. Open `car_label_review_v2.csv`. Sort by `least_sure`, then `model_disagrees`. Look at those rows first.
2. For each row open `car_label_previews_v2/<video_id>/<frame>.jpg` (boxed copy) or the per-video `_contact_sheet.jpg`.
   Colours: blue = male, magenta = female, orange = child. Check: every visible person boxed, the right class, box roughly tight.
3. Write `ok` or `wrong` in the last column. For a wrong frame either delete its label file in `passenger_data/annotations/labels/` or
   correct it (YOLO txt, class 0 male / 1 female / 2 child) and tell the assistant to re-import and re-export.
4. `car_labels_original_v2/` holds my labels as first written: diff your edits against it.
5. `car_label_dropped_v2.csv` lists every selected frame that was not labelled and why.
Per person, the class was decided once per clip, by eye; the repo face classifier was only a cross-check (it disagrees on about 1 face in 10).
