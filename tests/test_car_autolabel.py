"""Tests for passenger.car_autolabel: frames, names and paths, labels, review sheets, corrections, export, and the label stages."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger import car_autolabel as ca
from passenger.car_logic import CAR_CLASSES
from passenger.datasets import car_cabin


# -- helpers --------------------------------------------------------------------
def make_video(path: Path, frames: int = 50, fps: int = 25, size=(160, 90), moving: bool = True) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        if moving:
            cv2.rectangle(img, (i * 2 % size[0], 10), (i * 2 % size[0] + 30, 60), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


def make_run(tmp_path: Path, frames: dict) -> Path:
    """A run dir with the given frames: {fid: (status, [box dicts])}."""
    run = tmp_path / "root" / "autolabel" / "r1"
    (run / "images").mkdir(parents=True)
    m = {"run": "r1", "classes": list(CAR_CLASSES), "videos": {"v": {}}, "frames": {}}
    for fid, (status, boxes) in frames.items():
        cv2.imwrite(str(run / "images" / f"{fid}.jpg"), np.full((90, 160, 3), 120, np.uint8))
        m["frames"][fid] = {"video": "v", "frameno": 1, "file": f"{fid}.jpg", "width": 160, "height": 90,
                            "license": "CC BY 4.0", "url": "http://x", "status": status, "reject_reason": None, "boxes": boxes}
        ca.write_label(run, fid, m["frames"][fid])
    ca.save_manifest(run, m)
    return run


def B(cls, box=(0.1, 0.1, 0.5, 0.9), conf=0.9, source="face", queue=None, kind="person"):
    return ca.new_box(kind, box, cls, conf, source, queue)


def write_csv(path: Path, rows, header=("frame", "box_index", "new_class", "x1", "y1", "x2", "y2")):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# -- names and paths --------------------------------------------------------------
@pytest.mark.parametrize("name", ["../../evil.mp4", "C:\\Windows\\x.mp4", "a b\\c.mp4", "..", "", "...mp4", "ünï.mp4", "a" * 300 + ".mp4"])
def test_safe_id_is_a_plain_component(name):
    s = ca.safe_id(name)
    assert s and len(s) <= 48
    assert set(s) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def test_safe_id_unique_among_taken():
    a = ca.safe_id("clip one.mp4")
    b = ca.safe_id("clip one.mp4", taken={a})
    assert a == "clip_one" and b != a and b.startswith("clip_one")
    assert ca.safe_id("clip one.mp4", taken={a, b}) not in {a, b}


def test_frame_id_pads():
    assert ca.frame_id("v", 7) == "v_000007"


def test_run_path_confined(tmp_path):
    assert ca.run_path(tmp_path, "good-1").parent == (tmp_path / "autolabel").resolve()
    for bad in ["../x", "a/b", "a\\b", "..", ".", "", "x" * 80, "/abs", "C:evil", ".hidden", None]:
        with pytest.raises(ca.AutolabelError):
            ca.run_path(tmp_path, bad)


def test_reset_run_only_touches_run_outputs(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    (run / "keep.txt").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("x")
    ca.reset_run(run)
    assert not (run / "images").exists() and not (run / "manifest.json").exists()
    assert (run / "keep.txt").exists() and (outside / "f.txt").exists()


def test_reset_run_on_missing_dir_is_a_noop(tmp_path):
    ca.reset_run(tmp_path / "nothing")


# -- licences ---------------------------------------------------------------------
def test_read_licenses_loose_columns(tmp_path):
    p = tmp_path / "MANIFEST.csv"
    p.write_text("Filename,Licence,Source URL\nvideos\\Clip A.mp4,CC0,http://a\nb.mp4,,http://b\n,CC0,x\n", encoding="utf-8")
    d = ca.read_licenses(p)
    assert d["clip a"] == {"license": "CC0", "url": "http://a"}
    assert d["b"]["license"] == "unknown"
    assert len(d) == 2


def test_read_licenses_missing_file(tmp_path):
    assert ca.read_licenses(tmp_path / "none.csv") == {}
    assert ca.read_licenses(None) == {}


# -- frames -----------------------------------------------------------------------
def test_extract_frames_rate_names_and_dedup(tmp_path):
    v = make_video(tmp_path / "v.avi", frames=50, fps=25)
    out = tmp_path / "o"
    fr = ca.extract_frames(v, out, "vid", fps=5, dedup=0.0)
    assert [f["frameno"] for f in fr] == list(range(0, 50, 5))
    assert (out / "vid_000005.jpg").is_file() and fr[1]["id"] == "vid_000005"
    still = make_video(tmp_path / "s.avi", frames=50, moving=False)
    assert len(ca.extract_frames(still, tmp_path / "o2", "s", fps=5, dedup=2.0)) == 1       # nothing changes: one frame


def test_extract_frames_max_frames_and_resize(tmp_path):
    v = make_video(tmp_path / "v.avi", frames=50, size=(320, 180))
    fr = ca.extract_frames(v, tmp_path / "o", "v", fps=25, max_frames=3, dedup=0.0, max_side=100)
    assert len(fr) == 3 and max(fr[0]["width"], fr[0]["height"]) == 100
    assert max(cv2.imread(str(tmp_path / "o" / fr[0]["file"])).shape[:2]) == 100


def test_extract_frames_unreadable_and_missing(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(ca.AutolabelError):
        ca.extract_frames(bad, tmp_path / "o", "bad")
    with pytest.raises(ca.AutolabelError):
        ca.extract_frames(tmp_path / "missing.mp4", tmp_path / "o", "m")


def test_extract_frames_empty_video_gives_nothing(tmp_path):
    v = tmp_path / "empty.avi"
    cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 64)).release()
    try:
        assert ca.extract_frames(v, tmp_path / "o", "e") == []
    except ca.AutolabelError:
        pass                                                      # some backends refuse to open a frameless file: also fine


def test_extract_frames_bad_arguments(tmp_path):
    v = make_video(tmp_path / "v.avi")
    for kw in ({"fps": 0}, {"fps": -1}, {"max_frames": 0}):
        with pytest.raises(ca.AutolabelError):
            ca.extract_frames(v, tmp_path / "o", "v", **kw)


def test_is_duplicate():
    a = np.zeros((18, 32), np.float32)
    assert not ca.is_duplicate(None, a, 2.0)
    assert ca.is_duplicate(a, a + 1, 2.0)
    assert not ca.is_duplicate(a, a + 10, 2.0)


def test_find_videos(tmp_path):
    (tmp_path / "sub").mkdir()
    for n in ("a.mp4", "sub/b.MOV", "c.txt", "d.jpg"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in ca.find_videos(tmp_path)] == ["a.mp4", "b.MOV"]
    assert ca.find_videos(tmp_path / "a.mp4") == [tmp_path / "a.mp4"]
    assert ca.find_videos(tmp_path / "c.txt") == []
    with pytest.raises(ca.AutolabelError):
        ca.find_videos(tmp_path / "nope")


# -- boxes and labels -------------------------------------------------------------
def test_new_box_clips_and_validates():
    b = ca.new_box("person", (-0.2, 0.1, 1.5, 0.9), "male", 0.91234, "face")
    assert b["box"] == [0.0, 0.1, 1.0, 0.9] and b["conf"] == 0.912
    with pytest.raises(ca.AutolabelError):
        ca.new_box("person", (0, 0, 1, 1), "dog", 1, "x")
    assert ca.new_box("person", (0, 0, 1, 1), None, 0.1, "face", "person_unreadable")["cls"] is None


def test_yolo_lines_skip_unresolved_deleted_degenerate():
    f = {"boxes": [B("male", (0.0, 0.0, 0.5, 1.0)), B(None, queue="q"), B("female"), B("child", (0.3, 0.3, 0.3, 0.9))]}
    f["boxes"][2]["deleted"] = True
    assert ca.yolo_lines(f) == ["0 0.250000 0.500000 0.500000 1.000000"]
    assert ca.yolo_lines({"boxes": []}) == []


def test_load_manifest_errors(tmp_path):
    with pytest.raises(ca.AutolabelError):
        ca.load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("{nope")
    with pytest.raises(ca.AutolabelError):
        ca.load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("[]")
    with pytest.raises(ca.AutolabelError):
        ca.load_manifest(tmp_path)


# -- review sheets ----------------------------------------------------------------
def test_make_sheets_queue_only_and_index(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")]), "v_000002": ("auto", [B(None, queue=ca.Q_UNKNOWN_PERSON)]),
                              "v_000003": ("reviewed", [B(None, queue=ca.Q_UNKNOWN_PERSON)])})
    names = ca.make_sheets(run, "queue")
    assert names == ["sheet_001.jpg"]
    assert json.loads((run / "review" / "index.json").read_text()) == {"sheet_001.jpg": ["v_000002"]}
    assert cv2.imread(str(run / "review" / names[0])) is not None
    assert len(ca.make_sheets(run, "all", per_sheet=2)) == 2
    assert len(ca.make_sheets(run, "auto")) == 1


def test_make_sheets_empty_and_bad_args(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    assert ca.make_sheets(run, "queue") == []
    for kw in ({"which": "x"}, {"per_sheet": 0}, {"cols": 0}):
        with pytest.raises(ca.AutolabelError):
            ca.make_sheets(run, **kw)


def test_make_sheets_removes_only_its_own_old_sheets(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B(None, queue="q")])})
    (run / "review").mkdir()
    (run / "review" / "sheet_009.jpg").write_bytes(b"x")
    (run / "review" / "notes.txt").write_text("mine")
    ca.make_sheets(run, "queue")
    assert not (run / "review" / "sheet_009.jpg").exists() and (run / "review" / "notes.txt").exists()


def test_draw_frame_and_contact_sheet_shapes():
    f = {"boxes": [B("male"), B(None, queue="q")]}
    out = ca.draw_frame(np.zeros((90, 160, 3), np.uint8), f, width=320)
    assert out.shape == (180, 320, 3) and out.any()
    sheet = ca.contact_sheet([out, out, out], 2, ["a", "b", "c"])
    assert sheet.shape[1] == 640 and sheet.shape[0] == 2 * (180 + 26)
    assert ca.contact_sheet([], 3, []).size > 0


def test_box_sheet(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male"), B("female", (0.0, 0.0, 0.0, 0.0))])})
    out = tmp_path / "boxes.jpg"
    ca.box_sheet(run, [("v_000001", 0), ("v_000001", 1)], out, cols=2, size=64)       # the empty crop is skipped, not a crash
    assert cv2.imread(str(out)) is not None


# -- corrections ------------------------------------------------------------------
def test_apply_relabel_delete_add_reject(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B(None, queue="q"), B("male")]), "v_000002": ("auto", [B("female")]),
                              "v_000003": ("auto", [B("child")])})
    c = write_csv(tmp_path / "c.csv", [["v_000001", 0, "female", "", "", "", ""], ["v_000001", 1, "delete", "", "", "", ""],
                                       ["v_000002", "", "seatbelt", 0.2, 0.2, 0.4, 0.6], ["v_000003", "", "reject", "", "", "", ""]])
    r = ca.apply_corrections(run, c)
    assert r == {"frames_touched": 3, "relabel": 1, "delete": 1, "add": 1, "reject": 1, "keep": 0}
    m = ca.load_manifest(run)
    f1 = m["frames"]["v_000001"]
    assert f1["status"] == "reviewed" and f1["boxes"][0]["cls"] == "female" and f1["boxes"][0]["source"] == "human"
    assert f1["boxes"][1]["deleted"] and not ca.unresolved(f1)
    assert (run / "labels" / "v_000001.txt").read_text().startswith("1 ")
    assert m["frames"]["v_000002"]["boxes"][1]["cls"] == "seatbelt" and len(ca.yolo_lines(m["frames"]["v_000002"])) == 2
    assert m["frames"]["v_000003"]["status"] == "rejected"


def test_apply_keep_marks_reviewed_without_changes(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    ca.apply_corrections(run, write_csv(tmp_path / "c.csv", [["v_000001", "", "keep", "", "", "", ""]]))
    assert ca.load_manifest(run)["frames"]["v_000001"]["status"] == "reviewed"


@pytest.mark.parametrize("row", [
    ["nope", 0, "male", "", "", "", ""],                       # unknown frame
    ["v_000001", 5, "male", "", "", "", ""],                   # no such box
    ["v_000001", "x", "male", "", "", "", ""],                 # box index not a number
    ["v_000001", 0, "dog", "", "", "", ""],                    # unknown class
    ["v_000001", "", "male", "", "", "", ""],                  # add without coordinates
    ["v_000001", "", "male", 0.5, 0.5, 0.4, 0.9],              # inverted box
    ["v_000001", "", "male", 0.5, 0.5, 1.4, 0.9],              # outside 0-1
    ["v_000001", "", "dog", 0.1, 0.1, 0.4, 0.9],               # add with bad class
])
def test_apply_bad_row_aborts_everything(tmp_path, row):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    before = (run / "manifest.json").read_text()
    good = ["v_000001", 0, "female", "", "", "", ""]
    with pytest.raises(ca.AutolabelError):
        ca.apply_corrections(run, write_csv(tmp_path / "c.csv", [good, row]))
    assert (run / "manifest.json").read_text() == before                  # the good row was not applied either


def test_apply_deleted_box_cannot_be_edited_again_and_bad_header(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    ca.apply_corrections(run, write_csv(tmp_path / "c.csv", [["v_000001", 0, "delete", "", "", "", ""]]))
    with pytest.raises(ca.AutolabelError):
        ca.apply_corrections(run, write_csv(tmp_path / "c2.csv", [["v_000001", 0, "male", "", "", "", ""]]))
    with pytest.raises(ca.AutolabelError):
        ca.apply_corrections(run, write_csv(tmp_path / "c3.csv", [["v_000001", 0, "male"]], header=("frame", "idx", "cls")))
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ca.AutolabelError):
        ca.apply_corrections(run, empty)


def test_apply_empty_csv_changes_nothing(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")])})
    r = ca.apply_corrections(run, write_csv(tmp_path / "c.csv", []))
    assert r["frames_touched"] == 0 and ca.load_manifest(run)["frames"]["v_000001"]["status"] == "auto"


# -- export -----------------------------------------------------------------------
def test_export_only_trustworthy_frames(tmp_path):
    run = make_run(tmp_path, {
        "v_000001": ("auto", [B("male", conf=0.95), B("seatbelt", (0.2, 0.3, 0.4, 0.6), 0.9, "belt", kind="belt")]),
        "v_000002": ("auto", [B("female", conf=0.6)]),                              # low confidence
        "v_000003": ("auto", [B("male", conf=0.95), B(None, queue=ca.Q_UNKNOWN_PERSON)]),  # unresolved
        "v_000004": ("reviewed", [B("child", conf=1.0, source="human")]),
        "v_000005": ("reviewed", [B(None, queue="q")]),                                # reviewed but still unresolved
        "v_000006": ("rejected", [B("male")]),
        "v_000007": ("auto", []),                                                      # no boxes
    })
    out = tmp_path / "export"
    r = ca.export_dataset(run, out)
    assert r["frames"] == 2 and r["boxes"]["male"] == 1 and r["boxes"]["seatbelt"] == 1 and r["boxes"]["child"] == 1
    assert r["skipped"] == {"low_confidence": 1, "unresolved_boxes": 2, "rejected": 1, "no_boxes": 1}
    assert sorted(p.name for p in (out / "images").iterdir()) == ["v_000001.jpg", "v_000004.jpg"]
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["v_000004"]["license"] == "CC BY 4.0" and prov["v_000001"]["label_sources"] == ["belt", "face"]
    assert "male" in (out / "data.yaml").read_text()
    assert r["licenses"] == ["CC BY 4.0"]


def test_export_without_auto(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male")]), "v_000002": ("reviewed", [B("male")])})
    r = ca.export_dataset(run, tmp_path / "e", include_auto=False)
    assert r["frames"] == 1 and r["skipped"] == {"auto_not_included": 1}


def test_export_refuses_unsafe_or_busy_out_dir(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("reviewed", [B("male")])})
    for bad in (run, run / "sub", run.parent):
        with pytest.raises(ca.AutolabelError):
            ca.export_dataset(run, bad)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("mine")
    with pytest.raises(ca.AutolabelError):
        ca.export_dataset(run, busy)
    assert (busy / "x").read_text() == "mine"


def test_export_with_nothing_to_export(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("rejected", [B("male")])})
    r = ca.export_dataset(run, tmp_path / "e")
    assert r["frames"] == 0 and (tmp_path / "e" / "data.yaml").is_file()


def test_export_is_accepted_by_the_car_cabin_importer_with_identity_map(tmp_path):
    run = make_run(tmp_path, {f"vidA_{i:06d}": ("reviewed", [B("male"), B("seatbelt", (0.2, 0.3, 0.4, 0.6), kind="belt")]) for i in range(1, 4)})
    out = tmp_path / "e"
    ca.export_dataset(run, out)
    data = tmp_path / "car_data"
    entry = car_cabin.add(out, data, "auto1", {c: c for c in CAR_CLASSES}, license="CC BY 4.0")
    assert sum(entry["images"].values()) == 3
    assert len({s for s in car_cabin.SPLITS if entry["images"][s]}) == 1        # one video -> one split
    assert entry["boxes"]["train"].get("male", 0) + entry["boxes"]["val"].get("male", 0) + entry["boxes"]["test"].get("male", 0) == 3


# -- stats ------------------------------------------------------------------------
def test_run_stats(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("auto", [B("male"), B(None, queue=ca.Q_BELT_UNSURE)]), "v_000002": ("reviewed", [B("child")]),
                              "v_000003": ("auto", [B("female")]), "v_000004": ("rejected", [])})
    s = ca.run_stats(run)
    assert s["frames"] == 4 and s["frames_in_review_queue"] == 1 and s["review_queue_share"] == 0.25
    assert s["status"] == {"auto": 2, "reviewed": 1, "rejected": 1}
    assert s["boxes"]["male"] == 1 and s["unresolved_boxes_by_reason"] == {ca.Q_BELT_UNSURE: 1}


# -- label stages -----------------------------------------------------------------
from passenger.classifier import Attribute  # noqa: E402


def test_refine_people_drops_tiny_wide_duplicates_and_contained():
    dets = [((0.1, 0.1, 0.5, 0.9), 0.9),
            ((0.11, 0.1, 0.5, 0.88), 0.7),                 # same person twice (IoU)
            ((0.15, 0.2, 0.45, 0.6), 0.6),                 # almost wholly inside the first
            ((0.6, 0.6, 0.62, 0.62), 0.9),                 # tiny
            ((0.0, 0.0, 0.9, 0.2), 0.9),                   # wide band
            ((0.55, 0.1, 0.95, 0.9), 0.5),                 # a second person
            ((0.5, 0.5, 0.5, 0.9), 0.9)]                   # zero width
    out = ca.refine_people(dets)
    assert [b for b, _ in out] == [(0.1, 0.1, 0.5, 0.9), (0.55, 0.1, 0.95, 0.9)]
    assert ca.refine_people([]) == []


def test_refine_people_clips_to_frame():
    out = ca.refine_people([((-0.2, -0.1, 0.6, 1.4), 0.8)])
    assert out == [((0.0, 0.0, 0.6, 1.0), 0.8)]


def test_iou():
    assert ca.iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert ca.iou((0, 0, 0.4, 0.4), (0.5, 0.5, 1, 1)) == 0.0
    assert ca.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def strap_image(with_strap: bool) -> np.ndarray:
    img = np.full((300, 160, 3), 215, np.uint8)
    if with_strap:
        cv2.line(img, (20, 60), (120, 270), (40, 40, 40), 14)       # dark diagonal band, shoulder to hip
    return img


def test_strap_evidence_finds_a_clear_strap_and_not_a_blank_shirt():
    score, box = ca.strap_evidence(strap_image(True))
    assert score > 0.4 and box is not None and 0 <= box[0] < box[2] <= 1.0 and 0 <= box[1] < box[3] <= 1.0
    assert ca.strap_evidence(strap_image(False)) == (0.0, None)


def test_strap_evidence_tiny_or_degenerate_crops():
    assert ca.strap_evidence(np.zeros((10, 10, 3), np.uint8)) == (0.0, None)
    assert ca.strap_evidence(np.zeros((0, 0, 3), np.uint8)) == (0.0, None)
    assert ca.strap_evidence(np.zeros((300, 160, 3), np.uint8)) == (0.0, None)


IMG = np.full((200, 400, 3), 100, np.uint8)


def labeler(people, attrs, seats=None, strap=lambda c: (0.0, None), **cfg):
    it = iter(attrs)
    return ca.AutoLabeler(lambda img: people, lambda img, box: next(it), (lambda img: seats) if seats is not None else None,
                          strap, ca.LabelerConfig(**cfg))


def test_labeler_resolves_readable_faces_and_queues_unknown():
    people = [((0.1, 0.1, 0.4, 0.95), 0.9), ((0.55, 0.1, 0.9, 0.95), 0.9)]
    boxes = labeler(people, [Attribute("female", 0.93), Attribute("unknown", 0.0)]).label(IMG)
    persons = [b for b in boxes if b["kind"] == "person"]
    assert [b["cls"] for b in persons] == ["female", None]
    assert persons[1]["queue"] == ca.Q_UNKNOWN_PERSON and persons[0]["conf"] == 0.93
    belts = [b for b in boxes if b["kind"] == "belt"]
    assert len(belts) == 1 and belts[0]["cls"] is None and belts[0]["queue"] == ca.Q_BELT_UNSURE     # only the readable person gets one


def test_labeler_flags_small_unknown_as_possible_child():
    people = [((0.1, 0.05, 0.4, 0.95), 0.9), ((0.5, 0.5, 0.6, 0.8), 0.9)]
    boxes = labeler(people, [Attribute("male", 0.9), Attribute("unknown", 0.0)]).label(IMG)
    unk = [b for b in boxes if b["queue"] == ca.Q_UNKNOWN_PERSON]
    assert len(unk) == 1 and "child" in unk[0]["note"]


def test_labeler_no_people_no_boxes():
    assert labeler([], []).label(IMG) == []
    assert ca.frame_status([]) == ("rejected", "no_person")


def test_labeler_drops_low_confidence_people():
    assert labeler([((0.1, 0.1, 0.5, 0.9), 0.1)], []).label(IMG) == []


def test_labeler_belt_queue_mode_never_labels_a_belt():
    boxes = labeler([((0.1, 0.1, 0.5, 0.95), 0.9)], [Attribute("male", 0.9)], strap=lambda c: (0.9, (0.2, 0.3, 0.6, 0.8))).label(IMG)
    belt = [b for b in boxes if b["kind"] == "belt"][0]
    assert belt["cls"] is None and "strap suggested" in belt["note"]
    assert belt["box"][0] > 0.1                                    # the suggestion, not the torso placeholder


def test_labeler_belt_auto_mode_labels_strong_strap_only_and_never_no_seatbelt():
    strong = labeler([((0.1, 0.1, 0.5, 0.95), 0.9)], [Attribute("male", 0.9)], strap=lambda c: (0.8, (0.2, 0.3, 0.6, 0.8)), belt_mode="auto")
    belt = [b for b in strong.label(IMG) if b["kind"] == "belt"][0]
    assert belt["cls"] == "seatbelt" and belt["source"] == "strap_heuristic" and belt["conf"] == 0.8
    for sc, bx in ((0.0, None), (0.3, (0.2, 0.3, 0.6, 0.8))):
        weak = labeler([((0.1, 0.1, 0.5, 0.95), 0.9)], [Attribute("male", 0.9)], strap=lambda c, s=sc, b=bx: (s, b), belt_mode="auto")
        belt = [b for b in weak.label(IMG) if b["kind"] == "belt"][0]
        assert belt["cls"] is None and belt["queue"] == ca.Q_BELT_UNSURE


def test_labeler_skips_belt_for_tiny_person():
    assert [b["kind"] for b in labeler([((0.1, 0.1, 0.2, 0.2), 0.9)], [Attribute("male", 0.9)]).label(IMG)] == ["person"]


def test_labeler_rejects_bad_belt_mode():
    with pytest.raises(ca.AutolabelError):
        ca.AutoLabeler(lambda i: [], lambda i, b: None, cfg=ca.LabelerConfig(belt_mode="guess"))


def test_labeler_child_seat_candidates_and_child_rule():
    child = [((0.1, 0.1, 0.4, 0.95), 0.9)]
    # a confident seat holding the child: labelled, and no extra question about the child
    out = labeler(child, [Attribute("child", 0.9)], seats=[((0.05, 0.05, 0.5, 1.0), 0.7)]).label(IMG)
    seat = [b for b in out if b["kind"] == "child_seat"]
    assert len(seat) == 1 and seat[0]["cls"] == "child_seat"
    # a weak candidate elsewhere: queued, and the child with no seat near it is queued too
    out = labeler(child, [Attribute("child", 0.9)], seats=[((0.6, 0.1, 0.9, 0.9), 0.2)]).label(IMG)
    seat = [b for b in out if b["kind"] == "child_seat"]
    assert [b["cls"] for b in seat] == [None, None] and {b["source"] for b in seat} == {"yolo_world", "rule"}
    # below the queue confidence: ignored; no detector at all: only the child rule fires
    low = labeler(child, [Attribute("child", 0.9)], seats=[((0.6, 0.1, 0.9, 0.9), 0.05)]).label(IMG)
    assert [b["source"] for b in low if b["kind"] == "child_seat"] == ["rule"]
    none = labeler(child, [Attribute("child", 0.9)]).label(IMG)
    assert [b["source"] for b in none if b["kind"] == "child_seat"] == ["rule"]
    # an adult never triggers the child rule
    assert not [b for b in labeler(child, [Attribute("male", 0.9)]).label(IMG) if b["kind"] == "child_seat"]


# -- the whole run (fake models) ----------------------------------------------------
def fake_labeler(has_person=True, boom=False):
    def people(img):
        if boom:
            raise RuntimeError("model exploded")
        return [((0.1, 0.1, 0.5, 0.95), 0.9)] if has_person else []
    return ca.AutoLabeler(people, lambda img, box: Attribute("male", 0.9), None, lambda c: (0.0, None))


def test_run_pipeline_end_to_end(tmp_path):
    vids = tmp_path / "v"
    vids.mkdir()
    make_video(vids / "Clip One.avi")
    (vids / "broken.avi").write_bytes(b"garbage")
    root = tmp_path / "root"
    run = ca.run_path(root, "r1")
    stats = ca.run_pipeline(ca.find_videos(vids), run, fake_labeler(), licenses={"clip one": {"license": "CC0", "url": "http://x"}},
                            fps=5, dedup=0.0, log=lambda s: None)
    m = ca.load_manifest(run)
    assert set(m["videos"]) == {"Clip_One", "broken"} and m["videos"]["broken"]["error"] and m["videos"]["Clip_One"]["frames"] == 10
    f = m["frames"]["Clip_One_000005"]
    assert f["license"] == "CC0" and f["status"] == "auto" and (run / "images" / "Clip_One_000005.jpg").is_file()
    assert (run / "labels" / "Clip_One_000005.txt").read_text().startswith("0 ")                # male
    assert "names" in (run / "data.yaml").read_text()
    assert stats["frames"] == 10 and stats["frames_in_review_queue"] == 10                      # every belt is a question
    assert all(p.resolve().is_relative_to(run.resolve()) for p in run.rglob("*"))
    with pytest.raises(ca.AutolabelError):                                                      # never silently overwrites a run
        ca.run_pipeline([], run, fake_labeler(), log=lambda s: None)


def test_run_pipeline_zero_detections_rejects_frames(tmp_path):
    make_video(tmp_path / "a.avi")
    run = ca.run_path(tmp_path, "r")
    ca.run_pipeline([tmp_path / "a.avi"], run, fake_labeler(has_person=False), fps=5, dedup=0.0, log=lambda s: None)
    m = ca.load_manifest(run)
    assert {f["status"] for f in m["frames"].values()} == {"rejected"}
    assert {f["reject_reason"] for f in m["frames"].values()} == {"no_person"}
    assert (run / "labels" / "a_000000.txt").read_text() == ""


def test_run_pipeline_model_failure_rejects_that_frame_not_the_run(tmp_path):
    make_video(tmp_path / "a.avi")
    run = ca.run_path(tmp_path, "r")
    ca.run_pipeline([tmp_path / "a.avi"], run, fake_labeler(boom=True), fps=5, dedup=0.0, log=lambda s: None)
    m = ca.load_manifest(run)
    assert len(m["frames"]) == 10 and all(f["reject_reason"].startswith("label_error: RuntimeError") for f in m["frames"].values())


def test_run_pipeline_empty_video_list_and_hostile_names(tmp_path):
    run = ca.run_path(tmp_path, "r")
    assert ca.run_pipeline([], run, fake_labeler(), log=lambda s: None)["frames"] == 0
    evil = tmp_path / "..evil name.avi"
    make_video(evil)
    run2 = ca.run_path(tmp_path, "r2")
    ca.run_pipeline([evil], run2, fake_labeler(), fps=5, dedup=0.0, log=lambda s: None)
    assert {p.name for p in (run2 / "images").iterdir()} <= {f"evil_name_{n:06d}.jpg" for n in range(0, 50, 5)}


def test_run_pipeline_force_clears_only_the_run(tmp_path):
    make_video(tmp_path / "a.avi")
    run = ca.run_path(tmp_path, "r")
    ca.run_pipeline([tmp_path / "a.avi"], run, fake_labeler(), fps=5, dedup=0.0, log=lambda s: None)
    (tmp_path / "autolabel" / "other").mkdir()
    (tmp_path / "autolabel" / "other" / "keep.txt").write_text("x")
    ca.run_pipeline([], run, fake_labeler(), force=True, log=lambda s: None)
    assert ca.load_manifest(run)["frames"] == {} and (tmp_path / "autolabel" / "other" / "keep.txt").exists()


def test_dev_only_footage_is_never_exported_even_after_review(tmp_path):
    make_video(tmp_path / "a.avi")
    run = ca.run_path(tmp_path, "r")
    ca.run_pipeline([tmp_path / "a.avi"], run, fake_labeler(), default_license="DEV-ONLY-stock", fps=5, dedup=0.0, log=lambda s: None)
    m = ca.load_manifest(run)
    for fid in m["frames"]:
        for b in m["frames"][fid]["boxes"]:
            if b["cls"] is None:
                b["cls"] = "seatbelt"
        m["frames"][fid]["status"] = "reviewed"
    ca.save_manifest(run, m)
    r = ca.export_dataset(run, tmp_path / "e", allow_unknown_license=True)
    assert r["frames"] == 0 and r["skipped"] == {"dev_only_license": 10}


def test_unknown_license_is_not_exported_unless_allowed(tmp_path):
    run = make_run(tmp_path, {"v_000001": ("reviewed", [B("male")])})
    m = ca.load_manifest(run)
    m["frames"]["v_000001"]["license"] = "unknown"
    ca.save_manifest(run, m)
    assert ca.export_dataset(run, tmp_path / "e1")["skipped"] == {"unknown_license": 1}
    assert ca.export_dataset(run, tmp_path / "e2", allow_unknown_license=True)["frames"] == 1


# -- models and CLI ---------------------------------------------------------------------
def test_build_seat_finder_failure_returns_none(tmp_path, capsys):
    assert ca.build_seat_finder(tmp_path / "missing-weights.pt") is None
    assert "unavailable" in capsys.readouterr().out


def test_cli_stats_apply_sheets_export(tmp_path, capsys):
    root = tmp_path / "root"
    run = make_run(tmp_path, {"v_000001": ("auto", [B(None, queue="q")]), "v_000002": ("auto", [B("male")])})
    assert run == ca.run_path(root, "r1")
    assert ca.main(["--root", str(root), "stats", "--run", "r1"]) == 0
    assert json.loads(capsys.readouterr().out)["frames_in_review_queue"] == 1
    assert ca.main(["--root", str(root), "sheets", "--run", "r1"]) == 0
    capsys.readouterr()
    c = write_csv(tmp_path / "c.csv", [["v_000001", 0, "child", "", "", "", ""]])
    assert ca.main(["--root", str(root), "apply", "--run", "r1", str(c)]) == 0
    capsys.readouterr()
    assert ca.main(["--root", str(root), "export", "--run", "r1", "--out", str(tmp_path / "ex"), "--reviewed-only"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 1


def test_cli_errors_are_exit_2_not_tracebacks(tmp_path, capsys):
    root = tmp_path / "root"
    assert ca.main(["--root", str(root), "stats", "--run", "../x"]) == 2
    assert ca.main(["--root", str(root), "stats", "--run", "missing"]) == 2
    (root / "autolabel").mkdir(parents=True)
    assert ca.main(["--root", str(root), "run", "--videos", str(tmp_path / "nope"), "--run", "a"]) == 2
    (tmp_path / "empty").mkdir()
    assert ca.main(["--root", str(root), "run", "--videos", str(tmp_path / "empty"), "--run", "a"]) == 2
    assert "error:" in capsys.readouterr().err
