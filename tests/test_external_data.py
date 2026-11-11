"""Tests for external datasets: DataManager provenance / raw integrity / stable splits, and the Open Images adapter.

No network: the downloader is replaced by a fake that writes a small synthetic Open Images slice.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger.data_manager import DataManager, _hash_split
from passenger.datasets import open_images as oi

MIDS = {"Man": "/m/man", "Woman": "/m/woman", "Boy": "/m/boy", "Girl": "/m/girl", "Person": "/m/person",
        "Bus": "/m/bus", "Car": "/m/car", "Vehicle": "/m/veh"}
BOX_COLS = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
            "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]


def box(img, name, x0, x1, y0, y1, group=0, depiction=0):
    return [img, "x", MIDS[name], 1, x0, x1, y0, y1, 0, 0, group, depiction, 0]


def synthetic_boxes():
    rows = [
        box("img_bus", "Man", .1, .4, .2, .9), box("img_bus", "Person", .1, .4, .2, .9),
        box("img_general", "Girl", .5, .8, .3, .9), box("img_general", "Person", .5, .8, .3, .9),
        box("img_car", "Woman", .2, .5, .1, .8), box("img_car", "Person", .2, .5, .1, .8),
        box("img_two", "Man", .0, .3, .2, .9), box("img_two", "Boy", .6, .9, .5, .9),
        box("img_two", "Person", .0, .3, .2, .9), box("img_two", "Person", .6, .9, .5, .9),
        # rejected:
        box("img_uncovered", "Man", .1, .3, .2, .9), box("img_uncovered", "Person", .1, .3, .2, .9),
        box("img_uncovered", "Person", .6, .9, .2, .9),                       # a person with no gender label
        box("img_group", "Woman", .1, .9, .1, .9, group=1),
        box("img_drawing", "Man", .1, .5, .1, .9, depiction=1),
        box("img_noattr", "Person", .1, .5, .1, .9),
    ]
    return pd.DataFrame(rows, columns=BOX_COLS)


def synthetic_labels():
    return pd.DataFrame([["img_bus", "h", MIDS["Bus"], 1], ["img_car", "h", MIDS["Car"], 1],
                         ["img_general", "h", MIDS["Vehicle"], 0]],              # Confidence 0: not a real label
                        columns=["ImageID", "Source", "LabelName", "Confidence"])


# -- selection --------------------------------------------------------------------
def test_select_keeps_only_completely_labelled_images_and_orders_by_context():
    got = oi.select_images(synthetic_boxes(), synthetic_labels(), MIDS, max_images=50)
    ids = [s.image_id for s in got]
    assert set(ids) == {"img_bus", "img_car", "img_general", "img_two"}
    assert ids[0] == "img_bus" and ids[1] == "img_car"            # bus first, then other vehicles, then the rest
    assert [s.context for s in got][:2] == ["bus", "vehicle"] and {s.context for s in got[2:]} == {"general"}
    for rejected in ("img_uncovered", "img_group", "img_drawing", "img_noattr"):
        assert rejected not in ids


def test_select_maps_classes_and_converts_boxes_to_yolo():
    got = {s.image_id: s for s in oi.select_images(synthetic_boxes(), synthetic_labels(), MIDS)}
    (cls, cx, cy, w, h), = got["img_bus"].boxes
    assert cls == 0 and (cx, cy, w, h) == pytest.approx((0.25, 0.55, 0.3, 0.7))
    assert got["img_general"].boxes[0][0] == 2                     # Girl -> child
    assert sorted(b[0] for b in got["img_two"].boxes) == [0, 2]    # Man -> male, Boy -> child


def test_select_respects_cap_min_people_and_is_deterministic():
    a = oi.select_images(synthetic_boxes(), synthetic_labels(), MIDS, max_images=2)
    b = oi.select_images(synthetic_boxes().sample(frac=1, random_state=3), synthetic_labels(), MIDS, max_images=2)
    assert [s.image_id for s in a] == [s.image_id for s in b] and len(a) == 2
    assert [s.image_id for s in oi.select_images(synthetic_boxes(), synthetic_labels(), MIDS, min_people=2)] == ["img_two"]


# -- fake downloader + end-to-end ingest -------------------------------------------------
class FakeNet:
    def __init__(self, fail_images=()):
        self.fail_images, self.calls = set(fail_images), []

    def __call__(self, url: str, dest: Path) -> None:
        self.calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        if name.endswith(".jpg"):
            if name[:-4] in self.fail_images:
                raise OSError("network down")
            rng = np.random.default_rng(abs(hash(name)) % 1000)
            cv2.imwrite(str(dest), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
        elif "class-descriptions" in name:
            pd.DataFrame([[m, n] for n, m in MIDS.items()]).to_csv(dest, header=False, index=False)
        elif "annotations-bbox" in name:
            synthetic_boxes().to_csv(dest, index=False)
        elif "imagelabels" in name:
            synthetic_labels().to_csv(dest, index=False)
        elif "images-with-rotation" in name:
            ids = ["img_bus", "img_car", "img_general", "img_two", "img_group"]
            pd.DataFrame({"ImageID": ids, "Subset": "v", "OriginalURL": "u", "OriginalLandingURL": "page",
                          "License": "https://creativecommons.org/licenses/by/2.0/", "Author": "someone",
                          "Title": "t"}).to_csv(dest, index=False, quoting=csv.QUOTE_MINIMAL)
        else:
            raise AssertionError(f"unexpected url {url}")


@pytest.fixture
def dm(tmp_path):
    return DataManager(tmp_path / "pd")


def test_ingest_registers_images_with_provenance_and_keeps_raw(dm):
    ds, n, contexts = oi.ingest(dm, "validation", fetch=FakeNet())
    assert ds == "open_images_v7_validation" and n == 4 and contexts == {"bus": 1, "vehicle": 1, "general": 2}
    raw = dm.external_raw / ds
    assert len(list((raw / "images").glob("*.jpg"))) == 4
    assert (raw / "metadata" / "validation-annotations-bbox.csv").exists()          # original CSVs kept
    src = json.loads((raw / "SOURCE.json").read_text())
    assert src["annotations_license"] == "CC BY 4.0" and "CC BY 2.0" in src["images_license"]
    attribution = pd.read_csv(raw / "metadata" / "attribution.csv")
    assert set(attribution.ImageID) == {"img_bus", "img_car", "img_general", "img_two"}   # img_group was not kept
    assert dm.verify_raw() == []
    assert dm.external[ds]["annotations_license"] == "CC BY 4.0"                    # provenance is in the manifest too


def test_ingest_is_idempotent_and_does_not_redownload(dm):
    net = FakeNet()
    oi.ingest(dm, "validation", fetch=net)
    before = {k: v["split"] for k, v in dm.external["open_images_v7_validation"]["items"].items()}
    first_calls = len(net.calls)
    oi.ingest(dm, "validation", fetch=net)
    assert len(net.calls) == first_calls                           # everything was already in raw/
    assert {k: v["split"] for k, v in dm.external["open_images_v7_validation"]["items"].items()} == before


def test_failed_downloads_are_skipped_not_fatal(dm):
    ds, n, _ = oi.ingest(dm, "validation", fetch=FakeNet(fail_images={"img_car"}))
    assert n == 3 and "img_car" not in dm.external[ds]["items"]


def test_ingest_rejects_unsupported_split(dm):
    with pytest.raises(ValueError):
        oi.ingest(dm, "train", fetch=FakeNet())


# -- raw integrity ------------------------------------------------------------------------
def test_verify_raw_detects_modified_and_missing_files(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    images = sorted((dm.external_raw / ds / "images").glob("*.jpg"))
    images[0].write_bytes(images[0].read_bytes() + b"corrupt")
    images[1].unlink()
    problems = dm.verify_raw()
    assert len(problems) == 2 and any("checksum mismatch" in p for p in problems) and any("missing" in p for p in problems)


def test_verify_raw_covers_videos_too(dm, tmp_path):
    src = tmp_path / "a.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(4):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    vid = dm.register_video(src)
    assert dm.verify_raw() == []
    path = dm.root / dm.videos[vid]["path"]
    path.write_bytes(path.read_bytes() + b"x")
    assert "checksum mismatch" in dm.verify_raw()[0]


def test_raw_survives_extraction_export_and_re_export(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    dm.export_yolo("v1")
    dm.export_yolo("v1")                                           # rebuilt in place
    dm.export_yolo("v2")
    assert dm.verify_raw() == [] and len(list((dm.external_raw / ds / "images").glob("*.jpg"))) == 4


def test_derived_labels_can_be_deleted_and_rebuilt_from_raw(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    labels = dm.external_labels / ds / "labels"
    before = {p.name: p.read_text() for p in labels.glob("*.txt")}
    for p in labels.glob("*.txt"):
        p.unlink()
    oi.ingest(dm, "validation", fetch=FakeNet())                   # raw is reused, labels regenerated
    assert {p.name: p.read_text() for p in labels.glob("*.txt")} == before


# -- unified frames, export, stats, fingerprint ---------------------------------------------------
def test_iter_frames_filters_by_split_and_source(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    allf = list(dm.iter_frames())
    assert len(allf) == 4 and all(f.source == ds and f.label.exists() and f.image.exists() for f in allf)
    for split in ("train", "val", "test"):
        assert all(f.split == split for f in dm.iter_frames(split))
    assert sum(len(list(dm.iter_frames(s))) for s in ("train", "val", "test")) == 4
    assert list(dm.iter_frames(source="videos")) == [] and len(list(dm.iter_frames(source=ds))) == 4


def test_export_merges_sources_and_writes_a_trainable_layout(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    out = dm.export_yolo("mix")
    images = {p.stem for p in (out / "images").rglob("*.jpg")}
    labels = {p.stem for p in (out / "labels").rglob("*.txt")}
    assert images == labels and len(images) == 4 and all(s.startswith(f"{ds}__") for s in images)
    assert "nc: 3" in (out / "data.yaml").read_text()
    only = dm.export_yolo("only_videos", source="videos")                       # no footage registered: nothing to export
    assert not list(only.rglob("*.jpg")) and (only / "data.yaml").exists()


def test_stats_report_external_boxes_per_class(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    ext = dm.stats()["external"][ds]
    assert sum(v["images"] for v in ext.values()) == 4
    totals = {c: sum(v["boxes"].get(c, 0) for v in ext.values()) for c in ("male", "female", "child")}
    assert totals == {"male": 2, "female": 1, "child": 2}


def test_fingerprint_changes_when_a_label_or_image_set_changes(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    fp = {s: dm.fingerprint(s) for s in ("train", "val", "test")}
    assert dm.fingerprint("test") == fp["test"] and len(set(fp.values())) >= 1
    frames = list(dm.iter_frames())
    victim = frames[0]
    victim.label.write_text("0 0.5 0.5 0.1 0.1\n")
    assert dm.fingerprint(victim.split) != fp[victim.split]


def test_validate_labels_covers_external_labels(dm):
    ds, _, _ = oi.ingest(dm, "validation", fetch=FakeNet())
    next(iter((dm.external_labels / ds / "labels").glob("*.txt"))).write_text("9 0.5 0.5 0.1 0.1\n")
    assert any("class id 9" in p for p in dm.validate_labels())
    with pytest.raises(ValueError):
        dm.export_yolo("x")


# -- add_external_dataset guards -------------------------------------------------------------------
def test_add_external_requires_raw_files_and_valid_boxes_and_safe_names(dm):
    with pytest.raises(FileNotFoundError):
        dm.add_external_dataset("nothing_here", {}, [])
    (dm.external_raw / "ds" / "images").mkdir(parents=True)
    cv2.imwrite(str(dm.external_raw / "ds" / "images" / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
    item = {"name": "a", "image": "images/a.jpg", "boxes": [(0, 0.5, 0.5, 0.2, 0.2)]}
    assert dm.add_external_dataset("ds", {"source": "t"}, [item]) == 1
    for bad in ((7, .5, .5, .2, .2), (0, 1.5, .5, .2, .2), (0, .5, .5, 0, .2)):
        with pytest.raises(ValueError):
            dm.add_external_dataset("ds", {}, [{**item, "boxes": [bad]}])
    for name in ("../x", "a/b", ""):
        with pytest.raises(ValueError):
            dm.add_external_dataset("ds", {}, [{**item, "name": name}])
    with pytest.raises(ValueError):
        dm.add_external_dataset("../escape", {}, [])
    with pytest.raises(FileNotFoundError):
        dm.add_external_dataset("ds", {}, [{**item, "image": "images/missing.jpg"}])


# -- stable splits ------------------------------------------------------------------------------------
def test_hash_split_is_stable_and_roughly_proportional():
    keys = [f"ds/img{i}" for i in range(3000)]
    splits = [_hash_split(k, (0.7, 0.15, 0.15)) for k in keys]
    assert splits == [_hash_split(k, (0.7, 0.15, 0.15)) for k in keys]
    share = {s: splits.count(s) / len(keys) for s in ("train", "val", "test")}
    assert 0.66 < share["train"] < 0.74 and 0.12 < share["val"] < 0.18 and 0.12 < share["test"] < 0.18


def test_a_images_split_does_not_depend_on_what_else_is_registered(dm):
    (dm.external_raw / "ds" / "images").mkdir(parents=True)
    for n in range(40):
        cv2.imwrite(str(dm.external_raw / "ds" / "images" / f"i{n}.jpg"), np.full((8, 8, 3), n, np.uint8))
    mk = lambda n: {"name": f"i{n}", "image": f"images/i{n}.jpg", "boxes": [(0, .5, .5, .2, .2)]}   # noqa: E731
    dm.add_external_dataset("ds", {}, [mk(n) for n in range(10)])
    first = {k: v["split"] for k, v in dm.external["ds"]["items"].items()}
    dm.add_external_dataset("ds", {}, [mk(n) for n in range(40)])      # more arrive later
    assert {k: dm.external["ds"]["items"][k]["split"] for k in first} == first
