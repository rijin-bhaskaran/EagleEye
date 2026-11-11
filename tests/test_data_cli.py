"""Tests for the data-manager command line and the CVAT class-order safety check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger.data_manager import DataManager, main


def make_video(path: Path, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(6):
        w.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    w.release()
    return path


def run(tmp_path, *argv):
    return main(["--data", str(tmp_path / "pd"), *argv])


def test_full_annotation_workflow_through_the_cli(tmp_path, capsys):
    vids = []
    for i in range(3):
        assert run(tmp_path, "register", str(make_video(tmp_path / f"v{i}.mp4", i)), "--vehicle", "bus1") == 0
        vids.append(capsys.readouterr().out.split("as ")[1].strip())
    for vid in vids:
        assert run(tmp_path, "extract", vid, "--fps", "5") == 0
    capsys.readouterr()
    assert run(tmp_path, "splits") == 0
    assert set(json.loads(capsys.readouterr().out)) == {"train", "val", "test"}

    dm = DataManager(tmp_path / "pd")
    cvat = tmp_path / "cvat_export"
    (cvat / "obj_train_data").mkdir(parents=True)
    (cvat / "obj.names").write_text("male\nfemale\nchild\n")
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (cvat / "obj_train_data" / f"{jpg.stem}.txt").write_text("1 0.5 0.5 0.3 0.4\n")
    assert run(tmp_path, "import-labels", str(cvat)) == 0 and "imported" in capsys.readouterr().out
    assert run(tmp_path, "validate") == 0 and "labels OK" in capsys.readouterr().out
    assert run(tmp_path, "export", "v1") == 0
    assert (tmp_path / "pd" / "annotations" / "v1" / "data.yaml").exists()
    assert run(tmp_path, "stats") == 0 and "female" in capsys.readouterr().out
    assert run(tmp_path, "verify") == 0 and "intact" in capsys.readouterr().out


def test_cli_validate_and_verify_report_problems_with_a_nonzero_exit(tmp_path, capsys):
    run(tmp_path, "register", str(make_video(tmp_path / "a.mp4")))
    vid = capsys.readouterr().out.split("as ")[1].strip()
    dm = DataManager(tmp_path / "pd")
    (dm.labels_dir / "bad.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    assert run(tmp_path, "validate") == 1 and "class id 9" in capsys.readouterr().out
    path = dm.root / dm.videos[vid]["path"]
    path.write_bytes(path.read_bytes() + b"x")
    assert run(tmp_path, "verify") == 1 and "checksum mismatch" in capsys.readouterr().out


def test_import_refuses_a_cvat_export_with_a_different_class_order(tmp_path):
    dm = DataManager(tmp_path / "pd")
    src = tmp_path / "export"
    src.mkdir()
    (src / "obj.names").write_text("female\nmale\nchild\n")            # male and female swapped
    (src / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="class order"):
        dm.import_labels(src)
    assert not list(dm.labels_dir.glob("*.txt"))                        # nothing was imported


def test_import_accepts_the_right_order_and_exports_without_names_file(tmp_path):
    dm = DataManager(tmp_path / "pd")
    src = tmp_path / "export"
    src.mkdir()
    (src / "obj.names").write_text("male\nfemale\nchild\n")
    assert dm.import_labels(src) == 0                                    # right order, nothing matching: fine
    (src / "obj.names").unlink()
    assert dm.import_labels(src) == 0                                    # no names file: nothing to compare


def test_register_cli_records_group_and_rights(tmp_path, capsys):
    assert run(tmp_path, "register", str(make_video(tmp_path / "a.mp4")), "--vehicle", "car",
               "--group", "g_pair", "--rights", "unverified, internal only") == 0
    vid = capsys.readouterr().out.split("as ")[1].strip()
    meta = DataManager(tmp_path / "pd").videos[vid]
    assert meta["group"] == "g_pair" and meta["rights"] == "unverified, internal only"
