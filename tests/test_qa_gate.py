"""Tests for scripts/qa_gate.py: the gate must fail when it should and never report a skip as a pass."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.data_manager import DataManager

_spec = importlib.util.spec_from_file_location("qa_gate", ROOT / "scripts" / "qa_gate.py")
qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa)

GOOD_COST = {"pipeline": 1.0, "overlay": 1.0, "jpeg": 1.0, "payload": 0.1, "total": 3.1}


@pytest.mark.parametrize("text,expected", [
    ("160 passed in 41.54s", (160, 0)),
    ("1 failed, 157 passed in 39.66s", (157, 1)),
    ("2 failed, 3 passed, 1 error in 5s", (3, 3)),
    ("3 passed, 4 skipped in 1s", (3, 0)),
    ("", (0, 0)),
])
def test_parse_pytest_summary(text, expected):
    assert qa.parse_pytest_summary("...\n" + text + "\n") == expected


def patch_gate(monkeypatch, results, ok, cost=GOOD_COST):
    monkeypatch.setattr(qa, "run_tests", lambda runs: (results, ok))
    import scripts.bench_passenger_overhead as bench
    monkeypatch.setattr(bench, "measure", lambda: cost)


def test_gate_passes_and_says_evaluation_was_not_run(monkeypatch, capsys):
    patch_gate(monkeypatch, [(160, 0, 40.0), (160, 0, 41.0)], True)
    assert qa.main([]) == 0
    out = capsys.readouterr().out
    assert "160 passed / 0 failed" in out and "[OK]" in out
    assert "evaluation: NOT RUN" in out and "QA gate history" in out


def test_gate_fails_on_a_test_failure(monkeypatch, capsys):
    patch_gate(monkeypatch, [(159, 1, 40.0), (160, 0, 40.0)], False)
    assert qa.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_gate_fails_when_the_overhead_budget_is_blown(monkeypatch, capsys):
    patch_gate(monkeypatch, [(160, 0, 40.0)], True, cost={**GOOD_COST, "total": 40.0})
    assert qa.main(["--runs", "1"]) == 1
    assert "[FAIL]" in capsys.readouterr().out
    patch_gate(monkeypatch, [(160, 0, 40.0)], True, cost={**GOOD_COST, "total": 40.0})
    assert qa.main(["--runs", "1", "--overhead-budget-ms", "50"]) == 0


def test_run_tests_requires_repeat_runs_to_agree(monkeypatch):
    outputs = iter([(160, 0), (159, 0)])

    class Proc:
        returncode, stderr = 0, ""

        def __init__(self):
            p, f = next(outputs)
            self.stdout = f"{p} passed in 1s"

    monkeypatch.setattr(qa.subprocess, "run", lambda *a, **k: Proc())
    results, ok = qa.run_tests(2)
    assert len(results) == 2 and not ok                      # same failures-free but different counts: suspicious


def test_run_tests_treats_a_crash_without_summary_as_failure(monkeypatch):
    class Proc:
        returncode, stdout, stderr = 2, "collection exploded", ""

    monkeypatch.setattr(qa.subprocess, "run", lambda *a, **k: Proc())
    results, ok = qa.run_tests(1)
    assert not ok and results[0][1] == 1


def test_labelled_test_frames_counts_only_labelled_test_split(tmp_path):
    assert qa.labelled_test_frames(tmp_path / "missing") == 0
    dm = DataManager(tmp_path / "pd")
    for i in range(3):
        src = tmp_path / f"v{i}.mp4"
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        rng = np.random.default_rng(i)
        for _ in range(4):
            w.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
        w.release()
        dm.extract_frames(dm.register_video(src), target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    assert qa.labelled_test_frames(dm.root) == 0             # frames exist but none are labelled
    for vid, meta in dm.videos.items():
        for idx in meta["frames"]:
            (dm.labels_dir / f"{vid}_{idx:06d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    expected = sum(len(m["frames"]) for m in dm.videos.values() if m["split"] == "test")
    assert expected > 0 and qa.labelled_test_frames(dm.root) == expected


def test_gate_reports_skip_when_data_has_no_labelled_test_frames(monkeypatch, capsys, tmp_path):
    patch_gate(monkeypatch, [(160, 0, 40.0)], True)
    DataManager(tmp_path / "pd")
    assert qa.main(["--runs", "1", "--data", str(tmp_path / "pd")]) == 0
    assert "not a pass" in capsys.readouterr().out


def test_overhead_benchmark_returns_all_steps():
    from scripts.bench_passenger_overhead import measure
    r = measure(n=3, people=10)
    assert set(r) == {"pipeline", "overlay", "jpeg", "payload", "total"} and r["total"] == pytest.approx(
        r["pipeline"] + r["overlay"] + r["jpeg"] + r["payload"])


def test_the_evaluation_always_uses_the_bus_pipeline_whatever_vehicle_is_toggled(monkeypatch, tmp_path):
    """The gate's benchmark compares rows over time; toggling config.yaml to car must not change what it measures."""
    import eagleeye.config as ec

    seen = {}

    def spy(path=None, vehicle_type=None):
        seen["vehicle_type"] = vehicle_type
        raise RuntimeError("stop after the config is loaded")

    monkeypatch.setattr(ec, "load_config", spy)
    with pytest.raises(RuntimeError, match="stop after"):
        qa.run_evaluation(tmp_path, with_classifier=False)
    assert seen["vehicle_type"] == "bus"


def test_a_failing_run_names_the_failing_tests(monkeypatch, capsys):
    out = "..F\nFAILED tests/test_x.py::test_a - boom\nERROR tests/test_y.py::test_b\n1 failed, 2 passed, 1 error in 3.0s\n"
    monkeypatch.setattr(qa.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out, stderr="", returncode=1))
    results, ok = qa.run_tests(1)
    text = capsys.readouterr().out
    assert not ok and results[0][:2] == (2, 2)
    assert "tests/test_x.py::test_a" in text and "tests/test_y.py::test_b" in text
