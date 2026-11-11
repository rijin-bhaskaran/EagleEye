#!/usr/bin/env python3
"""The QA gate: run this after every code change, then paste the summary into WORKFLOW.md.

    python scripts/qa_gate.py                       # tests (2 runs) + overhead benchmark
    python scripts/qa_gate.py --data passenger_data # ...and the model evaluation once labelled test frames exist
    python scripts/qa_gate.py --runs 5              # more repeats after touching sockets, threads or timing

Exit code is non-zero if any test fails, the repeated runs disagree, or the overhead budget is blown.
The evaluation step is skipped (and says so) while there is no labelled test data, so a skipped
evaluation is never mistaken for a passing one.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: A tripwire, not a target: about 3x what the code costs on the dev laptop, so a real regression trips it.
DEFAULT_OVERHEAD_BUDGET_MS = 15.0


def parse_pytest_summary(output: str) -> Tuple[int, int]:
    """(passed, failed) from pytest's last line; failed counts errors too."""
    last = [line for line in output.strip().splitlines() if " passed" in line or " failed" in line or " error" in line]
    text = last[-1] if last else ""
    count = lambda word: int(m.group(1)) if (m := re.search(rf"(\d+) {word}", text)) else 0   # noqa: E731
    return count("passed"), count("failed") + count("error")


def run_tests(runs: int) -> Tuple[List[Tuple[int, int, float]], bool]:
    results = []
    for _ in range(runs):
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=ROOT, capture_output=True, text=True)
        passed, failed = parse_pytest_summary(proc.stdout)
        if proc.returncode != 0 and failed == 0:      # collection error, crash: pytest never reached a summary
            failed = 1
            print(proc.stdout[-2000:], proc.stderr[-2000:])
        names = [l.split(" - ")[0] for l in proc.stdout.splitlines() if l.startswith(("FAILED ", "ERROR "))]
        if names:                                     # a failure is a bug until its cause is found: say which test it was
            print("[gate] failing tests this run:", *names, sep="\n  ", flush=True)
        results.append((passed, failed, time.time() - t0))
    ok = all(f == 0 for _, f, _ in results) and len({p for p, _, _ in results}) == 1
    return results, ok


def labelled_test_frames(data_dir: Path) -> int:
    """How many test-split frames have a label file. Zero means there is nothing to evaluate yet."""
    from passenger.data_manager import DataManager

    if not data_dir.exists():
        return 0
    return sum(1 for _ in DataManager(data_dir).iter_frames("test"))


def run_evaluation(data_dir: Path, with_classifier: bool) -> dict:
    from eagleeye.config import load_config
    from passenger.classifier import PassengerClassifier
    from passenger.data_manager import DataManager
    from passenger.detector import PersonDetector
    from passenger.evaluate import evaluate
    from passenger.seats import SeatMap

    cfg = load_config(ROOT / "config.yaml", vehicle_type="bus")        # rows of the history are only comparable if this never follows the toggle
    det = cfg.passenger.detector
    seatmap = SeatMap.from_config(cfg.passenger.seats, cfg.passenger.occupancy.min_overlap)
    clf = PassengerClassifier.from_config(cfg) if with_classifier else None
    return evaluate(DataManager(data_dir), PersonDetector(det.model, det.conf, det.imgsz, det.classes), seatmap, "test", classifier=clf)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EagleEye QA gate")
    ap.add_argument("--runs", type=int, default=2, help="full test-suite repeats (default 2)")
    ap.add_argument("--data", default=None, help="passenger_data directory, to also run the model evaluation")
    ap.add_argument("--no-classifier", action="store_true", help="evaluate detection/count/seats only")
    ap.add_argument("--overhead-budget-ms", type=float, default=DEFAULT_OVERHEAD_BUDGET_MS)
    args = ap.parse_args(argv)

    lines = [f"QA gate {time.strftime('%Y-%m-%d %H:%M')}"]
    failed = False

    results, tests_ok = run_tests(args.runs)
    failed |= not tests_ok
    lines.append("- tests: " + ", ".join(f"{p} passed / {f} failed ({s:.0f}s)" for p, f, s in results)
                 + ("  [OK]" if tests_ok else "  [FAIL: failures, or runs disagree]"))

    from scripts.bench_passenger_overhead import measure
    cost = measure()
    over = cost["total"] > args.overhead_budget_ms
    failed |= over
    lines.append(f"- overhead per frame (excl. model): {cost['total']:.1f} ms "
                 f"[pipeline {cost['pipeline']:.1f}, overlay {cost['overlay']:.1f}, jpeg {cost['jpeg']:.1f}, payload {cost['payload']:.2f}] "
                 f"budget {args.overhead_budget_ms:g} ms " + ("[FAIL]" if over else "[OK]"))

    n_labelled = labelled_test_frames(Path(args.data)) if args.data else 0
    if not args.data:
        lines.append("- evaluation: NOT RUN (pass --data passenger_data)")
    elif n_labelled == 0:
        lines.append("- evaluation: NOT RUN (no labelled test frames yet; this is a skip, not a pass)")
    else:
        rep = run_evaluation(Path(args.data), not args.no_classifier)
        d, c, o = rep["detection"], rep["count"], rep["occupancy"]
        seats = f"seats F1 {o['f1']:.3f} macro {o['macro_f1']:.3f}" if o else "seats n/a (no cabin footage)"
        lines.append(f"- evaluation on {rep['frames']} test frames {rep['sources']}: det P {d['precision']:.3f} R {d['recall']:.3f} "
                     f"AP50 {d['ap50']:.3f} mAP50-95 {d['map50_95']:.3f} | count MAE {c['mae']:.2f} exact {c['exact']:.2f} | "
                     f"{seats} | {rep['speed']['mean_ms']:.0f} ms/frame")
        if "classification" in rep:
            k = rep["classification"]
            lines.append(f"  classification: macro-F1 {k['macro_f1']:.3f}, coverage {k['coverage']:.2f}, "
                         f"accuracy answered {k['accuracy_answered']:.3f}")

    print("\n".join(lines))
    print("\nPaste the block above into the WORKFLOW.md 'QA gate history' table.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
