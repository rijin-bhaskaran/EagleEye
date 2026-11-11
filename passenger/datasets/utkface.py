"""UTKFace: aligned face crops whose file name is ``<age>_<gender>_<race>_<timestamp>.jpg`` (gender 0 = male, 1 = female).

Read straight from the downloaded ``part*.tar.gz`` files (never extracted, never modified) and returned in the
same ``(jpeg bytes, age-group index, gender index)`` form as FairFace, so the trainer treats both alike.
Exact ages are mapped to the FairFace age groups (0-2, 3-9, 10-19, 20-29, ...).
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Row = Tuple[bytes, int, int]
_EDGES = (3, 10, 20, 30, 40, 50, 60, 70)          # upper bounds of groups 0..7; 70+ is group 8
CHILD_MAX_AGE, ADULT_MIN_AGE = 12, 18


def age_group(age: int) -> int:
    return sum(age >= e for e in _EDGES)


def parse_age_gender(name: str) -> Optional[Tuple[int, int]]:
    """(exact age, gender) from a UTKFace file name, or None if it is not a valid one (some files are misnamed)."""
    parts = Path(name).name.split("_")
    if len(parts) < 4 or not name.lower().endswith(".jpg"):
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not 0 <= age <= 120 or gender not in (0, 1):
        return None
    return age, gender


def parse_name(name: str) -> Optional[Tuple[int, int]]:
    """(FairFace age group, gender) from a UTKFace file name, or None if it is not a valid one."""
    parsed = parse_age_gender(name)
    return None if parsed is None else (age_group(parsed[0]), parsed[1])


def face_class(age: int, gender: int) -> Optional[str]:
    """Project class of a face: child (age <= 12), male / female (age >= 18); 13-17 is ambiguous, so None (docs/ANNOTATION.md)."""
    if age <= CHILD_MAX_AGE:
        return "child"
    return ("male", "female")[gender] if age >= ADULT_MIN_AGE else None


def load_rows(tars: Sequence[Path]) -> List[Row]:
    rows: List[Row] = []
    for path in tars:
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                label = parse_name(m.name)
                if label is None:
                    continue
                data = tf.extractfile(m).read()
                rows.append((data, label[0], label[1]))
    return rows
