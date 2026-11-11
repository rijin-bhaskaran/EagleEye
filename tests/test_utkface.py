import io
import tarfile

import pytest

from passenger.datasets import utkface


@pytest.mark.parametrize("name,expected", [
    ("part1/2_1_2_2016.jpg", (0, 1)), ("part1/3_0_0_x.jpg", (1, 0)), ("part1/9_0_0_x.jpg", (1, 0)),
    ("part1/10_1_0_x.jpg", (2, 1)), ("part1/29_0_0_x.jpg", (3, 0)), ("part1/70_1_0_x.jpg", (8, 1)),
    ("part1/116_0_0_x.jpg", (8, 0)),
])
def test_parse_name_maps_age_to_fairface_groups(name, expected):
    assert utkface.parse_name(name) == expected


@pytest.mark.parametrize("name", ["part1/61_3_20170109150557335.jpg.chip.jpg", "part1/x_1_0_a.jpg", "part1/5_1.jpg",
                                  "part1/5_2_0_a.jpg", "part1/500_1_0_a.jpg", "part1/5_1_0_a.png"])
def test_parse_name_rejects_bad_names(name):
    assert utkface.parse_name(name) is None


def test_load_rows_reads_tar_and_skips_bad_files(tmp_path):
    p = tmp_path / "part9.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for n, data in (("part9/30_0_0_a.jpg", b"AAA"), ("part9/bad.jpg", b"B"), ("part9/4_1_1_b.jpg", b"CC")):
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    assert utkface.load_rows([p]) == [(b"AAA", 4, 0), (b"CC", 1, 1)]


@pytest.mark.parametrize("age,gender,expected", [
    (0, 0, "child"), (12, 1, "child"), (13, 0, None), (17, 1, None),
    (18, 0, "male"), (18, 1, "female"), (80, 1, "female"),
])
def test_face_class_uses_the_project_age_cut_offs(age, gender, expected):
    assert utkface.face_class(age, gender) == expected


def test_parse_age_gender_keeps_the_exact_age():
    assert utkface.parse_age_gender("part1/12_1_2_2016.jpg") == (12, 1)
    assert utkface.parse_age_gender("part1/61_3_20170109150557335.jpg.chip.jpg") is None
