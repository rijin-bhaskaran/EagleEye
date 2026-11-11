"""Tests for scripts/fetch_passenger_models.py using a local file:// source (no network)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_models", Path(__file__).resolve().parents[1] / "scripts" / "fetch_passenger_models.py")
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"alpha" * 1000)
    (src / "b.bin").write_bytes(b"beta" * 1000)
    files = {n: fm.sha256(src / n) for n in ("a.bin", "b.bin")}
    return src.as_uri(), files


def test_fetch_downloads_and_verifies(tmp_path, source):
    base, files = source
    fm.fetch(tmp_path / "dest", files, base)
    assert all(fm.is_valid(tmp_path / "dest" / n, h) for n, h in files.items())
    assert not list((tmp_path / "dest").glob("*.part"))


def test_fetch_repairs_truncated_file_and_skips_good_ones(tmp_path, source, capsys):
    base, files = source
    dest = tmp_path / "dest"
    fm.fetch(dest, files, base)
    (dest / "a.bin").write_bytes(b"alpha" * 10)                 # simulate a cut-off download
    capsys.readouterr()
    fm.fetch(dest, files, base)
    out = capsys.readouterr().out
    assert "fetching a.bin" in out and "ok       b.bin" in out
    assert fm.is_valid(dest / "a.bin", files["a.bin"])


def test_fetch_gives_up_on_bad_checksum(tmp_path, source):
    base, files = source
    with pytest.raises(SystemExit):
        fm.fetch(tmp_path / "dest", {"a.bin": "0" * 64}, base, retries=2)
    assert not (tmp_path / "dest" / "a.bin").exists() and not list((tmp_path / "dest").glob("*.part"))


def test_real_weights_match_pinned_checksums():
    d = Path("models/passenger")
    if not d.exists():
        pytest.skip("weights not downloaded")
    assert all(fm.is_valid(d / n, h) for n, h in fm.FILES.items())
