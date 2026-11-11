"""Where the passenger monitor gets its pictures: a live camera or a video file.

``passenger.camera.mode`` in ``config.yaml`` chooses (``camera`` or ``video``); ``--source`` on the command line
overrides it for one run (a number is a camera index, anything else is a video path). In production the cabin has its
own camera; the video mode exists so the whole passenger channel can be tested without one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

Source = Union[int, str]


def resolve_source(cam, override: Optional[str] = None, base: Optional[Path] = None) -> Tuple[Source, bool]:
    """Return ``(source, is_video)``: a camera index, or the path of an existing video file.

    Raises ``FileNotFoundError`` for a video that is not there (with how to fix it) and ``ValueError`` for a bad mode."""
    base = Path.cwd() if base is None else Path(base)

    def video(path_text: str) -> Tuple[Source, bool]:
        path = Path(path_text)
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            raise FileNotFoundError(
                f"passenger video not found: {path}. Set passenger.camera.video_path in config.yaml (or use --source), "
                "or build a demo one with: python scripts/make_demo_video.py")
        return str(path), True

    if override is not None:
        text = str(override)
        return (int(text), False) if text.isdigit() else video(text)
    mode = str(cam.get("mode", "camera")).lower()
    if mode == "camera":
        return int(cam.get("index", 1)), False
    if mode == "video":
        return video(str(cam.get("video_path", "")))
    raise ValueError(f"passenger.camera.mode must be 'camera' or 'video', not {mode!r}")
