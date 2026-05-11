"""Repository layout helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory that contains `deepSORT_rtdetr.py` and `inference_parameter/`."""
    return Path(__file__).resolve().parent.parent


def inference_dir() -> Path:
    return repo_root() / "inference_parameter"


def default_ground_truth() -> Path:
    """First existing candidate under `inference_parameter/camera_1/`."""
    cam = inference_dir() / "camera_1"
    for name in ("ground_truth.csv", "gt.csv"):
        p = cam / name
        if p.is_file():
            return p
    return cam / "ground_truth.csv"
