"""
Generate randomised trial YAML configs from a base template.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import yaml

from rtdetr_eval.paths import default_trials_dir

SEARCH_SPACE = [
    {"path": "inference.confidence", "type": "float", "low": 0.40, "high": 0.85, "dp": 2},
    {"path": "inference.iou", "type": "float", "low": 0.50, "high": 0.85, "dp": 2},
    {"path": "inference.downscale_width", "type": "choice", "choices": [640, 800, 960, 1280]},
    {"path": "passes.top_region.ratio", "type": "float", "low": 0.40, "high": 0.70, "dp": 2},
    {"path": "nms.hard_iou", "type": "float", "low": 0.15, "high": 0.40, "dp": 2},
    {"path": "nms.containment_fraction", "type": "float", "low": 0.35, "high": 0.65, "dp": 2},
    {"path": "nms.soft_nms_iou", "type": "float", "low": 0.15, "high": 0.35, "dp": 2},
    {"path": "nms.soft_nms_sigma", "type": "float", "low": 0.10, "high": 0.40, "dp": 2},
    {"path": "classes.0.min_confidence", "type": "float", "low": 0.45, "high": 0.80, "dp": 2},
    {"path": "classes.1.min_confidence", "type": "float", "low": 0.45, "high": 0.75, "dp": 2},
    {"path": "classes.0.tracker.max_age", "type": "int", "low": 5, "high": 25},
    {"path": "classes.0.tracker.n_init", "type": "int", "low": 1, "high": 5},
    {"path": "classes.0.tracker.max_iou_distance", "type": "float", "low": 0.60, "high": 0.95, "dp": 2},
    {"path": "classes.0.tracker.max_cosine_distance", "type": "float", "low": 0.40, "high": 0.80, "dp": 2},
    {"path": "classes.1.tracker.max_age", "type": "int", "low": 8, "high": 40},
    {"path": "classes.1.tracker.n_init", "type": "int", "low": 2, "high": 6},
    {"path": "classes.1.tracker.max_iou_distance", "type": "float", "low": 0.30, "high": 0.65, "dp": 2},
    {"path": "classes.1.tracker.max_cosine_distance", "type": "float", "low": 0.20, "high": 1.00, "dp": 2},
]

RESOLUTION_PAIRS = {640: 360, 800: 450, 960: 540, 1280: 720}


def sample_params() -> dict:
    sample = {}
    for param in SEARCH_SPACE:
        p = param["path"]
        if param["type"] == "float":
            sample[p] = round(random.uniform(param["low"], param["high"]), param.get("dp", 3))
        elif param["type"] == "int":
            sample[p] = random.randint(param["low"], param["high"])
        elif param["type"] == "choice":
            sample[p] = random.choice(param["choices"])

    floor = sample.get("inference.confidence", 0.60)
    for key in ("classes.0.min_confidence", "classes.1.min_confidence"):
        if key in sample and sample[key] < floor:
            sample[key] = round(floor, 2)

    if "inference.downscale_width" in sample:
        sample["inference.downscale_height"] = RESOLUTION_PAIRS[sample["inference.downscale_width"]]

    return sample


def set_nested(d: dict, dotpath: str, value):
    keys = dotpath.split(".")
    obj = d
    for key in keys[:-1]:
        obj = obj[int(key)] if key.isdigit() else obj[key]
    last = keys[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        obj[last] = value


def apply_sample(base_cfg: dict, sample: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for path, value in sample.items():
        try:
            set_nested(cfg, path, value)
        except (KeyError, IndexError, TypeError) as exc:
            print(f"  [warn] Cannot set {path}: {exc}")
    return cfg


def generate_trials(base_config: Path, out_dir: Path, n: int, seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)

    existing = sorted(out_dir.glob("trial_*.yaml"))
    start_n = len(existing) + 1
    if existing:
        print(f"Found {len(existing)} existing trials — starting from trial_{start_n:03d}")

    manifest_path = out_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    print(f"Generating {n} trial configs → {out_dir}/")
    print(f"{'Trial':<12} {'Key params'}")
    print("-" * 60)

    for i in range(start_n, start_n + n):
        trial_id = f"trial_{i:03d}"
        sample = sample_params()
        cfg = apply_sample(base_cfg, sample)

        cfg["input"] = ""
        cfg["output"] = ""

        out_path = out_dir / f"{trial_id}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        manifest[trial_id] = sample

        conf = sample.get("inference.confidence", "?")
        p_min = sample.get("classes.1.min_confidence", "?")
        p_age = sample.get("classes.1.tracker.max_age", "?")
        res = sample.get("inference.downscale_width", "?")
        print(f"  {trial_id}   conf={conf}  ped_min={p_min}  ped_age={p_age}  res={res}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = start_n + n - 1
    print(f"\nDone. {n} configs written ({start_n:03d}–{total:03d}).")
    print(f"Manifest: {manifest_path}")
    print("\nNext step:")
    print(
        f"  python -m rtdetr_eval run-trials \\\n"
        f"      --trials-dir {out_dir} \\\n"
        f"      --video <path/to/video.mp4>"
    )


def main():
    p = argparse.ArgumentParser(description="Generate randomised trial configs.")
    p.add_argument("--config", required=True, help="Base YAML template")
    p.add_argument("--n", type=int, default=50, help="Number of trials (default: 50)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {default_trials_dir()})",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()
    out = args.out_dir if args.out_dir is not None else default_trials_dir()
    generate_trials(Path(args.config).resolve(), out.resolve(), args.n, args.seed)


if __name__ == "__main__":
    main()
