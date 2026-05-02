"""
generate_configs.py — Stage 1: Generate randomised trial YAML configs.

Reads your base_config.yaml (with fixed [CAMERA] params already set),
samples N random hyperparameter combinations, and writes one YAML per trial.

Usage:
    python generate_configs.py \\
        --config base_config.yaml \\
        --n 50 \\
        --out-dir trials/cam_01 \\
        --seed 42

Outputs:
    trials/cam_01/trial_001.yaml
    trials/cam_01/trial_002.yaml
    ...
    trials/cam_01/trial_050.yaml
    trials/cam_01/manifest.json   ← maps trial ID → params (for scoring later)
"""

import argparse
import copy
import json
import random
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Search space
# [CAMERA] fields (warp.src_points, model, half) are intentionally excluded.
# ---------------------------------------------------------------------------
SEARCH_SPACE = [
    # ── Global confidence floor ───────────────────────────────────────────
    {"path": "inference.confidence",        "type": "float",  "low": 0.40, "high": 0.85, "dp": 2},

    # ── Inference NMS ─────────────────────────────────────────────────────
    {"path": "inference.iou",               "type": "float",  "low": 0.50, "high": 0.85, "dp": 2},

    # ── Resolution (height auto-paired) ───────────────────────────────────
    {"path": "inference.downscale_width",   "type": "choice", "choices": [640, 800, 960, 1280]},

    # ── Top-region pass ───────────────────────────────────────────────────
    {"path": "passes.top_region.ratio",     "type": "float",  "low": 0.40, "high": 0.70, "dp": 2},

    # ── Pre-tracker NMS ───────────────────────────────────────────────────
    {"path": "nms.hard_iou",                "type": "float",  "low": 0.15, "high": 0.40, "dp": 2},
    {"path": "nms.containment_fraction",    "type": "float",  "low": 0.35, "high": 0.65, "dp": 2},
    {"path": "nms.soft_nms_iou",            "type": "float",  "low": 0.15, "high": 0.35, "dp": 2},
    {"path": "nms.soft_nms_sigma",          "type": "float",  "low": 0.10, "high": 0.40, "dp": 2},

    # ── Per-class confidence ──────────────────────────────────────────────
    {"path": "classes.0.min_confidence",    "type": "float",  "low": 0.45, "high": 0.80, "dp": 2},  # Cyclist
    {"path": "classes.1.min_confidence",    "type": "float",  "low": 0.45, "high": 0.75, "dp": 2},  # Pedestrian

    # ── Cyclist tracker ───────────────────────────────────────────────────
    {"path": "classes.0.tracker.max_age",              "type": "int",   "low": 5,    "high": 25},
    {"path": "classes.0.tracker.n_init",               "type": "int",   "low": 1,    "high": 5},
    {"path": "classes.0.tracker.max_iou_distance",     "type": "float", "low": 0.60, "high": 0.95, "dp": 2},
    {"path": "classes.0.tracker.max_cosine_distance",  "type": "float", "low": 0.40, "high": 0.80, "dp": 2},

    # ── Pedestrian tracker ────────────────────────────────────────────────
    {"path": "classes.1.tracker.max_age",              "type": "int",   "low": 8,    "high": 40},
    {"path": "classes.1.tracker.n_init",               "type": "int",   "low": 2,    "high": 6},
    {"path": "classes.1.tracker.max_iou_distance",     "type": "float", "low": 0.30, "high": 0.65, "dp": 2},
    {"path": "classes.1.tracker.max_cosine_distance",  "type": "float", "low": 0.20, "high": 1.00, "dp": 2},
]

RESOLUTION_PAIRS = {640: 360, 800: 450, 960: 540, 1280: 720}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    # Constraint: inference.confidence <= per-class min_confidence
    floor = sample.get("inference.confidence", 0.60)
    for key in ("classes.0.min_confidence", "classes.1.min_confidence"):
        if key in sample and sample[key] < floor:
            sample[key] = round(floor, 2)

    # Constraint: resolution height paired with width
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate randomised trial configs.")
    p.add_argument("--config",   required=True, help="Base YAML template")
    p.add_argument("--n",        type=int, default=50, help="Number of trials (default: 50)")
    p.add_argument("--out-dir",  type=Path, default=Path("trials"), help="Output directory")
    p.add_argument("--seed",     type=int, default=None, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Check for existing trials to avoid overwriting
    existing = sorted(args.out_dir.glob("trial_*.yaml"))
    start_n  = len(existing) + 1
    if existing:
        print(f"Found {len(existing)} existing trials — starting from trial_{start_n:03d}")

    manifest_path = args.out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    print(f"Generating {args.n} trial configs → {args.out_dir}/")
    print(f"{'Trial':<12} {'Key params'}")
    print("-" * 60)

    for i in range(start_n, start_n + args.n):
        trial_id = f"trial_{i:03d}"
        sample   = sample_params()
        cfg      = apply_sample(base_cfg, sample)

        # Clear I/O fields — run_trials.py injects these at runtime
        cfg["input"]  = ""
        cfg["output"] = ""

        out_path = args.out_dir / f"{trial_id}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        manifest[trial_id] = sample

        # Print a one-line summary of interesting params
        conf  = sample.get("inference.confidence", "?")
        p_min = sample.get("classes.1.min_confidence", "?")
        p_age = sample.get("classes.1.tracker.max_age", "?")
        res   = sample.get("inference.downscale_width", "?")
        print(f"  {trial_id}   conf={conf}  ped_min={p_min}  ped_age={p_age}  res={res}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = start_n + args.n - 1
    print(f"\nDone. {args.n} configs written ({start_n:03d}–{total:03d}).")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step:")
    print(f"  python run_trials.py \\")
    print(f"      --trials-dir {args.out_dir} \\")
    print(f"      --videos ../trim1.mp4 ../trim2.mp4 ../trim3.mp4 \\")
    print(f"      --out-dir results/cam_01")


if __name__ == "__main__":
    main()

# python3 generate_configs.py --config base.yaml --n 75 --out-dir trials/cam_01 --seed 42