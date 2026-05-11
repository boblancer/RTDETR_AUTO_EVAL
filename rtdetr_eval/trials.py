"""
Run trial YAMLs on one video, score vs ground truth, write best_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import subprocess
from pathlib import Path

import yaml

from rtdetr_eval.evaluate import run_evaluation
from rtdetr_eval.paths import default_ground_truth, inference_dir, repo_root


def predictions_path(video: Path) -> Path:
    return video.parent / f"{video.stem}_rtdetr.csv"


def trial_score(metrics: dict) -> float:
    ped_ap = metrics["classes"]["Pedestrian"]["ap"]
    cyc_ap = metrics["classes"]["Cyclist"]["ap"]
    aps = [ped_ap] + ([cyc_ap] if cyc_ap > 0 else [])
    ap = sum(aps) / len(aps)
    iou = metrics["iou_stats"]["mean"]
    return ap if iou >= 0.65 else ap * 0.5


def run_inference(
    infer_script: Path,
    config: Path,
    video: Path,
    *,
    cwd: Path,
) -> None:
    cmd = [
        sys.executable,
        str(infer_script),
        "--config",
        str(config.resolve()),
        "-i",
        str(video.resolve()),
        "--csv",
    ]
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_trials(
    trials_dir: Path,
    video: Path,
    gt_csv: Path,
    *,
    infer_script: Path | None = None,
    inference_cwd: Path | None = None,
    iou_thresh: float = 0.5,
    best_out: Path | None = None,
    eval_work: Path | None = None,
    eval_plots: bool = False,
) -> Path:
    """
    Returns path to written best_config.yaml.
    """
    root = repo_root()
    infer_script = infer_script or (root / "deepSORT_rtdetr.py")
    inference_cwd = inference_cwd or inference_dir()
    best_out = best_out or (trials_dir / "best_config.yaml")
    eval_work = eval_work or (trials_dir / "_eval_scratch")
    eval_work.mkdir(parents=True, exist_ok=True)

    trials = sorted(trials_dir.glob("trial_*.yaml"))
    if not trials:
        raise FileNotFoundError(f"No trial_*.yaml under {trials_dir}")

    video = video.resolve()
    gt_csv = gt_csv.resolve()

    results: list[tuple[float, str, dict]] = []
    pred_csv = predictions_path(video)

    print("Trials:", len(trials))
    for trial in trials:
        trial_eval_dir = eval_work / trial.stem
        if trial_eval_dir.exists():
            shutil.rmtree(trial_eval_dir)
        trial_eval_dir.mkdir(parents=True)

        print(f"\n--- {trial.name} ---")
        run_inference(infer_script, trial, video, cwd=inference_cwd)

        if not pred_csv.is_file():
            raise FileNotFoundError(
                f"Expected predictions CSV at {pred_csv} after inference "
                "(deepSORT_rtdetr.py writes next to the input video)."
            )

        metrics = run_evaluation(
            gt_csv,
            pred_csv,
            trial_eval_dir,
            iou_thresh,
            plots=eval_plots,
            write_json=False,
        )

        s = trial_score(metrics)
        print(
            f"  score={s:.4f}  ap={metrics['classes']['Pedestrian']['ap']:.4f}  "
            f"recall={metrics['classes']['Pedestrian']['recall']:.4f}"
        )
        results.append((s, trial.stem, metrics))

    results.sort(key=lambda x: x[0], reverse=True)
    print("\n--- Top 5 ---")
    for s, name, _ in results[:5]:
        print(f"  {name}  score={s:.4f}")

    best_name = results[0][1]
    best_yaml = trials_dir / f"{best_name}.yaml"
    cfg = yaml.safe_load(best_yaml.read_text())
    cfg["input"] = ""
    with open(best_out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved to {best_out} ({best_name})")

    # Optional: persist leaderboard next to best config
    leaderboard = [
        {"trial": name, "score": round(s, 6), "metrics": m} for s, name, m in results
    ]
    lb_path = best_out.parent / "trial_leaderboard.json"
    lb_path.write_text(json.dumps(leaderboard, indent=2))
    print(f"Leaderboard: {lb_path}")

    return best_out


def main():
    p = argparse.ArgumentParser(description="Run trial configs on a video and pick the best.")
    p.add_argument("--trials-dir", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth CSV (default: inference_parameter/camera_1/ground_truth.csv if present)",
    )
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for evaluation")
    p.add_argument(
        "--infer-script",
        type=Path,
        default=None,
        help="Path to deepSORT_rtdetr.py (default: repo root)",
    )
    p.add_argument(
        "--inference-cwd",
        type=Path,
        default=None,
        help="Working directory for inference (default: inference_parameter/, for relative model paths in YAML)",
    )
    p.add_argument(
        "--best-out",
        type=Path,
        default=None,
        help="Where to write best_config.yaml (default: <trials-dir>/best_config.yaml)",
    )
    p.add_argument(
        "--eval-plots",
        action="store_true",
        help="Write PR / IoU plots for every trial (slower; default off)",
    )
    args = p.parse_args()

    gt = args.gt
    if gt is None:
        gt = default_ground_truth()
    if not gt.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {gt}. Pass --gt explicitly."
        )

    run_trials(
        Path(args.trials_dir).resolve(),
        Path(args.video),
        Path(gt),
        infer_script=args.infer_script,
        inference_cwd=args.inference_cwd,
        iou_thresh=args.iou,
        best_out=args.best_out,
        eval_plots=args.eval_plots,
    )


if __name__ == "__main__":
    main()
