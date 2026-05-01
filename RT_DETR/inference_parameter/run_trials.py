"""
run_trials.py — Run every trial config against a single video and score it.

Usage:
    python3 run_trials.py --trials-dir trials/cam_01 --video ../trim1.mp4
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

# Inference command
INFERENCE_CMD = "python3 ../deepSORT_rtdetr.py --config {config} -i {video} --csv"
EVAL_CMD = "python3 evaluate_detections.py --gt ./camera1/ground_truth.csv --pred {predictions} --out {results} --iou 0.5"

def predictions_path(video: Path) -> str:
    return video.stem + ".csv"   # adjust if your script writes a fixed filename


def run(cmd):
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def score(metrics: dict) -> float:
    ped_ap = metrics["classes"]["Pedestrian"]["ap"]
    cyc_ap = metrics["classes"]["Cyclist"]["ap"]
    aps    = [ped_ap] + ([cyc_ap] if cyc_ap > 0 else [])
    ap     = sum(aps) / len(aps)
    iou    = metrics["iou_stats"]["mean"]
    return ap if iou >= 0.65 else ap * 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials-dir", type=Path, required=True)
    p.add_argument("--video",      type=Path, required=True)
    args = p.parse_args()

    trials  = sorted(args.trials_dir.glob("trial_*.yaml"))
    video   = args.video
    results = []

    print("ALL trials", trials)
    for trial in trials:
        print(f"\n{trial.stem}")
        results_json = f"_eval_{trial.stem}.json"

        run(INFERENCE_CMD.format(config=trial, video=video))
        run(EVAL_CMD.format(predictions=predictions_path(video), results=results_json))

        with open(results_json) as f:
            metrics = json.load(f)
        os.unlink(results_json)

        s = score(metrics)
        print(f"  score={s:.4f}  ap={metrics['classes']['Pedestrian']['ap']:.4f}  "
              f"recall={metrics['classes']['Pedestrian']['recall']:.4f}")
        results.append((s, trial.stem))

    results.sort(reverse=True)
    print("\n--- Top 5 ---")
    for s, name in results[:5]:
        print(f"  {name}  score={s:.4f}")

    best_name = results[0][1]
    best_yaml = args.trials_dir / f"{best_name}.yaml"
    cfg = yaml.safe_load(best_yaml.read_text())
    cfg["input"] = ""
    with open("best_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved to best_config.yaml ({best_name})")


if __name__ == "__main__":
    main()

# python3 run_trials.py --trials-dir trials/cam_01 --video ./camera1/trim5.mp4