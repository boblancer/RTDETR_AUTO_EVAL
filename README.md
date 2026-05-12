# RTDETR_AUTO_EVAL

Auto-generate RT-DETR inference trial configs, run them on a labeled video, and pick the best YAML by detection metrics.

## Folder layout

- `data/` contains **datasets** (videos, ground truth, camera-specific assets)
- `runs/` contains **outputs** (trial YAML sweeps, leaderboards, best config)

## One-command pipeline

From the **repository root** (so imports resolve):

```bash
python -m rtdetr_eval full \
  --config inference_parameter/base.yaml \
  --n 20 \
  --seed 42 \
  --video /path/to/your_video.mp4
```

Omitting `--trials-dir` uses `runs/camera_1/trials`. Override with `--trials-dir /other/path` if needed.

Omitting `--video` uses `data/camera_1/videos/trim5.mp4` when that file exists, or the only `.mp4` in that folder. Otherwise pass `--video /path/to/clip.mp4`.

Ground truth defaults to `data/camera_1/ground_truth.csv` if that file exists; otherwise pass `--gt /path/to/gt.csv`.

Outputs:

- `trial_*.yaml` and `manifest.json` under the trials directory
- `best_config.yaml` and `trial_leaderboard.json` in that same directory

## Step-by-step

```bash
python -m rtdetr_eval generate \
  --config inference_parameter/base.yaml \
  --n 50 --seed 42

python -m rtdetr_eval run-trials
```

Single evaluation (plots + JSON):

```bash
python -m rtdetr_eval eval \
  --gt data/camera_1/ground_truth.csv \
  --pred /path/to/video_rtdetr.csv \
  --out ./eval_out
```

## Paths and working directory

Inference runs with **cwd** `inference_parameter/` by default so relative paths in YAML (for example `model: ./best.pt`) match a typical layout. Override with `--inference-cwd` if your weights live elsewhere.

```bash
python -m rtdetr_eval print-paths
```

## Legacy scripts

`inference_parameter/generate_configs.py`, `run_trials.py`, and `evaluate_detections.py` still work when executed as files; they delegate to the same package (repo root is added to `sys.path` automatically).
