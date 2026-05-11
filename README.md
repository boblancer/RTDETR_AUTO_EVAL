# RTDETR_AUTO_EVAL

Auto-generate RT-DETR inference trial configs, run them on a labeled video, and pick the best YAML by detection metrics.

## One-command pipeline

From the **repository root** (so imports resolve):

```bash
python -m rtdetr_eval full \
  --config inference_parameter/base.yaml \
  --trials-dir inference_parameter/trials/cam_01 \
  --n 20 \
  --seed 42 \
  --video /path/to/your_video.mp4
```

Ground truth defaults to `inference_parameter/camera_1/ground_truth.csv` if that file exists; otherwise pass `--gt /path/to/gt.csv`.

Outputs:

- `trial_*.yaml` and `manifest.json` under `--trials-dir`
- `<trials-dir>/best_config.yaml` — winning hyperparameters
- `<trials-dir>/trial_leaderboard.json` — all trials with scores

## Step-by-step

```bash
python -m rtdetr_eval generate \
  --config inference_parameter/base.yaml \
  --out-dir inference_parameter/trials/cam_01 \
  --n 50 --seed 42

python -m rtdetr_eval run-trials \
  --trials-dir inference_parameter/trials/cam_01 \
  --video /path/to/your_video.mp4
```

Single evaluation (plots + JSON):

```bash
python -m rtdetr_eval eval \
  --gt inference_parameter/camera_1/ground_truth.csv \
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
