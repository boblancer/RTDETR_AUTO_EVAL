"""Unified CLI: generate configs, run trials, evaluate, or full pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rtdetr_eval.evaluate import run_evaluation
from rtdetr_eval.generate import generate_trials
from rtdetr_eval.paths import (
    default_eval_video,
    default_ground_truth,
    default_trials_dir,
    inference_dir,
    repo_root,
    resolve_eval_video,
    resolve_ground_truth,
)
from rtdetr_eval.trials import run_trials


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rtdetr_eval",
        description="RT-DETR hyperparameter sweep: generate trials, run inference, evaluate vs GT.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Sample trial YAMLs from a base config")
    g.add_argument("--config", type=Path, required=True, help="Base YAML template")
    g.add_argument("--n", type=int, default=50, help="Number of new trials")
    g.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Directory for trial_*.yaml + manifest.json (default: {default_trials_dir()})",
    )
    g.add_argument("--seed", type=int, default=None)

    r = sub.add_parser("run-trials", help="Run each trial on a video and write best_config.yaml")
    r.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help=f"Trial YAML directory (default: {default_trials_dir()})",
    )
    r.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video (default: data/camera_1/videos/trim5.mp4 if present)",
    )
    r.add_argument("--gt", type=Path, default=None)
    r.add_argument("--iou", type=float, default=0.5)
    r.add_argument("--infer-script", type=Path, default=None)
    r.add_argument("--inference-cwd", type=Path, default=None)
    r.add_argument("--best-out", type=Path, default=None)
    r.add_argument("--eval-plots", action="store_true")

    e = sub.add_parser("eval", help="Single GT vs predictions evaluation (+ plots)")
    e.add_argument("--gt", type=Path, required=True)
    e.add_argument("--pred", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("."))
    e.add_argument("--iou", type=float, default=0.5)

    f = sub.add_parser("full", help="generate then run-trials (one command)")
    f.add_argument("--config", type=Path, required=True, help="Base YAML template")
    f.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help=f"Trial YAML directory (default: {default_trials_dir()})",
    )
    f.add_argument("--n", type=int, default=50)
    f.add_argument("--seed", type=int, default=None)
    f.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video (default: data/camera_1/videos/trim5.mp4 if present)",
    )
    f.add_argument("--gt", type=Path, default=None)
    f.add_argument("--iou", type=float, default=0.5)
    f.add_argument("--infer-script", type=Path, default=None)
    f.add_argument("--inference-cwd", type=Path, default=None)
    f.add_argument("--best-out", type=Path, default=None)
    f.add_argument("--eval-plots", action="store_true")

    sub.add_parser("print-paths", help="Show resolved repo defaults")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "print-paths":
        root = repo_root()
        print(f"repo_root:          {root}")
        print(f"inference_parameter: {inference_dir()}")
        print(f"default --gt:         {default_ground_truth()}")
        print(f"default trials dir:   {default_trials_dir()}")
        dv = default_eval_video()
        print(f"default --video:      {dv if dv else '(none — add trim5.mp4 under data/camera_1/videos/ or pass --video)'}")
        print(f"deepSORT_rtdetr:      {root / 'deepSORT_rtdetr.py'}")
        return

    if args.command == "generate":
        out_dir = args.out_dir if args.out_dir is not None else default_trials_dir()
        generate_trials(args.config.resolve(), out_dir.resolve(), args.n, args.seed)
        return

    if args.command == "eval":
        run_evaluation(args.gt, args.pred, args.out.resolve(), args.iou, plots=True, write_json=True)
        return

    if args.command == "run-trials":
        if args.trials_dir is None:
            args.trials_dir = default_trials_dir()
        gt = resolve_ground_truth(args.gt)
        video = resolve_eval_video(args.video)
        run_trials(
            args.trials_dir.resolve(),
            video,
            gt,
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            iou_thresh=args.iou,
            best_out=args.best_out,
            eval_plots=args.eval_plots,
        )
        return

    if args.command == "full":
        if args.trials_dir is None:
            args.trials_dir = default_trials_dir()
        gt = resolve_ground_truth(args.gt)
        video = resolve_eval_video(args.video)
        generate_trials(args.config.resolve(), args.trials_dir.resolve(), args.n, args.seed)
        run_trials(
            args.trials_dir.resolve(),
            video,
            gt,
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            iou_thresh=args.iou,
            best_out=args.best_out,
            eval_plots=args.eval_plots,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
