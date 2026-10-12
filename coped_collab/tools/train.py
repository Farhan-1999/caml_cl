"""Train a CAML teacher or student.

Examples:
    python tools/train.py --config configs/caml/teacher_rgb_depth_ir.yaml --seed 0
    python tools/train.py --config configs/caml/student_rgb.yaml --seed 0       # needs that seed's teacher
    python tools/train.py --config configs/caml/teacher_rgb_depth_ir.yaml --smoke --device cpu
    python tools/train.py --config configs/caml/teacher_rgb_depth_ir.yaml --overfit 8 --set train.epochs=60

--resume continues an interrupted run (and skips a finished one); --overwrite restarts it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import load_config  # noqa: E402
from coped.engine import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 epochs x 2 steps, 4-sample eval sets")
    parser.add_argument("--overfit", type=int, default=0, metavar="N", help="train and evaluate on N samples")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg, args.seed, torch.device(args.device), resume=args.resume, overwrite=args.overwrite,
          smoke=args.smoke, overfit=args.overfit)


if __name__ == "__main__":
    main()
