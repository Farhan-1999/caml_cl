"""Evaluate a trained run on a split: full inputs, with collaborators removed, or over every
combination of what each robot sends.

Examples:
    python tools/evaluate.py --run ../runs/caml_student_rgb/seed0
    python tools/evaluate.py --run ../runs/caml_student_rgb/seed0 --split val --checkpoint last
    python tools/evaluate.py --run ../runs/caml_student_rgb/seed0 --drop-robots race1
    python tools/evaluate.py --run ../runs/caml_teacher_rgb_depth_ir/seed0 --subsets all   # all 63 combinations

Writes eval_<split>_<checkpoint>[_without_<robots>|_subsets_<mode>].json in the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import load_config  # noqa: E402
from coped.data.classes import CLASS_NAMES  # noqa: E402
from coped.data.dataset import build_dataset  # noqa: E402
from coped.engine import (evaluate, evaluate_subsets, load_run_model, make_loader,  # noqa: E402
                          modality_combinations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory with config.yaml and checkpoints")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--drop-robots", nargs="*", default=[], help="robots whose features are zeroed")
    parser.add_argument("--subsets", choices=["selected", "robots", "all"],
                        help="evaluate input combinations: the config's eval.selected_subsets, each robot "
                             "alone (robots), or every combination (all)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides of the run config, e.g. data.export_root=/data/exported")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_run_model(args.run, args.checkpoint, device)
    cfg = load_config(args.run / "config.yaml", args.set)
    loader = make_loader(build_dataset(cfg, args.split), args.batch_size, False, args.workers, 0, device)
    amp = cfg["train"].get("amp", True)
    if args.subsets:
        combos = modality_combinations(model.robots, model.modalities, args.subsets,
                                       (cfg.get("eval") or {}).get("selected_subsets"))
        results = evaluate_subsets(model, loader, device, cfg, combos, amp)
        print(f"{args.run} [{args.checkpoint}] {args.split}: {len(combos)} input combinations")
        print(f"  {'what each robot sends':<50} {'mIoU':>6} {'common':>7}")
        for key, r in sorted(results.items(), key=lambda kv: -kv[1]["miou"]):
            print(f"  {key:<50} {r['miou'] * 100:6.2f} {r['miou_common'] * 100:7.2f}")
        out = args.run / f"eval_{args.split}_{args.checkpoint}_subsets_{args.subsets}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"-> {out}")
        return
    result = evaluate(model, loader, device, cfg, drop_robots=args.drop_robots, amp=amp)

    names = CLASS_NAMES.get(cfg["data"]["sequence"], [str(c) for c in range(cfg["data"]["num_classes"])])
    print(f"{args.run} [{args.checkpoint}] {args.split}"
          + (f" without {args.drop_robots}" if args.drop_robots else ""))
    print(f"  mIoU {result['miou'] * 100:.2f}  mIoU common {result['miou_common'] * 100:.2f}  "
          f"pixel acc {result['pixel_accuracy'] * 100:.2f}")
    for c, iou in enumerate(result["iou"]):
        print(f"  {c} {names[c]:<16} {'absent' if iou is None else f'{iou * 100:.2f}'}")
    suffix = f"_without_{'_'.join(args.drop_robots)}" if args.drop_robots else ""
    out = args.run / f"eval_{args.split}_{args.checkpoint}{suffix}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
