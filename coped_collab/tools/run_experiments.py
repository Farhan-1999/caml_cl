"""Run an experiment suite: every config of a manifest for every seed, teachers before their students.

Each run is `tools/train.py --resume` in its own process, so starting the suite again continues
interrupted runs and skips finished ones. A student starts only once its teacher (same seed) has
finished; a failed teacher makes its students wait for the next invocation.
Full output of each run goes to <train.out_dir>/_logs/<name>_seed<k>.log.

Examples:
    python tools/run_experiments.py --manifest configs/caml/experiments.yaml --dry-run
    python tools/run_experiments.py --manifest configs/caml/experiments.yaml
    python tools/run_experiments.py --manifest configs/caml/experiments.yaml --only E1 E2 --seeds 0
    python tools/run_experiments.py --manifest configs/caml/experiments.yaml --smoke   # 5-minute end-to-end test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
from coped.config import resolve_path  # noqa: E402
from coped.engine import run_dir_for, run_suffix, teacher_run_dir  # noqa: E402
from coped.experiments import load_manifest, planned_configs, training_order  # noqa: E402


def status(run_dir: Path) -> str:
    if (run_dir / "metrics.json").exists():
        return "done"
    if (run_dir / "log.csv").exists():
        epochs = max(0, sum(1 for _ in open(run_dir / "log.csv")) - 1)
        return f"partial ({epochs} epochs)"
    return "todo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, help="default: the manifest's seeds")
    parser.add_argument("--only", nargs="*", default=[], help="experiment ids to run, e.g. E1 E2")
    parser.add_argument("--optional", action="store_true", help="include experiments marked optional")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="tiny runs into <name>_smoke folders")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--verbose", action="store_true", help="echo every line of every run")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    seeds = args.seeds if args.seeds else manifest["seeds"]
    plan = training_order(planned_configs(manifest, args.set, args.only, args.optional))
    if not plan:
        raise SystemExit("nothing to run")
    suffix = run_suffix(smoke=args.smoke)
    log_dir = resolve_path(plan[0].cfg["train"]["out_dir"]) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(plan)} configs x {len(seeds)} seeds on {args.device}{' (smoke)' if args.smoke else ''}")
    for seed in seeds:
        for c in plan:
            teacher = f" <- {teacher_run_dir(c.cfg, seed, suffix).parent.name}" if c.is_student else ""
            print(f"  seed {seed}  {c.experiment:<3} {c.name + suffix:<45} {status(run_dir_for(c.cfg, seed, suffix))}{teacher}")
    if args.dry_run:
        return

    counts = {"done": 0, "trained": 0, "failed": 0, "waiting": 0}
    for seed in seeds:
        for c in plan:
            run_dir = run_dir_for(c.cfg, seed, suffix)
            if status(run_dir) == "done":
                counts["done"] += 1
                continue
            if c.is_student and not (teacher_run_dir(c.cfg, seed, suffix) / "metrics.json").exists():
                print(f"[seed {seed}] {c.name}: teacher not finished, skipped")
                counts["waiting"] += 1
                continue
            cmd = [sys.executable, str(TOOLS / "train.py"), "--config", str(c.path), "--seed", str(seed),
                   "--resume", "--device", args.device]
            if args.smoke:
                cmd.append("--smoke")
            if args.set:
                cmd += ["--set", *args.set]
            log_path = log_dir / f"{c.name}{suffix}_seed{seed}.log"
            print(f"[seed {seed}] {c.experiment} {c.name}: training ({log_path.name})", flush=True)
            t0 = time.time()
            tail = []
            with open(log_path, "a") as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    log.write(line)
                    tail = (tail + [line.rstrip()])[-20:]
                    if args.verbose:
                        print("    " + line.rstrip(), flush=True)
                code = proc.wait()
            minutes = (time.time() - t0) / 60
            if code != 0:
                counts["failed"] += 1
                print(f"    FAILED after {minutes:.1f} min (exit {code}); last lines:")
                print("\n".join("      " + t for t in tail))
                continue
            counts["trained"] += 1
            best = json.loads((run_dir / "metrics.json").read_text())["best"]
            print(f"    done in {minutes:.1f} min: val mIoU {best['val']['miou'] * 100:.2f}, "
                  f"test mIoU {best['test']['miou'] * 100:.2f}", flush=True)

    print("summary: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if counts["failed"] or counts["waiting"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
