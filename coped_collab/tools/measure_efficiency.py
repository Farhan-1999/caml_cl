"""Measure the efficiency of every model in a manifest (CAML Table 1 / Table 8 analogues).

Per config, on the chosen device, in eval mode with the config's mixed-precision setting:
    parameters        total and per part
    messages          what each collaborator transmits per frame (fp32 / fp16 bytes)
    batch_ms          one forward pass of a batch (CAML reports latency per batch)
    frame_ms          one forward pass of one frame, all robots
    robot_ms          per robot: computing its messages for one frame (runs on that robot)
    server_ms         fusing all robots' messages and decoding the ego view (ego robot / edge server)
    epoch_s           mean training time per epoch, read from finished runs when they exist
Latencies are CUDA-synchronized means over --iters passes after --warmup passes. Weights are
random: speed does not depend on them, so this runs before any training (and on a Jetson).

Example:
    python tools/measure_efficiency.py --manifest configs/caml/experiments.yaml --optional
Writes <train.out_dir>/_efficiency/efficiency_<device>.json and .md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import resolve_path  # noqa: E402
from coped.engine import autocast_for, message_sizes, run_dir_for  # noqa: E402
from coped.experiments import load_manifest, planned_configs  # noqa: E402
from coped.models import build_model, parameter_counts  # noqa: E402


def time_ms(fn, iters: int, warmup: int, device: torch.device) -> dict:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean": round(float(np.mean(times)), 3), "std": round(float(np.std(times)), 3)}


def measure(cfg: dict, device: torch.device, batch: int, iters: int, warmup: int, seeds: list[int]) -> dict:
    cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(cfg).to(device).eval()
    s = cfg["data"]["image_size"]
    autocast, dtype = autocast_for(device, cfg["train"].get("amp", True))
    ctx = autocast if dtype else contextlib.nullcontext

    def inputs(n):
        return {r: {m: torch.randn(n, 3, s, s, device=device) for m in model.modalities} for r in model.robots}

    big, one = inputs(batch), inputs(1)
    with torch.no_grad(), ctx():
        msgs = {r: model.robot_forward(one[r]) for r in model.robots}
        result = {
            "batch_size": batch,
            "batch_ms": time_ms(lambda: model(big), iters, warmup, device),
            "frame_ms": time_ms(lambda: model(one), iters, warmup, device),
            "robot_ms": {r: time_ms(lambda r=r: model.robot_forward(one[r]), iters, warmup, device)
                         for r in model.robots},
            "server_ms": time_ms(lambda: model.server_forward(msgs, (s, s)), iters, warmup, device),
        }
    epoch_times = []
    for seed in seeds:
        info = run_dir_for(cfg, seed) / "model_info.json"
        if info.exists():
            epoch_times.append(json.loads(info.read_text())["mean_epoch_time_s"])
    result.update({
        "parameters": parameter_counts(model),
        "messages": message_sizes(model, cfg, device),
        "epoch_s": round(float(np.mean(epoch_times)), 2) if epoch_times else None,
        "amp_dtype": str(dtype),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--only", nargs="*", default=[], help="experiment ids, e.g. E1 E2")
    parser.add_argument("--optional", action="store_true", help="include experiments marked optional")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = load_manifest(args.manifest)
    configs = planned_configs(manifest, args.set, args.only, args.optional)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = True
    results = {}
    for c in configs:
        print(f"{c.experiment} {c.name} ...", flush=True)
        results[c.name] = {"experiment": c.experiment, "fusion": c.cfg["model"]["fusion"],
                           "modalities": c.cfg["model"]["modalities"],
                           **measure(c.cfg, device, args.batch_size, args.iters, args.warmup, manifest["seeds"])}

    out_dir = resolve_path(configs[0].cfg["train"]["out_dir"]) / "_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", device_name).strip("_").lower()
    payload = {"device": device_name, "iters": args.iters, "warmup": args.warmup, "results": results}
    (out_dir / f"efficiency_{tag}.json").write_text(json.dumps(payload, indent=2))

    ego = configs[0].cfg["data"]["ego"]
    lines = [f"# Efficiency on {device_name}", "",
             f"Latency in ms (mean of {args.iters} passes); messages are what each collaborator sends per frame.", "",
             "| Exp | Model | Params (M) | Sent per frame (KiB fp16) | Robot encode (ms) | Server fuse+decode (ms) "
             f"| Frame (ms) | Batch {args.batch_size} (ms) | Train s/epoch |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        sent = sum(m["bytes_fp16"] for m in r["messages"].values()) / 1024
        robots = ", ".join(f"{k} {v['mean']:.1f}" for k, v in r["robot_ms"].items() if k != ego)
        lines.append(f"| {r['experiment']} | {name} | {r['parameters']['total'] / 1e6:.1f} | {sent:.0f} | {robots} | "
                     f"{r['server_ms']['mean']:.1f} | {r['frame_ms']['mean']:.1f} | {r['batch_ms']['mean']:.1f} | "
                     f"{r['epoch_s'] if r['epoch_s'] is not None else '-'} |")
    (out_dir / f"efficiency_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:]))
    print(f"-> {out_dir / f'efficiency_{tag}.json'}")


if __name__ == "__main__":
    main()
