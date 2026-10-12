"""Build the model of each config and check it: parameter counts, output shape, removing collaborators,
and the per-robot messages a collaborator would transmit.

Example:
    python tools/check_model.py --configs configs/caml/*.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import load_config  # noqa: E402
from coped.engine import message_sizes, modality_combinations  # noqa: E402
from coped.models import build_model, parameter_counts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for path in args.configs:
        cfg = load_config(path, args.set)
        if "model" not in cfg or "data" not in cfg:
            print(f"{path}: not a model config, skipped")
            continue
        d = cfg["data"]
        model = build_model(cfg).to(device).eval()
        robots, s = model.robots, d["image_size"]
        inputs = {r: {m: torch.randn(2, 3, s, s, device=device) for m in model.modalities} for r in robots}
        with torch.no_grad():
            out = model(inputs)
            alone = model(inputs, drop_robots=d["collaborators"])
            split = model.server_forward({r: model.robot_forward(inputs[r]) for r in robots}, (s, s))
            feats = model.encode(inputs)
            combos = modality_combinations(robots, model.modalities, "all")
            ego_only = {r: list(model.modalities) if r == d["ego"] else [] for r in robots}
            subset_ok = (torch.allclose(model.forward_available(feats, combos[0], (s, s)), out, atol=1e-4)
                         and torch.allclose(model.forward_available(feats, ego_only, (s, s)), alone, atol=1e-4))
            for combo in combos:  # every combination runs
                model.forward_available(feats, combo, (s, s))
        params = parameter_counts(model)
        print(f"{cfg['name']}: fusion={model.fusion} modalities={model.modalities} robots={robots}")
        print("  parameters (M): " + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in params.items()))
        ok_shape = tuple(out.shape) == (2, d["num_classes"], s, s)
        same_split = torch.allclose(out, split, atol=1e-4)
        print(f"  output {tuple(out.shape)} {'ok' if ok_shape else 'WRONG'}; "
              f"without {d['collaborators']}: changes output = {not torch.allclose(out, alone)}; "
              f"robot/server split matches = {same_split}")
        print(f"  {len(combos)} input combinations run; full and {d['ego']}-only match the plain forward = {subset_ok}")
        for key, m in message_sizes(model, cfg, device).items():
            print(f"  message {key}: {m['shape']} = {m['bytes_fp32'] / 1024:.0f} KiB fp32, {m['bytes_fp16'] / 1024:.0f} KiB fp16")
        if not (ok_shape and same_split and subset_ok):
            raise SystemExit("model check failed")


if __name__ == "__main__":
    main()
