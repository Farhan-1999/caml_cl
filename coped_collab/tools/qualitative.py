"""Qualitative comparison in the style of CAML Fig. 5: for a few test frames, both robots' RGB input,
the ground truth and each model's prediction, plus the teacher with the ego robot alone.

Each prediction tile shows that frame's mIoU. Uses the best checkpoints; a few frames run on CPU.

Examples:
    python tools/qualitative.py --manifest configs/caml/experiments.yaml --seed 0
    python tools/qualitative.py --manifest configs/caml/experiments.yaml --models caml_teacher_rgb_depth_ir \
        caml_student_rgb --frames 6
Writes <train.out_dir>/_report/fig_qualitative.png (_report_smoke with --smoke).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import resolve_path  # noqa: E402
from coped.data.classes import CLASS_NAMES  # noqa: E402
from coped.data.dataset import build_dataset  # noqa: E402
from coped.engine import load_run_model, run_dir_for, run_suffix, to_device  # noqa: E402
from coped.experiments import load_manifest, planned_configs  # noqa: E402
from coped.metrics import ConfusionMatrix  # noqa: E402
from coped.visualize import CLASS_COLORS, colorize, to_image  # noqa: E402

DEFAULT_MODELS = ["caml_teacher_rgb_depth_ir", "caml_student_rgb", "caml_student_rgb_from_rgb_depth",
                  "caml_student_rgb_nokd"]
TILE = 200


def put(img: np.ndarray, text: str, y: int = 18, scale: float = 0.45) -> None:
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def frame_miou(pred: torch.Tensor, label: torch.Tensor, cfg: dict) -> float:
    cm = ConfusionMatrix(cfg["data"]["num_classes"], cfg["data"]["ignore_index"])
    cm.update(pred.cpu(), label.cpu())
    return cm.compute()["miou"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="config names, shown left to right")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=4, help="test frames, evenly spaced over the test split")
    parser.add_argument("--sample-ids", nargs="*", default=[], help="specific test samples instead")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    suffix = run_suffix(smoke=args.smoke)
    manifest = load_manifest(args.manifest)
    plan = {p.name: p for p in planned_configs(manifest, args.set, include_optional=True)}
    models = []
    for name in args.models:
        if name not in plan:
            raise SystemExit(f"{name} is not in {args.manifest}")
        run = run_dir_for(plan[name].cfg, args.seed, suffix)
        if not (run / "best.pt").exists():
            print(f"skipping {name}: no checkpoint in {run}")
            continue
        model, _ = load_run_model(run, "best", device)
        models.append((plan[name], model))
    if not models:
        raise SystemExit("no trained models found")
    cfg = models[0][0].cfg
    d = cfg["data"]
    ego, collab = d["ego"], d["collaborators"]
    ds = build_dataset(cfg, "test")
    if args.sample_ids:
        index = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
        picks = [index[s] for s in args.sample_ids]
    else:
        picks = np.linspace(0, len(ds) - 1, args.frames + 2)[1:-1].astype(int).tolist()

    teacher = next(((p, m) for p, m in models if not (p.cfg.get("distill") or {}).get("teacher_run")
                    and "teacher" in p.name), None)
    columns = [f"{ego} RGB (noisy input)", f"{collab[0]} RGB", "ground truth"]
    columns += [f"{p.experiment} {p.name.replace('caml_', '')}" for p, _ in models]
    if teacher:
        columns.append(f"{teacher[0].experiment} teacher, {ego} alone")

    rows = []
    with torch.no_grad():
        for i in picks:
            item = ds[i]
            inputs = to_device({r: {m: x.unsqueeze(0) for m, x in mods.items()} for r, mods in item["inputs"].items()},
                               device)
            label = item["label"]
            tiles = [to_image(item["inputs"][ego]["rgb"]), to_image(item["inputs"][collab[0]]["rgb"]),
                     colorize(label.numpy(), d["ignore_index"])]
            preds = []
            for p, model in models:
                preds.append(model(inputs).argmax(1)[0])
            if teacher:
                tmodel = teacher[1]
                feats = tmodel.encode(inputs)
                alone = {r: list(tmodel.modalities) if r == ego else [] for r in tmodel.robots}
                preds.append(tmodel.forward_available(feats, alone, label.shape[-2:]).argmax(1)[0])
            for pred in preds:
                tile = colorize(pred.cpu().numpy(), d["ignore_index"])
                tile[label.numpy() == d["ignore_index"]] //= 3  # dim pixels that are not scored
                tiles.append(tile)
            tiles = [cv2.resize(t, (TILE, TILE), interpolation=cv2.INTER_NEAREST) for t in tiles]
            put(tiles[0], item["sample_id"])
            for k, pred in enumerate(preds):
                put(tiles[3 + k], f"mIoU {frame_miou(pred, label, cfg) * 100:.1f}")
            rows.append(np.hstack(tiles))

    width = rows[0].shape[1]
    header = np.full((44, width, 3), 255, np.uint8)
    for k, name in enumerate(columns):
        words, lines = name.split(" "), [""]
        for w in words:  # wrap to two lines inside the tile width
            if len(lines[-1]) + len(w) > 26 and len(lines) < 2:
                lines.append("")
            lines[-1] = (lines[-1] + " " + w).strip()
        for j, text in enumerate(lines):
            cv2.putText(header, text, (k * TILE + 4, 17 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1,
                        cv2.LINE_AA)
    names = CLASS_NAMES.get(d["sequence"], [str(c) for c in range(d["num_classes"])])
    legend = np.full((30, width, 3), 255, np.uint8)
    x = 6
    for c, name in enumerate(names):
        color = tuple(int(v) for v in CLASS_COLORS[c][::-1])
        cv2.rectangle(legend, (x, 9), (x + 14, 23), color, -1)
        cv2.putText(legend, name, (x + 18, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        x += 24 + 8 * len(name)
    cv2.rectangle(legend, (x, 9), (x + 14, 23), (0, 0, 0), -1)
    cv2.putText(legend, "unlabeled (dimmed in predictions)", (x + 18, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (40, 40, 40), 1, cv2.LINE_AA)
    out_dir = resolve_path(cfg["train"]["out_dir"]) / ("_report_smoke" if args.smoke else "_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig_qualitative.png"
    cv2.imwrite(str(out), np.vstack([header, *rows, legend]))
    print(f"{len(picks)} frames x {len(columns)} columns -> {out}")


if __name__ == "__main__":
    main()
