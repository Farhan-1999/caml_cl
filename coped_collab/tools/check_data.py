"""Smoke-test the data pipeline of a config on CPU and render a preview of test samples.

Checks per split: samples load through a DataLoader with the expected shapes and value ranges,
labels only hold class ids or the ignore value, val/test noise is identical across loads while
training noise changes, and noise touches only the configured robots and modalities. If a cache
is configured, cached samples must equal samples decoded from the export.

Example:
    python tools/check_data.py --config configs/track_a.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import load_config, resolve_path  # noqa: E402
from coped.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_dataset  # noqa: E402
from coped.visualize import colorize, to_image  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--preview", type=int, default=4, help="test samples in the preview image")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    robots, modalities = [d["ego"], *d["collaborators"]], d["modalities"]
    noise = d.get("noise") or {}
    failures = []

    def check(ok: bool, msg: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    for split in d["split"]["parts"]:
        ds = build_dataset(cfg, split)
        print(f"[{split}] {len(ds)} samples")
        t = time.time()
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)))
        per_sample = (time.time() - t) / 4
        shapes_ok = all(batch["inputs"][r][m].shape == (4, 3, d["image_size"], d["image_size"])
                        for r in robots for m in modalities)
        check(shapes_ok, f"input shapes (4, 3, {d['image_size']}, {d['image_size']}) for {robots} x {modalities}")
        check(batch["label"].shape == (4, d["image_size"], d["image_size"]) and batch["label"].dtype == torch.int64,
              "label shape and dtype")
        values = set(torch.unique(batch["label"]).tolist())
        check(values <= set(range(d["num_classes"])) | {d["ignore_index"]}, f"label values {sorted(values)}")
        finite = all(torch.isfinite(batch["inputs"][r][m]).all() for r in robots for m in modalities)
        check(finite, f"inputs finite ({per_sample * 1000:.0f} ms per sample, no workers)")

        a, b = ds[0], ds[0]
        same = {(r, m): torch.equal(a["inputs"][r][m], b["inputs"][r][m]) for r in robots for m in modalities}
        noisy = {(r, m) for r in robots for m in modalities
                 if noise.get("enabled") and r in noise["robots"] and m in noise["modalities"]}
        if split == "train":
            check(all(not same[k] for k in noisy) and all(same[k] for k in same if k not in noisy),
                  "training noise is redrawn for noisy streams, other streams unchanged")
        else:
            check(all(same.values()), "val/test noise identical across loads")

        clean = build_dataset(cfg, split, noise=False)[0]
        diff = {(r, m): float((a["inputs"][r][m] - clean["inputs"][r][m]).abs().max()) for r in robots for m in modalities}
        check(all((diff[k] > 0) == (k in noisy) for k in diff),
              f"noise only on {sorted(f'{r}.{m}' for r, m in noisy)}")
        if noisy:
            r, m = sorted(noisy)[0]
            x = (a["inputs"][r][m] - clean["inputs"][r][m]) * IMAGENET_STD  # back to [0, 1] units
            sigma = float(x[:, 20:-20, 20:-20].std())
            check(abs(sigma - noise["gaussian_sigma"]) < 0.03,
                  f"measured noise std {sigma:.3f} vs sigma {noise['gaussian_sigma']} ({r}.{m}, clipping lowers it)")

        if d.get("cache_dir"):
            raw_cached, raw_file = ds.raw(0), build_dataset(cfg, split, use_cache=False).raw(0)
            check(all(np.array_equal(raw_cached[k], raw_file[k]) for k in raw_file), "cache equals decoded export")

    test = build_dataset(cfg, "test")
    picks = np.linspace(0, len(test) - 1, args.preview).astype(int)
    tile = 160
    rows = []
    for i in picks:
        item = test[int(i)]
        tiles = [to_image(item["inputs"][r][m]) for r in robots for m in modalities]
        tiles.append(colorize(item["label"].numpy(), d["ignore_index"]))
        rows.append(np.hstack([cv2.resize(t, (tile, tile), interpolation=cv2.INTER_NEAREST) for t in tiles]))
    head = np.zeros((24, rows[0].shape[1], 3), np.uint8)
    for k, name in enumerate([f"{r} {m}" for r in robots for m in modalities] + [f"{d['ego']} label"]):
        cv2.putText(head, name, (k * tile + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out = resolve_path(d["splits_dir"]) / "preview.jpg"
    cv2.imwrite(str(out), np.vstack([head, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"preview -> {out}")
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
