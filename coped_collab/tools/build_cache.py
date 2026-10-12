"""Pre-resize every split's inputs and labels into uint8 .npy arrays (optional, speeds up training).

Writes <data.cache_dir>/<split>/<robot>.<modality>.npy, label.npy and meta.json. The arrays hold
clean images; noise is still added on the fly. The dataset refuses a cache built for other samples.

Example:
    python tools/build_cache.py --config configs/track_a.yaml --set data.cache_dir=../cache/track_a
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import load_config, resolve_path  # noqa: E402
from coped.data.dataset import build_dataset, split_fingerprint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if not cfg["data"].get("cache_dir"):
        raise SystemExit("set data.cache_dir (e.g. --set data.cache_dir=../cache/track_a)")
    root = resolve_path(cfg["data"]["cache_dir"])
    for split in cfg["data"]["split"]["parts"]:
        ds = build_dataset(cfg, split, use_cache=False, noise=False)
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        first = ds.raw(0)
        arrays = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.uint8,
                                               shape=(len(ds), *v.shape)) for k, v in first.items()}

        def fill(i):
            for k, v in ds.raw(i).items():
                arrays[k][i] = v

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(fill, range(len(ds))))
        for a in arrays.values():
            a.flush()
        (out / "meta.json").write_text(json.dumps({
            "fingerprint": split_fingerprint(ds.rows), "image_size": ds.size, "samples": len(ds),
            "keys": sorted(arrays)}, indent=2))
        size = sum((out / f"{k}.npy").stat().st_size for k in arrays) / 1e9
        print(f"{split}: {len(ds)} samples, {size:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
