"""Build the train/val/test split files of a config, plus a report on them.

Writes into data.splits_dir:
    train.csv, val.csv, test.csv   one row per sample: sample_id, split, time_s, ego_stamp_ns, label,
                                   and "<robot>.<modality>" path + ".stamp_ns" for every robot/modality
                                   (paths are relative to data.export_root)
    split_report.json / .md        window, sample counts, time blocks, pairing offsets, class shares

Example:
    python tools/build_splits.py --config configs/track_a.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from coped.data.classes import CLASS_NAMES  # noqa: E402
from coped.data.index import build_samples, flight_window  # noqa: E402
from coped.data.splits import assign_splits, subsample  # noqa: E402


def stamp_name(ns: int) -> str:
    return f"{ns // 10**9}_{ns % 10**9:09d}"


def time_blocks(times_s: np.ndarray, gap_s: float = 1.0) -> list[list[float]]:
    """Contiguous [start, end] stretches of a split, in seconds from the window start."""
    if len(times_s) == 0:
        return []
    breaks = np.where(np.diff(times_s) > gap_s)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(times_s) - 1]
    return [[round(float(times_s[s]), 2), round(float(times_s[e]), 2)] for s, e in zip(starts, ends)]


def class_shares(export_root: Path, label_paths: list[str], num_classes: int, ignore: int) -> dict:
    def count(path):
        lab = cv2.imread(str(export_root / path), cv2.IMREAD_UNCHANGED)
        return np.bincount(lab.ravel(), minlength=256)

    with ThreadPoolExecutor(8) as pool:
        total = np.sum(list(pool.map(count, label_paths)), axis=0)
    pixels = total.sum()
    stray = int(total[num_classes:].sum() - total[ignore])  # ids outside the class range, should be 0
    return {"class_share": [round(float(total[c] / pixels), 5) for c in range(num_classes)],
            "unlabeled_share": round(float(total[ignore] / pixels), 5),
            "out_of_range_pixels": stray}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    export_root, out_dir = resolve_path(d["export_root"]), resolve_path(d["splits_dir"])
    robots = [d["ego"], *d["collaborators"]]
    modalities = d["modalities"]

    window = None
    w = d.get("window") or {}
    if w.get("flight_of"):
        window = flight_window(export_root, d["sequence"], w["flight_of"], w["min_height_m"], w["trim_s"])
    built = build_samples(export_root, d["sequence"], d["ego"], d["collaborators"], modalities,
                          d["pairing_tolerance_s"], window)
    cols = built["columns"]
    if len(cols["ego_stamp_ns"]) == 0:
        raise SystemExit("no complete samples; check the window and pairing tolerance")
    t0 = window[0] if window else int(cols["ego_stamp_ns"][0])
    times = (cols["ego_stamp_ns"] - t0) / 1e9
    s = d["split"]
    split = assign_splits(times, s["period_s"], s["parts"], s["buffer_s"])

    out_dir.mkdir(parents=True, exist_ok=True)
    data_cols = [f"{r}.{m}{suffix}" for r in robots for m in modalities for suffix in ("", ".stamp_ns")]
    header = ["sample_id", "split", "time_s", "ego_stamp_ns", "label", *data_cols]
    report = {
        # relative path, so the report is byte-identical on every machine that rebuilds the splits
        "config": Path(cfg["config_path"]).relative_to(PROJECT_ROOT).as_posix(),
        "sequence": d["sequence"], "robots": robots, "modalities": modalities,
        "window_ns": list(window) if window else None,
        "window_s": round((window[1] - window[0]) / 1e9, 2) if window else None,
        "complete_samples_30hz": int(len(times)),
        "dropped": built["dropped"],
        "in_buffers_30hz": int(np.sum(split == None)),  # noqa: E711 (object array comparison)
        "pairing_offset_ms": {},
        "splits": {},
    }
    ego_ns = cols["ego_stamp_ns"]
    for r in robots:
        for m in modalities:
            off = np.abs(cols[f"{r}.{m}.stamp_ns"] - ego_ns) / 1e6
            report["pairing_offset_ms"][f"{r}.{m}"] = {
                "median": round(float(np.median(off)), 2), "p95": round(float(np.percentile(off, 95)), 2),
                "max": round(float(off.max()), 2)}

    for name in s["parts"]:
        idx = np.where(split == name)[0]
        idx = idx[subsample(times[idx], s["rate_hz"].get(name))]
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")  # LF on every OS, so rebuilt splits are byte-identical
            writer.writerow(header)
            for i in idx:
                writer.writerow([stamp_name(int(ego_ns[i])), name, f"{times[i]:.3f}", int(ego_ns[i]),
                                 cols["label"][i], *[cols[c][i] for c in data_cols]])
        blocks = time_blocks(times[idx])
        report["splits"][name] = {
            "samples": int(len(idx)),
            "rate_hz": s["rate_hz"].get(name) or "all frames",
            "seconds_covered": round(sum(b[1] - b[0] for b in blocks), 1),
            "blocks_s": blocks,
            **class_shares(export_root, list(cols["label"][idx]), d["num_classes"], d["ignore_index"]),
        }
        print(f"{name}: {len(idx)} samples in {len(blocks)} blocks")

    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), newline="\n")
    (out_dir / "split_report.md").write_text(render_markdown(report, CLASS_NAMES.get(d["sequence"])), newline="\n")
    print(f"wrote {out_dir}")


def render_markdown(rep: dict, class_names) -> str:
    lines = [f"# Split report: {rep['sequence']}, robots {', '.join(rep['robots'])}", ""]
    if rep["window_s"]:
        lines.append(f"Window: {rep['window_s']} s while the drone is airborne "
                     f"({rep['complete_samples_30hz']} complete 30 Hz frames, {rep['in_buffers_30hz']} in buffers).")
    lines.append(f"Dropped frames: {rep['dropped']}")
    lines += ["", "| Split | Samples | Rate | Seconds | Blocks (s from window start) | Unlabeled |",
              "|---|---|---|---|---|---|"]
    for name, sp in rep["splits"].items():
        blocks = ", ".join(f"{a:.0f}-{b:.0f}" for a, b in sp["blocks_s"])
        lines.append(f"| {name} | {sp['samples']} | {sp['rate_hz']} | {sp['seconds_covered']} | {blocks} | "
                     f"{sp['unlabeled_share'] * 100:.1f}% |")
    names = class_names or [str(i) for i in range(len(next(iter(rep["splits"].values()))["class_share"]))]
    lines += ["", "Class pixel share (% of all pixels):", "",
              "| Class | " + " | ".join(rep["splits"]) + " |", "|---" * (len(rep["splits"]) + 1) + "|"]
    for c, cname in enumerate(names):
        cells = " | ".join(f"{sp['class_share'][c] * 100:.2f}" for sp in rep["splits"].values())
        lines.append(f"| {c} {cname} | {cells} |")
    lines += ["", "Pairing offset to the ego frame (ms):", "", "| Stream | Median | p95 | Max |", "|---|---|---|---|"]
    for k, v in rep["pairing_offset_ms"].items():
        lines.append(f"| {k} | {v['median']} | {v['p95']} | {v['max']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
