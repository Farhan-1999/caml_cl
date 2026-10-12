"""Time-aligned multi-robot samples from an export made by tools/export_sequence.py.

A sample is anchored on one labelled frame of the ego robot. Each robot contributes its RGB frame
nearest to that anchor (the ego's own RGB frame is the anchor itself) and, for every other modality,
the file paired with that RGB frame: same stamp for modalities made on the RGB stream (labels,
ZoeDepth), nearest stamp within the tolerance for separate sensors (IR, LiDAR).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Exported with the robot's RGB header stamps, so they are looked up by exact stamp.
RGB_STAMPED = frozenset({"rgb", "depth_zoe", "label"})


@dataclass
class ModalityIndex:
    stamps: np.ndarray  # int64 nanoseconds, sorted
    paths: list         # relative to the export root, '/'-separated


def load_index(export_root: Path, sequence: str, robot: str, modality: str) -> ModalityIndex:
    stamps, paths = [], []
    with open(export_root / sequence / robot / f"{modality}.csv", newline="") as f:
        for row in csv.DictReader(f):
            stamps.append(int(row["stamp_ns"]))
            paths.append(f"{sequence}/{robot}/{row['path']}")
    order = np.argsort(np.asarray(stamps, np.int64), kind="stable")
    return ModalityIndex(np.asarray(stamps, np.int64)[order], [paths[i] for i in order])


def exact(stamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of each query stamp in `stamps`, -1 where absent."""
    query = np.asarray(query, np.int64)
    i = np.clip(np.searchsorted(stamps, query), 0, len(stamps) - 1)
    return np.where(stamps[i] == query, i, -1)


def nearest(stamps: np.ndarray, query: np.ndarray, tol_ns: int) -> np.ndarray:
    """Index of the nearest stamp for each query, -1 if farther than tol_ns or the query is -1."""
    query = np.asarray(query, np.int64)
    i = np.searchsorted(stamps, query)
    lo = np.clip(i - 1, 0, len(stamps) - 1)
    hi = np.clip(i, 0, len(stamps) - 1)
    pick = np.where(np.abs(stamps[hi] - query) < np.abs(stamps[lo] - query), hi, lo)
    ok = (np.abs(stamps[pick] - query) <= tol_ns) & (query >= 0)
    return np.where(ok, pick, -1)


def read_odometry(export_root: Path, sequence: str, robot: str) -> tuple[np.ndarray, np.ndarray]:
    """(stamps_ns, xyz) from odom.csv; stamps are read as integers to keep nanosecond precision."""
    stamps, xyz = [], []
    with open(export_root / sequence / robot / "odom.csv", newline="") as f:
        for row in csv.DictReader(f):
            stamps.append(int(row["stamp_ns"]))
            xyz.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return np.asarray(stamps, np.int64), np.asarray(xyz, np.float64)


def flight_window(export_root: Path, sequence: str, robot: str, min_height_m: float,
                  trim_s: float) -> tuple[int, int]:
    """[start, end] ns of a drone's airborne period: height above its starting height exceeds
    min_height_m; trim_s is cut from both ends to drop takeoff and landing."""
    stamps, xyz = read_odometry(export_root, sequence, robot)
    height = xyz[:, 2] - np.median(xyz[:200, 2])  # the drone starts on the ground
    airborne = stamps[height > min_height_m]
    if len(airborne) == 0:
        raise ValueError(f"{robot} never rises above {min_height_m} m")
    trim = int(trim_s * 1e9)
    return int(airborne[0]) + trim, int(airborne[-1]) - trim


def build_samples(export_root: Path, sequence: str, ego: str, collaborators: list[str],
                  modalities: list[str], tol_s: float, window: tuple[int, int] | None = None) -> dict:
    """Columns of all complete samples (every robot has every modality) inside `window`.

    Returns {"columns": {name: list}, "dropped": {...}} where columns hold ego_stamp_ns, label,
    and for each robot r and modality m: "r.m" (path) and "r.m.stamp_ns".
    """
    robots = [ego, *collaborators]
    tol = int(tol_s * 1e9)
    labels = load_index(export_root, sequence, ego, "label")
    anchor = labels.stamps
    keep = np.ones(len(anchor), bool)
    if window is not None:
        keep &= (anchor >= window[0]) & (anchor <= window[1])
    dropped = {"outside_window": int((~keep).sum())}
    columns = {"ego_stamp_ns": anchor, "label": np.asarray(labels.paths, object)}

    for robot in robots:
        rgb = load_index(export_root, sequence, robot, "rgb")
        ref = exact(rgb.stamps, anchor) if robot == ego else nearest(rgb.stamps, anchor, tol)
        missing = keep & (ref < 0)
        dropped[f"{robot}.rgb"] = int(missing.sum())
        keep &= ref >= 0
        ref_stamps = np.where(ref >= 0, rgb.stamps[np.maximum(ref, 0)], -1)
        for modality in modalities:
            mi = rgb if modality == "rgb" else load_index(export_root, sequence, robot, modality)
            j = exact(mi.stamps, ref_stamps) if modality in RGB_STAMPED else nearest(mi.stamps, ref_stamps, tol)
            if modality != "rgb":
                dropped[f"{robot}.{modality}"] = int((keep & (j < 0)).sum())
            keep &= j >= 0
            columns[f"{robot}.{modality}"] = np.asarray([mi.paths[k] if k >= 0 else "" for k in j], object)
            columns[f"{robot}.{modality}.stamp_ns"] = np.where(j >= 0, mi.stamps[np.maximum(j, 0)], -1)

    return {"columns": {k: v[keep] for k, v in columns.items()}, "dropped": dropped}
