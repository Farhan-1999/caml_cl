"""Block-interleaved temporal splits.

Consecutive 30 Hz frames are near-duplicates, so a random frame split would leak test frames into
training. Instead the timeline is cut into periods; each period is divided into parts (e.g. 70 s
train, 10 s val, 20 s test) and a buffer around every part boundary is dropped. Every place the
robots visit appears in every split, and no test frame is within buffer_s/2 of a training frame.
"""
from __future__ import annotations

import numpy as np


def assign_splits(times_s: np.ndarray, period_s: float, parts: dict, buffer_s: float) -> np.ndarray:
    """Split name per frame (object array); None for frames inside a buffer.

    parts: ordered {name: seconds per period}; the seconds must add up to period_s.
    """
    names = list(parts)
    lengths = [float(parts[n]) for n in names]
    if abs(sum(lengths) - period_s) > 1e-6:
        raise ValueError(f"split parts {parts} do not add up to period_s={period_s}")
    edges = np.cumsum([0.0, *lengths])
    phase = np.mod(np.asarray(times_s, float), period_s)
    half = buffer_s / 2
    out = np.full(len(phase), None, dtype=object)
    for k, name in enumerate(names):
        out[(phase >= edges[k] + half) & (phase < edges[k + 1] - half)] = name
    return out


def subsample(times_s: np.ndarray, rate_hz: float | None) -> np.ndarray:
    """Mask keeping the first frame of each 1/rate_hz time bucket; None keeps every frame.
    times_s must be sorted."""
    times_s = np.asarray(times_s, float)
    if not rate_hz:
        return np.ones(len(times_s), bool)
    buckets = np.floor(times_s * rate_hz).astype(np.int64)
    keep = np.ones(len(buckets), bool)
    keep[1:] = buckets[1:] != buckets[:-1]
    return keep
