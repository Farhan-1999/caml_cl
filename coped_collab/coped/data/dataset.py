"""PyTorch dataset of time-aligned multi-robot samples listed in a split CSV.

Every sample returns all configured modalities of all robots; a model picks the subset it uses, so
teacher and students train on exactly the same samples (and, for distillation, the same noise).

Item layout:
    {"inputs": {robot: {modality: float tensor (3, S, S), ImageNet-normalized}},
     "label": int64 tensor (S, S) of the ego robot, ignore_index where unlabeled,
     "sample_id": str}
Gray modalities (IR, ZoeDepth) are repeated to 3 channels so ImageNet-pretrained encoders load as is.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import resolve_path

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GRAY_MODALITIES = frozenset({"ir", "ir_right", "depth_zoe"})


def read_split(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def split_fingerprint(rows: list[dict]) -> str:
    return hashlib.sha1("\n".join(r["sample_id"] for r in rows).encode()).hexdigest()


def load_image(path: Path, modality: str, size: int) -> np.ndarray:
    """uint8 (S, S, 3) RGB for color modalities, (S, S) for gray ones; squashed to S x S like CAML."""
    if modality in GRAY_MODALITIES:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_label(path: Path, size: int) -> np.ndarray:
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    return cv2.resize(lab, (size, size), interpolation=cv2.INTER_NEAREST)


def to_unit_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW or HxWx3 -> float (3, H, W) in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
    return x.unsqueeze(0).expand(3, -1, -1).clone() if x.ndim == 2 else x.permute(2, 0, 1).contiguous()


class CollabSegDataset(Dataset):
    def __init__(self, rows: list[dict], export_root: Path, robots: list[str], modalities: list[str],
                 image_size: int, noise: dict | None = None, deterministic_noise: bool = False,
                 cache: Path | None = None):
        self.rows = rows
        self.export_root = export_root
        self.robots = list(robots)
        self.modalities = list(modalities)
        self.size = image_size
        self.noise = noise if noise and noise.get("enabled", True) else None
        self.deterministic_noise = deterministic_noise
        self.cache = self._open_cache(cache) if cache else None

    def __len__(self) -> int:
        return len(self.rows)

    def _open_cache(self, cache_dir: Path) -> dict:
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no cache at {cache_dir}; run tools/build_cache.py")
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] != split_fingerprint(self.rows) or meta["image_size"] != self.size:
            raise ValueError(f"cache {cache_dir} was built for other samples or image size; rebuild it")
        keys = [f"{r}.{m}" for r in self.robots for m in self.modalities] + ["label"]
        return {k: np.load(cache_dir / f"{k}.npy", mmap_mode="r") for k in keys}

    def raw(self, i: int) -> dict:
        """uint8 arrays at S x S, keyed "robot.modality" and "label" (no noise, no normalization)."""
        if self.cache is not None:
            return {k: np.array(v[i]) for k, v in self.cache.items()}  # copy out of the read-only memmap
        row = self.rows[i]
        out = {f"{r}.{m}": load_image(self.export_root / row[f"{r}.{m}"], m, self.size)
               for r in self.robots for m in self.modalities}
        out["label"] = load_label(self.export_root / row["label"], self.size)
        return out

    def _is_noisy(self, robot: str, modality: str) -> bool:
        return self.noise is not None and robot in self.noise["robots"] and modality in self.noise["modalities"]

    def _add_noise(self, x: torch.Tensor, sample_id: str, robot: str, modality: str) -> torch.Tensor:
        sigma = float(self.noise["gaussian_sigma"])
        if self.deterministic_noise:
            # same noise for a given sample on every run and machine
            seed = zlib.crc32(f"{self.noise.get('eval_seed', 0)}/{sample_id}/{robot}/{modality}".encode())
            gen = torch.Generator().manual_seed(seed)
            noise = torch.randn(x.shape, generator=gen)
        else:
            noise = torch.randn(x.shape)  # DataLoader seeds each worker, so this differs per epoch
        if modality in GRAY_MODALITIES:
            noise = noise[:1].expand_as(x)  # gray input: one noise field shared by the 3 copies
        return (x + sigma * noise).clamp_(0.0, 1.0)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        raw = self.raw(i)
        inputs = {}
        for r in self.robots:
            inputs[r] = {}
            for m in self.modalities:
                x = to_unit_tensor(raw[f"{r}.{m}"])
                if self._is_noisy(r, m):
                    # noise is added at network resolution, as in CAML's Fig. 5 inputs
                    x = self._add_noise(x, row["sample_id"], r, m)
                inputs[r][m] = (x - IMAGENET_MEAN) / IMAGENET_STD
        label = torch.from_numpy(raw["label"].astype(np.int64))
        return {"inputs": inputs, "label": label, "sample_id": row["sample_id"]}


def build_dataset(cfg: dict, split: str, deterministic_noise: bool | None = None,
                  use_cache: bool = True, noise: bool = True) -> CollabSegDataset:
    """Dataset for one split of a config. Val/test noise is deterministic unless told otherwise."""
    d = cfg["data"]
    rows = read_split(resolve_path(d["splits_dir"]) / f"{split}.csv")
    cache = resolve_path(d["cache_dir"]) / split if use_cache and d.get("cache_dir") else None
    return CollabSegDataset(
        rows=rows,
        export_root=resolve_path(d["export_root"]),
        robots=[d["ego"], *d["collaborators"]],
        modalities=d["modalities"],
        image_size=d["image_size"],
        noise=d.get("noise") if noise else None,
        deterministic_noise=(split != "train") if deterministic_noise is None else deterministic_noise,
        cache=cache,
    )
