"""Experiment manifests (e.g. configs/caml/experiments.yaml): experiment ids, their configs and seeds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import load_config


@dataclass
class PlannedConfig:
    experiment: str          # e.g. "E3"
    path: Path               # config file
    cfg: dict                # loaded config, overrides applied
    optional: bool

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def is_student(self) -> bool:
        return bool((self.cfg.get("distill") or {}).get("teacher_run"))


def load_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = yaml.safe_load(path.read_text())
    manifest["path"] = path
    return manifest


def planned_configs(manifest: dict, overrides=(), only=(), include_optional: bool = False) -> list[PlannedConfig]:
    """Configs of the manifest in listed order, each once; `only` keeps the given experiment ids."""
    out, seen = [], set()
    for exp in manifest["experiments"]:
        if only and exp["id"] not in only:
            continue
        if exp.get("optional") and not include_optional and exp["id"] not in only:
            continue
        for rel in exp.get("configs") or []:
            path = (manifest["path"].parent / rel).resolve()
            if path in seen:
                continue
            seen.add(path)
            out.append(PlannedConfig(exp["id"], path, load_config(path, overrides), bool(exp.get("optional"))))
    return out


def training_order(configs: list[PlannedConfig]) -> list[PlannedConfig]:
    """Runs without a teacher first, then students, each group in manifest order."""
    return [c for c in configs if not c.is_student] + [c for c in configs if c.is_student]
