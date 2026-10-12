"""Experiment configs: YAML files, optional `base:` inheritance and dotted command-line overrides.

Relative paths in a config resolve against the project root (the coped_collab folder), so a config
works on any machine that keeps the layout `<workspace>/coped_collab` + `<workspace>/exported`.
When the data lives elsewhere, override it: --set data.export_root=/abs/path/to/exported
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, overrides: list[str] | tuple = ()) -> dict:
    """Load a YAML config. A top-level `base: other.yaml` (relative to this file) is loaded first
    and deep-merged under it. Each override is `dotted.key=value`, the value parsed as YAML."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(value))
    cfg["config_path"] = str(path)
    return cfg


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_by_path(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def resolve_path(path: str | Path) -> Path:
    """Expand ~ and $VARS; relative paths are taken from the project root."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
