"""Training and evaluation of CAML teachers and students (CAML App. A.4-A.5).

A run lives in <train.out_dir>/<name>/seed<k>/ and holds:
    config.yaml       the resolved config the run used
    log.csv           one row per epoch: lr, losses, val mIoU (full inputs), epoch time
    subset_eval.csv   one row per eval epoch, split and input combination: which modalities each robot
                      sent (e.g. ptah=rgb+ir, race1=none) and the resulting mIoU
    last.pt, best.pt  checkpoints; best = highest val mIoU with full inputs
    metrics.json      for best and last: val/test with full inputs, test with the collaborators
                      removed, and val/test for every input combination ("subsets")
    model_info.json   parameter counts, per-robot message sizes, mean epoch time

Input combinations: every robot can send any subset of the model's modalities, including nothing;
eval.subsets picks which combinations are evaluated (by default the few in eval.selected_subsets).
Missing features are zeros (see CAML.server_forward), so a model trained on full inputs shows how
it degrades when robots or sensors are absent; it was never trained on such inputs.

A student is a run whose config sets distill.teacher_run: the frozen teacher sees the same batch
(all its modalities, the same noise) and the student is trained with CAML's distillation loss.
"""
from __future__ import annotations

import contextlib
import csv
import itertools
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .config import resolve_path
from .data.dataset import build_dataset
from .losses import distillation_loss, task_loss
from .metrics import ConfusionMatrix
from .models import build_model, parameter_counts

LOG_FIELDS = ["epoch", "lr", "train_loss", "train_ce", "train_kd", "val_miou", "val_miou_common", "epoch_time_s"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(inputs: dict, device: torch.device) -> dict:
    return {r: {m: x.to(device, non_blocking=True) for m, x in mods.items()} for r, mods in inputs.items()}


def autocast_for(device: torch.device, enabled: bool):
    """bf16 autocast on GPUs that support it (no loss scaling needed), fp16 otherwise; off on CPU."""
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return (lambda: torch.autocast("cuda", dtype=dtype)), dtype


def run_suffix(smoke: bool = False, overfit: int = 0) -> str:
    """Smoke and overfit runs go to <name>_smoke / <name>_overfit<N> so they never mix with real runs."""
    return "_smoke" if smoke else (f"_overfit{overfit}" if overfit else "")


def run_dir_for(cfg: dict, seed: int, suffix: str = "") -> Path:
    return resolve_path(cfg["train"]["out_dir"]) / f"{cfg['name']}{suffix}" / f"seed{seed}"


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, seed: int, device: torch.device,
                drop_last: bool = False) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=gen,
                      pin_memory=device.type == "cuda", drop_last=drop_last,
                      persistent_workers=workers > 0)


def load_run_model(run_dir: Path, checkpoint: str, device: torch.device):
    """(model, config) of a finished or running run, weights from best.pt or last.pt."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    state = torch.load(run_dir / f"{checkpoint}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def teacher_run_dir(cfg: dict, seed: int, suffix: str = "") -> Path:
    """Teacher run of a student config for this seed. Smoke/overfit students (suffix "_smoke",
    "_overfit8") use the teacher trained in the same mode: <out>/<name><suffix>/seed<k>."""
    run = resolve_path(str(cfg["distill"]["teacher_run"]).format(seed=seed))
    return run.parent.with_name(run.parent.name + suffix) / run.name if suffix else run


def load_teacher(cfg: dict, seed: int, device: torch.device, suffix: str = ""):
    run = teacher_run_dir(cfg, seed, suffix)
    if not (run / "best.pt").exists():
        raise FileNotFoundError(f"teacher checkpoint {run / 'best.pt'} not found; train the teacher first")
    teacher, tcfg = load_run_model(run, "best", device)
    for key in ("ego", "collaborators", "num_classes", "image_size", "splits_dir"):
        if tcfg["data"][key] != cfg["data"][key]:
            raise ValueError(f"teacher data.{key}={tcfg['data'][key]!r} differs from student {cfg['data'][key]!r}")
    teacher.requires_grad_(False)
    return teacher, run


@torch.no_grad()
def evaluate(model, loader, device: torch.device, cfg: dict, drop_robots=(), amp: bool = True) -> dict:
    d = cfg["data"]
    autocast, _ = autocast_for(device, amp)
    model.eval()
    cm = ConfusionMatrix(d["num_classes"], d["ignore_index"], device)
    for batch in loader:
        with autocast():
            logits = model(to_device(batch["inputs"], device), drop_robots=drop_robots)
        cm.update(logits.argmax(1), batch["label"].to(device, non_blocking=True))
    return cm.compute(cfg.get("eval", {}).get("rare_classes", ()))


def modality_combinations(robots: list[str], modalities: list[str], mode: str = "selected",
                          selected: list[dict] | None = None) -> list[dict]:
    """What each robot sends, for every evaluated case; the first entry is always full inputs.

    "selected" the cases listed in eval.selected_subsets ({robot: [modalities]}), each restricted to
               this model's modalities; cases that become empty or repeat an earlier one are dropped
    "robots"   all robots together, then each robot alone, always with all modalities
    "all"      every robot sends any subset of the modalities (2^M choices per robot), except the
               case where nobody sends anything: 63 cases for 2 robots and 3 modalities
    "full"     only all robots with all modalities
    """
    full = {r: list(modalities) for r in robots}
    alone = [{r: list(modalities) if r == q else [] for r in robots} for q in robots]
    if mode == "full":
        return [full]
    if mode == "robots":
        return [full, *alone]
    if mode == "selected":
        if not selected:
            raise ValueError("eval.subsets is 'selected' but eval.selected_subsets is empty")
        combos = [full] + [{r: [m for m in modalities if m in (spec.get(r) or [])] for r in robots}
                           for spec in selected]
    elif mode == "all":
        subsets = [list(c) for k in range(len(modalities), -1, -1) for c in itertools.combinations(modalities, k)]
        combos = [dict(zip(robots, choice)) for choice in itertools.product(subsets, repeat=len(robots))]
    else:
        raise ValueError(f"eval.subsets must be selected, robots, all or full, got {mode!r}")
    out, seen = [], set()
    for c in combos:
        key = combo_key(c)
        if any(c.values()) and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def eval_combinations(cfg: dict, robots: list[str], modalities: list[str], final: bool = False) -> list[dict]:
    """Input combinations evaluated during training (eval.subsets); for the final evaluation each
    robot alone is always added, so the without-collaborators result is available."""
    ev = cfg.get("eval") or {}
    combos = modality_combinations(robots, modalities, ev.get("subsets", "selected"), ev.get("selected_subsets"))
    if final:
        keys = {combo_key(c) for c in combos}
        combos += [c for c in modality_combinations(robots, modalities, "robots") if combo_key(c) not in keys]
    return combos


def combo_key(combo: dict) -> str:
    """e.g. "ptah=rgb+ir race1=none"."""
    return " ".join(f"{r}={'+'.join(mods) if mods else 'none'}" for r, mods in combo.items())


def alone_key(robots: list[str], modalities: list[str], robot: str) -> str:
    return combo_key({r: list(modalities) if r == robot else [] for r in robots})


@torch.no_grad()
def evaluate_subsets(model, loader, device: torch.device, cfg: dict, combos: list[dict], amp: bool = True) -> dict:
    """Metrics per input combination, keyed by combo_key. The encoders run once per batch; only
    fusion and decoding are repeated for each combination."""
    d = cfg["data"]
    autocast, _ = autocast_for(device, amp)
    model.eval()
    cms = [ConfusionMatrix(d["num_classes"], d["ignore_index"], device) for _ in combos]
    for batch in loader:
        inputs = to_device(batch["inputs"], device)
        label = batch["label"].to(device, non_blocking=True)
        with autocast():
            feats = model.encode(inputs)
            for cm, combo in zip(cms, combos):
                cm.update(model.forward_available(feats, combo, label.shape[-2:]).argmax(1), label)
    rare = cfg.get("eval", {}).get("rare_classes", ())
    return {combo_key(c): cm.compute(rare) for c, cm in zip(combos, cms)}


def append_subset_rows(path: Path, epoch: int, per_split: dict, combos: list[dict], robots: list[str]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for split, results in per_split.items():
            for combo in combos:
                key = combo_key(combo)
                if key not in results:
                    continue
                r = results[key]
                writer.writerow([epoch, split, *("+".join(combo[x]) or "none" for x in robots),
                                 sum(len(combo[x]) for x in robots),
                                 f"{r['miou']:.5f}", f"{r['miou_common']:.5f}", f"{r['pixel_accuracy']:.5f}"])


def message_sizes(model, cfg: dict, device: torch.device) -> dict:
    """Shape and size of every map a collaborator transmits for one frame (the ego keeps its own)."""
    d = cfg["data"]
    s = d["image_size"]
    model.eval()
    out = {}
    for robot in model.robots:
        if robot == d["ego"]:
            continue
        with torch.no_grad():
            msgs = model.robot_forward({m: torch.zeros(1, 3, s, s, device=device) for m in model.modalities})
        for key, t in msgs.items():
            n = t[0].numel()
            out[f"{robot}.{key}"] = {"shape": list(t.shape[1:]), "bytes_fp32": n * 4, "bytes_fp16": n * 2}
    return out


def train(cfg: dict, seed: int, device: torch.device, resume: bool = False, overwrite: bool = False,
          smoke: bool = False, overfit: int = 0) -> Path:
    """Train one run; returns its directory. `smoke` runs 2 epochs of 2 steps on 4-sample eval sets;
    `overfit` trains and evaluates on the first N training samples (a learning sanity check)."""
    tr, d = cfg["train"], cfg["data"]
    suffix = run_suffix(smoke, overfit)
    run_dir = run_dir_for(cfg, seed, suffix)
    if (run_dir / "metrics.json").exists() and resume:
        print(f"{run_dir} already finished, skipping")
        return run_dir
    if run_dir.exists() and not resume:
        if not overwrite:
            raise SystemExit(f"{run_dir} exists; pass --resume to continue it or --overwrite to restart")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    epochs, batch_size, workers = tr["epochs"], tr["batch_size"], tr.get("num_workers", 8)
    max_steps, eval_limit = None, None
    if smoke:
        epochs, batch_size, workers, max_steps, eval_limit = 2, 2, 0, 2, 4
    train_ds = build_dataset(cfg, "train")
    val_ds, test_ds = build_dataset(cfg, "val"), build_dataset(cfg, "test")
    if overfit:
        keep = list(range(overfit))
        train_ds = Subset(train_ds, keep)
        val_ds = test_ds = Subset(build_dataset(cfg, "train", deterministic_noise=True), keep)
        batch_size = min(batch_size, overfit)
    if eval_limit:
        val_ds, test_ds = Subset(val_ds, range(eval_limit)), Subset(test_ds, range(eval_limit))
    train_loader = make_loader(train_ds, batch_size, True, workers, seed, device, drop_last=len(train_ds) > batch_size)
    val_loader = make_loader(val_ds, batch_size, False, workers, seed, device)
    test_loader = make_loader(test_ds, batch_size, False, workers, seed, device)

    model = build_model(cfg).to(device)
    teacher, teacher_run = (None, None)
    dist = cfg.get("distill") or {}
    if dist.get("teacher_run"):
        teacher, teacher_run = load_teacher(cfg, seed, device, suffix)
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    autocast, amp_dtype = autocast_for(device, tr.get("amp", True))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    start_epoch, best, best_epoch = 0, -1.0, -1
    if resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
        torch.set_rng_state(state["rng_cpu"])
        print(f"resumed {run_dir} at epoch {start_epoch}")

    snapshot = {**cfg, "run": {"seed": seed, "smoke": smoke, "overfit": overfit, "device": str(device),
                               "amp_dtype": str(amp_dtype), "teacher_run": str(teacher_run) if teacher_run else None,
                               "train_samples": len(train_ds), "val_samples": len(val_ds),
                               "test_samples": len(test_ds)}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
    log_path = run_dir / "log.csv"
    if not (resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    eval_every = tr.get("eval_every", 1)
    ignore = d["ignore_index"]
    ev = cfg.get("eval") or {}
    combos = eval_combinations(cfg, model.robots, model.modalities)
    full_key = combo_key(combos[0])
    alone = {r: alone_key(model.robots, model.modalities, r) for r in model.robots}
    subset_splits = ev.get("subset_splits", ["val", "test"])
    eval_loaders = {"val": val_loader, "test": test_loader}
    subset_log = run_dir / "subset_eval.csv"
    if not (resume and subset_log.exists()):
        with open(subset_log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "split", *model.robots, "inputs_sent", "miou", "miou_common",
                                    "pixel_accuracy"])
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        sums, steps = {"loss": 0.0, "ce": 0.0, "kd": 0.0}, 0
        for step, batch in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            inputs = to_device(batch["inputs"], device)
            label = batch["label"].to(device, non_blocking=True)
            with autocast():
                logits = model(inputs)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(inputs)
            if teacher is not None:
                losses = distillation_loss(logits, teacher_logits, label, dist["alpha"], dist["temperature"], ignore)
            else:
                ce = task_loss(logits, label, ignore)
                losses = {"loss": ce, "ce": ce.detach(), "kd": torch.zeros(())}
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in sums:
                sums[k] += float(losses[k].detach())
            steps += 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - t0

        val, per_split = None, {}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            for split, loader in eval_loaders.items():
                if split in subset_splits:
                    per_split[split] = evaluate_subsets(model, loader, device, cfg, combos, tr.get("amp", True))
                elif split == "val":  # model selection always needs val with full inputs
                    per_split[split] = evaluate_subsets(model, loader, device, cfg, combos[:1], tr.get("amp", True))
            append_subset_rows(subset_log, epoch, per_split, combos, model.robots)
            val = per_split["val"][full_key]
            if val["miou"] > best:  # selection: val, full inputs only; test never influences training
                best, best_epoch = val["miou"], epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best, "best_epoch": best_epoch, "rng_cpu": torch.get_rng_state()},
                   run_dir / "last.pt")
        row = [epoch, f"{lr:.6g}", *(f"{sums[k] / max(1, steps):.5f}" for k in ("loss", "ce", "kd")),
               f"{val['miou']:.5f}" if val else "", f"{val['miou_common']:.5f}" if val else "", f"{epoch_time:.2f}"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        msg = f"epoch {epoch + 1}/{epochs} loss {sums['loss'] / max(1, steps):.4f}"
        if val:
            msg += f" val mIoU {val['miou'] * 100:.2f} (best {best * 100:.2f} @ {best_epoch + 1})"
            solo = [f"{r} {per_split['val'][k]['miou'] * 100:.1f}" for r, k in alone.items() if k in per_split["val"]]
            if solo:
                msg += " | alone: " + ", ".join(solo)
        print(f"{msg} [{epoch_time:.1f}s]", flush=True)

    finalize(run_dir, model, cfg, device, val_loader, test_loader, best_epoch)
    return run_dir


def finalize(run_dir: Path, model, cfg: dict, device: torch.device, val_loader, test_loader, best_epoch: int) -> None:
    amp = cfg["train"].get("amp", True)
    ego = cfg["data"]["ego"]
    combos = eval_combinations(cfg, model.robots, model.modalities, final=True)
    full_key = combo_key(combos[0])
    alone = {r: alone_key(model.robots, model.modalities, r) for r in model.robots}
    results = {"best_epoch": best_epoch, "full_inputs": full_key, "alone": alone}
    for ckpt in ("best", "last"):
        model.load_state_dict(torch.load(run_dir / f"{ckpt}.pt", map_location=device, weights_only=False)["model"])
        subsets = {split: evaluate_subsets(model, loader, device, cfg, combos, amp)
                   for split, loader in (("val", val_loader), ("test", test_loader))}
        results[ckpt] = {"val": subsets["val"][full_key], "test": subsets["test"][full_key],
                         "test_without_collaborators": subsets["test"][alone[ego]], "subsets": subsets}
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    times = []
    with open(run_dir / "log.csv", newline="") as f:
        times = [float(r["epoch_time_s"]) for r in csv.DictReader(f)]
    info = {"parameters": parameter_counts(model), "messages_per_frame": message_sizes(model, cfg, device),
            "mean_epoch_time_s": round(float(np.mean(times)), 2) if times else None}
    (run_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    b = results["best"]
    solo = ", ".join(f"{r} alone {b['subsets']['test'][k]['miou'] * 100:.2f}" for r, k in alone.items())
    print(f"done: best epoch {best_epoch + 1}, val mIoU {b['val']['miou'] * 100:.2f}, "
          f"test mIoU {b['test']['miou'] * 100:.2f} ({solo}; {len(combos)} input combinations) -> {run_dir}")
