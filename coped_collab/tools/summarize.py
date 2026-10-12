"""Summarize the experiment suite into a results report: tables for every CAML comparison, figures
and key comparisons, as mean +- std over seeds.

Reads only the run folders (metrics.json, model_info.json, log.csv, subset_eval.csv), so it needs
no GPU and no data. Writes to <train.out_dir>/_report/ (or <...>/_report_smoke with --smoke):
    results.md     the report; every figure also appears as a table
    results.json   every aggregated number
    runs.csv       one row per run, checkpoint, split and input combination
    fig_*.png      learning curves, the teacher's input combinations over training, accuracy vs data sent

Examples:
    python tools/summarize.py --manifest configs/caml/experiments.yaml --optional
    python tools/summarize.py --manifest configs/caml/experiments.yaml --optional --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coped.config import resolve_path  # noqa: E402
from coped.data.classes import CLASS_NAMES  # noqa: E402
from coped.engine import combo_key, run_dir_for, run_suffix  # noqa: E402
from coped.experiments import load_manifest, planned_configs  # noqa: E402

# Reference palette of the dataviz guide (light mode), slots in their validated order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# CAML's reported indoor (NYUARPL) mIoU, Table 6 and Sec. 4.2; different splits, robots, labels and noise.
CAML_REPORTED = {"FCN (ground robot, RGB)": 51.2, "AML (ground robot, RGB+D -> RGB)": 55.9,
                 "CAML (aerial + ground, RGB+D -> RGB)": 60.1, "Pre-fusion CAML": 59.2}


# ----------------------------------------------------------------------------- loading
class Model:
    """All seeds of one config."""

    def __init__(self, planned, seeds: list[int], suffix: str, checkpoint: str):
        self.exp, self.name, self.cfg = planned.experiment, planned.name, planned.cfg
        d = self.cfg["data"]
        self.robots = self.cfg["model"].get("robots") or [d["ego"], *d["collaborators"]]
        self.ego, self.collaborators = d["ego"], d["collaborators"]
        self.modalities = self.cfg["model"]["modalities"]
        self.fusion = self.cfg["model"]["fusion"]
        teacher = (self.cfg.get("distill") or {}).get("teacher_run")
        self.teacher = Path(str(teacher)).parent.name if teacher else None
        self.checkpoint = checkpoint
        self.runs = {}
        for seed in seeds:
            rd = run_dir_for(self.cfg, seed, suffix)
            if not (rd / "metrics.json").exists():
                continue
            run = {"dir": rd, "metrics": json.loads((rd / "metrics.json").read_text())}
            info = rd / "model_info.json"
            run["info"] = json.loads(info.read_text()) if info.exists() else {}
            run["log"] = read_csv(rd / "log.csv")
            run["subset"] = read_csv(rd / "subset_eval.csv")
            self.runs[seed] = run

    @property
    def label(self) -> str:
        """e.g. "student RGB", "pre-fusion teacher RGB+D+IR", "student, no KD RGB"."""
        if self.teacher:
            role = "student"
        elif "teacher" in self.name:
            role = "teacher"
        else:
            role = "student, no KD"
        return f"{'pre-fusion ' if self.fusion == 'pre' else ''}{role} {short_mods(self.modalities)}"

    @property
    def tag(self) -> str:
        """Label with the experiment id, unique across the suite (E2 and E5 both have an RGB student)."""
        return f"{self.exp} {self.label}"

    def ckpt(self, seed: int) -> dict:
        return self.runs[seed]["metrics"][self.checkpoint]

    def per_seed(self, getter) -> dict[int, float]:
        out = {}
        for seed in self.runs:
            try:
                v = getter(seed)
            except (KeyError, TypeError, IndexError):
                v = None
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[seed] = float(v)
        return out

    def test(self, seed):
        return self.ckpt(seed)["test"]["miou"]

    def test_common(self, seed):
        return self.ckpt(seed)["test"]["miou_common"]

    def val(self, seed):
        return self.ckpt(seed)["val"]["miou"]

    def alone(self, robot: str):
        return lambda seed: self.ckpt(seed)["subsets"]["test"][self.runs[seed]["metrics"]["alone"][robot]]["miou"]

    def combo(self, key: str):
        return lambda seed: self.ckpt(seed)["subsets"]["test"][key]["miou"]

    def class_iou(self, c: int):
        return lambda seed: self.ckpt(seed)["test"]["iou"][c]

    def info(self, *path):
        for run in self.runs.values():
            node = run["info"]
            try:
                for p in path:
                    node = node[p]
                return node
            except (KeyError, TypeError):
                continue
        return None

    def kib_sent(self, combo: dict) -> float | None:
        """KiB (fp16) the collaborators transmit per frame when they send `combo`."""
        msgs = self.info("messages_per_frame")
        if msgs is None:
            return None
        total = 0
        for r in self.collaborators:
            sent = combo.get(r) or []
            if not sent:
                continue
            if self.fusion == "pre":
                total += msgs.get(f"{r}.fused", {}).get("bytes_fp16", 0)
            else:
                total += sum(msgs.get(f"{r}.{m}", {}).get("bytes_fp16", 0) for m in sent)
        return total / 1024


SHORT = {"rgb": "RGB", "depth_zoe": "D", "ir": "IR"}


def short_mods(modalities) -> str:
    return "+".join(SHORT.get(m, m) for m in modalities)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- statistics
def stats(values) -> dict | None:
    vals = [v for v in (values.values() if isinstance(values, dict) else values) if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": len(arr)}


def paired(a: dict[int, float], b: dict[int, float]) -> dict | None:
    """a - b over the seeds both have (a student of seed k pairs with its own teacher)."""
    return stats([a[s] - b[s] for s in a if s in b])


def fmt(st: dict | None, n_expected: int, scale: float = 100, digits: int = 1, sign: bool = False) -> str:
    if st is None:
        return "—"
    m = st["mean"] * scale
    text = f"{m:+.{digits}f}" if sign else f"{m:.{digits}f}"
    if st["n"] > 1:
        text += f" ± {st['std'] * scale:.{digits}f}"
    if st["n"] < n_expected:
        text += f" (n={st['n']})"
    return text


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


# ----------------------------------------------------------------------------- input combinations
def patterns(cfg: dict) -> list[dict]:
    """The selected input combinations of the config, with a label each."""
    out = []
    for i, spec in enumerate((cfg.get("eval") or {}).get("selected_subsets") or [], 1):
        out.append({"label": spec.get("label") or f"combination {i}",
                    "spec": {k: v for k, v in spec.items() if k != "label"}})
    return out


def restricted(model: Model, spec: dict) -> dict:
    return {r: [m for m in model.modalities if m in (spec.get(r) or [])] for r in model.robots}


# ----------------------------------------------------------------------------- figures
def integer_epochs(ax) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 1.6, "figure.dpi": 150,
    })
    return plt


def mean_curve(model: Model, rows_of, value: str):
    """Mean over seeds of a per-epoch value; returns epochs (1-based) and values in %."""
    per_epoch = {}
    for run in model.runs.values():
        for row in rows_of(run):
            if row.get(value):
                per_epoch.setdefault(int(row["epoch"]) + 1, []).append(float(row[value]) * 100)
    epochs = sorted(per_epoch)
    return epochs, [float(np.mean(per_epoch[e])) for e in epochs]


def end_labels(ax, series: list[tuple[str, list, list, str]]):
    """Direct labels at the line ends, nudged apart so they do not collide; only for up to four
    series (beyond that the legend alone carries identity)."""
    if len(series) > 4:
        return
    ends = sorted(((ys[-1], name, color) for name, xs, ys, color in series if ys), key=lambda t: t[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = []
    for y, name, color in ends:
        y_text = max([y] + [p + gap for p in placed[-1:]])
        placed.append(y_text)
        ax.annotate(name, xy=(1.0, y_text), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)


def fig_learning_curves(plt, models: list[Model], out: Path) -> Path | None:
    series = []
    for i, m in enumerate(models[:4]):  # color follows the model's slot, whether or not others exist
        xs, ys = mean_curve(m, lambda run: run["log"], "val_miou")
        if xs:
            series.append((m.tag, xs, ys, SERIES[i]))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, xs, ys, color in series:
        ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel("epoch")
    integer_epochs(ax)
    ax.set_ylabel("val mIoU (%)")
    ax.set_title("Validation mIoU with full inputs (mean over seeds)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)  # below the plot: never covers data
    end_labels(ax, series)
    fig.tight_layout()
    path = out / "fig_learning_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_teacher_combinations(plt, teacher: Model, pats: list[dict], out: Path) -> Path | None:
    series = []
    for i, p in enumerate(pats[:5]):
        combo = restricted(teacher, p["spec"])
        key_cols = {r: "+".join(combo[r]) or "none" for r in teacher.robots}
        xs, ys = mean_curve(teacher, lambda run, kc=key_cols: [
            r for r in run["subset"] if r["split"] == "test" and all(r[k] == v for k, v in kc.items())], "miou")
        if xs:
                series.append((f"#{i + 1} {p['label']}", xs, ys, SERIES[i]))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, xs, ys, color in series:
        ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel("epoch")
    integer_epochs(ax)
    ax.set_ylabel("test mIoU (%)")
    ax.set_title(f"{teacher.tag}: test mIoU per input combination over training", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)  # below the plot: never covers data
    end_labels(ax, series)
    fig.tight_layout()
    path = out / "fig_teacher_combinations.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_tradeoff(plt, points: list[dict], out: Path) -> Path | None:
    """Accuracy against the data the collaborator sends; three series at most (scatter rule)."""
    colors = {"teacher, input combinations": SERIES[0], "students, full inputs": SERIES[1],
              "other models, full inputs": SERIES[2]}
    dodge = {"teacher, input combinations": -3.0, "students, full inputs": 0.0, "other models, full inputs": 3.0}
    groups = [g for g in colors if any(p["group"] == g for p in points)]
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for g in groups:
        pts = [p for p in points if p["group"] == g]
        xs = [p["kib"] + dodge[g] for p in pts]  # many models send the same amount; shift groups apart
        ys = [p["mean"] * 100 for p in pts]
        es = [p["std"] * 100 for p in pts]
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=colors[g], ms=6, mec=SURFACE, mew=1.5,
                    elinewidth=0.9, capsize=0, label=g, zorder=3)
        for p, x, y in zip(pts, xs, ys):
            if p.get("label_point"):  # selective labels; every point is in the table under the figure
                ax.annotate(p["tag"], (x, y), xytext=(-6 if g.startswith("teacher") else 6, 0),
                            textcoords="offset points", fontsize=7.5, color=INK2, va="center",
                            ha="right" if g.startswith("teacher") else "left")
    ax.set_xlabel("data race1 sends per frame (KiB, fp16)")
    ax.set_ylabel("test mIoU (%)")
    ax.set_title("Accuracy against communication", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)  # below the plot: never covers data
    fig.tight_layout()
    path = out / "fig_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- report
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--optional", action="store_true", help="include experiments marked optional")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--smoke", action="store_true", help="summarize the <name>_smoke runs (to test this script)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    seeds = manifest["seeds"]
    suffix = run_suffix(smoke=args.smoke)
    plan = planned_configs(manifest, args.set, include_optional=args.optional)
    models = [Model(p, seeds, suffix, args.checkpoint) for p in plan]
    by_name = {m.name: m for m in models}
    n = len(seeds)
    found = sum(len(m.runs) for m in models)
    if not found:
        raise SystemExit(f"no finished runs found{' (smoke)' if args.smoke else ''}; train first")
    out = resolve_path(plan[0].cfg["train"]["out_dir"]) / ("_report_smoke" if args.smoke else "_report")
    out.mkdir(parents=True, exist_ok=True)
    cfg0 = plan[0].cfg
    d0 = cfg0["data"]
    ego, collab = d0["ego"], d0["collaborators"]
    pats = patterns(cfg0)
    class_names = CLASS_NAMES.get(d0["sequence"], [str(c) for c in range(d0["num_classes"])])

    def get(name):
        return by_name.get(name) if name in by_name and by_name[name].runs else None

    def teacher_tag(m: Model) -> str:
        if not m.teacher:
            return "—"
        return by_name[m.teacher].tag if m.teacher in by_name else m.teacher

    teacher = get("caml_teacher_rgb_depth_ir")
    student = get("caml_student_rgb")
    student_rgbd = get("caml_student_rgb_from_rgb_depth")
    teacher_rgbd = get("caml_teacher_rgb_depth")
    nokd = get("caml_student_rgb_nokd")
    pre_t, pre_s = get("caml_prefusion_teacher_rgb_depth_ir"), get("caml_prefusion_student_rgb")

    L = [f"# CAML reproduction on CoPeD NYUARPL ({ego} + {', '.join(collab)})", ""]
    L.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from {found} finished runs "
             f"({len(models)} models x {n} seeds expected); `{args.checkpoint}` checkpoints"
             f"{' — SMOKE RUNS, numbers are meaningless' if args.smoke else ''}.")
    L.append("Numbers are test mIoU in %, mean ± std over seeds, with every robot sending all of the model's "
             "modalities unless a column says otherwise. (n=k) marks cells with fewer seeds than planned.")
    L.append("")
    missing = [f"{m.name} seed {s}" for m in models for s in seeds if s not in m.runs]
    if missing:
        L += [f"**Missing runs ({len(missing)}):** " + ", ".join(missing), ""]
    results = {"generated": str(datetime.datetime.now()), "checkpoint": args.checkpoint, "smoke": args.smoke,
               "models": {}}

    # 1. all models
    L += ["## 1. All models", ""]
    rows = []
    for m in models:
        full = {r: list(m.modalities) for r in m.robots}
        rec = {"experiment": m.exp, "label": m.label, "fusion": m.fusion, "modalities": m.modalities,
               "teacher": m.teacher, "seeds": sorted(m.runs),
               "test": stats(m.per_seed(m.test)), "test_common": stats(m.per_seed(m.test_common)),
               "val": stats(m.per_seed(m.val)),
               "alone": {r: stats(m.per_seed(m.alone(r))) for r in m.robots},
               "best_epoch": stats(m.per_seed(lambda s: m.runs[s]["metrics"]["best_epoch"] + 1)),
               "parameters_m": (m.info("parameters", "total") or 0) / 1e6 or None,
               "kib_sent_full": m.kib_sent(full), "epoch_s": m.info("mean_epoch_time_s")}
        results["models"][m.name] = rec
        rows.append([m.exp, m.label, teacher_tag(m), fmt(rec["test"], n), fmt(rec["test_common"], n),
                     fmt(rec["val"], n), *(fmt(rec["alone"][r], n) for r in m.robots),
                     fmt(rec["best_epoch"], n, scale=1, digits=0),
                     f"{rec['parameters_m']:.1f}" if rec["parameters_m"] else "—",
                     f"{rec['kib_sent_full']:.0f}" if rec["kib_sent_full"] is not None else "—"])
    L += md_table(["Exp", "Model", "Teacher", "Test mIoU", "Test mIoU-common", "Val mIoU",
                   *(f"{r} alone" for r in models[0].robots), "Best epoch", "Params (M)",
                   f"{', '.join(collab)} sends (KiB/frame)"], rows)
    L += ["mIoU-common leaves out the rare classes " +
          ", ".join(class_names[c] for c in (cfg0.get("eval") or {}).get("rare_classes", [])) +
          f". '{ego} alone' / '{collab[0]} alone': only that robot sends (all its modalities); "
          "the missing robot's features are zeros.", ""]

    # 2. main result
    L += ["## 2. Main result (CAML Table 6 analogue)", ""]
    main_models = [m for m in (teacher, student, student_rgbd, nokd) if m]
    L += md_table(["Model", "Test mIoU", f"{ego} alone", f"{collab[0]} alone"],
                  [[m.tag + (f" (from {teacher_tag(m)})" if m.teacher else ""), fmt(stats(m.per_seed(m.test)), n),
                    fmt(stats(m.per_seed(m.alone(ego))), n), fmt(stats(m.per_seed(m.alone(collab[0]))), n)]
                   for m in main_models])
    L += ["CAML reports for the indoor scene: " + "; ".join(f"{k} {v}" for k, v in CAML_REPORTED.items()) +
          ". Those use other splits, robots, label processing and noise, and single-robot models trained "
          "separately, so they are context, not a like-for-like comparison.", ""]

    # 3. which modalities to share
    students = [m for m in models if m.exp in ("E2", "E3") and m.runs]
    if students:
        L += ["## 3. Which modalities to share (CAML Table 4 analogue)", "",
              "Students distilled from the RGB+depth+IR teacher; both robots send the listed modalities.", ""]
        ranked = sorted(students, key=lambda m: -(stats(m.per_seed(m.test)) or {"mean": -1})["mean"])
        L += md_table(["Shared modalities", "Test mIoU", f"{ego} alone", f"{collab[0]} sends (KiB/frame)"],
                      [[short_mods(m.modalities), fmt(stats(m.per_seed(m.test)), n),
                        fmt(stats(m.per_seed(m.alone(ego))), n),
                        f"{m.kib_sent({r: m.modalities for r in m.robots}):.0f}"] for m in ranked])

    # 4-6. paired comparisons
    comparisons = []

    def compare(title, a, b, getter_a, getter_b, note=""):
        if a and b:
            diff = paired(a.per_seed(getter_a(a)), b.per_seed(getter_b(b)))
            comparisons.append({"comparison": title, "a": a.name, "b": b.name, "diff": diff, "note": note})

    full_test = lambda m: m.test  # noqa: E731
    compare("Extra teacher modality: student from RGB+D+IR teacher minus student from RGB+D teacher",
            student, student_rgbd, full_test, full_test, "CAML Table 3 analogue")
    compare("Extra modality in the teacher itself: RGB+D+IR teacher minus RGB+D teacher",
            teacher, teacher_rgbd, full_test, full_test)
    compare("Pre-fusion minus post-fusion teacher", pre_t, teacher, full_test, full_test, "CAML App. A.3.2")
    compare("Pre-fusion minus post-fusion RGB student", pre_s, student, full_test, full_test,
            "same network; only the teacher differs")
    compare("Distillation: RGB student minus RGB student without KD", student, nokd, full_test, full_test,
            "CAML Table 2 analogue")
    for m in (teacher, student):
        if m:
            compare(f"Collaboration ({m.label}): both robots minus {ego} alone", m, m,
                    full_test, lambda mm: mm.alone(ego), "CAML Fig. 4 analogue")
    if teacher and len(pats) >= 3:
        full_key = combo_key(restricted(teacher, pats[0]["spec"]))
        cheap_key = combo_key(restricted(teacher, pats[2]["spec"]))
        compare(f"Bandwidth: teacher with '{pats[2]['label']}' minus '{pats[0]['label']}'", teacher, teacher,
                lambda mm: mm.combo(cheap_key), lambda mm: mm.combo(full_key),
                f"{teacher.kib_sent(restricted(teacher, pats[2]['spec'])):.0f} vs "
                f"{teacher.kib_sent(restricted(teacher, pats[0]['spec'])):.0f} KiB per frame")
    results["comparisons"] = comparisons
    if comparisons:
        L += ["## 4. Key comparisons", "",
              "Differences in test mIoU points, computed per seed (each student with its own teacher) and then "
              "averaged; positive means the first model is better.", ""]
        L += md_table(["Comparison", "Δ test mIoU", "Note"],
                      [[c["comparison"], fmt(c["diff"], n, sign=True), c["note"]] for c in comparisons])

    # 5. input combinations
    if pats:
        L += ["## 5. Input combinations", "",
              "Test mIoU when each robot sends only part of its inputs (missing features are zeros; no model "
              "was trained on such inputs). Each model keeps only its own modalities; '= #k' marks a "
              "combination that is the same as column k for that model.", ""]
        L += md_table(["#", "Combination", *models[0].robots],
                      [[f"#{i}", p["label"], *(short_mods(p["spec"].get(r) or []) or "nothing" for r in models[0].robots)]
                       for i, p in enumerate(pats, 1)])
        rows = []
        for m in models:
            if not m.runs:
                continue
            cells, seen = [], {}
            for i, p in enumerate(pats, 1):
                combo = restricted(m, p["spec"])
                key = combo_key(combo)
                if not any(combo.values()):
                    cells.append("n/a")
                elif key in seen:
                    cells.append(f"= #{seen[key]}")
                else:
                    seen[key] = i
                    st = stats(m.per_seed(m.combo(key)))
                    results["models"][m.name].setdefault("combinations", {})[p["label"]] = {
                        "key": key, "test": st, "kib_sent": m.kib_sent(combo)}
                    cells.append(fmt(st, n))
            rows.append([m.exp, m.label, *cells])
        L += md_table(["Exp", "Model", *(f"#{i}" for i in range(1, len(pats) + 1))], rows)

    # 6. per-class IoU
    cls_models = [m for m in (teacher, student, student_rgbd, nokd, pre_s) if m]
    if cls_models:
        L += ["## 6. Per-class IoU (test, full inputs)", ""]
        L += md_table(["Model", *class_names],
                      [[m.tag, *(fmt(stats(m.per_seed(m.class_iou(c))), n, digits=1).split(" (n")[0]
                                   for c in range(len(class_names)))] for m in cls_models])

    # 7. efficiency
    eff_files = sorted((out.parent / "_efficiency").glob("efficiency_*.json"))
    eff = json.loads(eff_files[-1].read_text()) if eff_files else None
    L += ["## 7. Efficiency (CAML Tables 1 and 8 analogues)", ""]
    rows = []
    for m in models:
        e = (eff or {}).get("results", {}).get(m.name, {})
        robot_ms = ", ".join(f"{r} {v['mean']:.1f}" for r, v in e.get("robot_ms", {}).items() if r != ego) or "—"
        rows.append([m.exp, m.label, f"{(m.info('parameters', 'total') or e.get('parameters', {}).get('total', 0)) / 1e6:.1f}",
                     f"{m.kib_sent({r: m.modalities for r in m.robots}) or 0:.0f}",
                     robot_ms, f"{e['server_ms']['mean']:.1f}" if e else "—",
                     f"{e['batch_ms']['mean']:.1f}" if e else "—",
                     f"{m.info('mean_epoch_time_s'):.1f}" if m.info("mean_epoch_time_s") else "—"])
    L += md_table(["Exp", "Model", "Params (M)", f"{collab[0]} sends (KiB/frame)", "Robot encode (ms)",
                   "Server fuse+decode (ms)", f"Batch {eff['results'][next(iter(eff['results']))]['batch_size'] if eff else 32} (ms)",
                   "Train s/epoch"], rows)
    L += [f"Latency from {eff_files[-1].name} ({eff['device']})." if eff else
          "Latency columns fill in after `tools/measure_efficiency.py` has run.", ""]

    # figures
    figs = []
    if not args.no_figures:
        plt = setup_matplotlib()
        curve_models = [m for m in (teacher, student, student_rgbd, nokd) if m]
        figs.append(("Learning curves", fig_learning_curves(plt, curve_models, out)))
        if teacher:
            figs.append(("Teacher input combinations", fig_teacher_combinations(plt, teacher, pats, out)))
        points = []
        if teacher:
            for i, p in enumerate(pats, 1):
                combo = restricted(teacher, p["spec"])
                st = stats(teacher.per_seed(teacher.combo(combo_key(combo))))
                if st and combo.get(ego):  # the ego-sensor failure case is not a bandwidth choice
                    points.append({"group": "teacher, input combinations", "tag": f"T #{i}",
                                   "kib": teacher.kib_sent(combo), "label_point": True, **st})
        for m in models:
            if m is teacher or not m.runs:
                continue
            st = stats(m.per_seed(m.test))
            if st:
                group = "students, full inputs" if m.exp in ("E2", "E3") else "other models, full inputs"
                points.append({"group": group, "tag": f"{m.exp} {short_mods(m.modalities)}", "kib": m.kib_sent(
                    {r: m.modalities for r in m.robots}), "label_point": m.exp == "E2", **st})
        results["tradeoff_points"] = points
        figs.append(("Accuracy against communication", fig_tradeoff(plt, points, out)))
        L += ["## 8. Figures", ""]
        for title, path in figs:
            if path:
                L += [f"**{title}**", "", f"![{title}]({path.name})", ""]
        if points:
            L += ["Accuracy against communication, as a table:", ""]
            L += md_table(["Point", "Group", "KiB/frame", "Test mIoU"],
                          [[p["tag"], p["group"], f"{p['kib']:.0f}", fmt(p, n)] for p in points])
            L += [f"'T #k' is the teacher under input combination #k of section 5 (#5, {ego}'s sensors failing, "
                  "is left out because it is not a bandwidth choice). The figure labels only the teacher's points "
                  "and the E2 student; groups are shifted slightly sideways where they send the same amount.", ""]

    # protocol
    L += ["## 9. Protocol and deviations from CAML", "",
          "- Data: CoPeD NYUARPL, ego ptah (Jackal) + race1 (drone), only while race1 is airborne (207 s); "
          "block-interleaved splits (train 1,345 samples at 10 Hz, val 300 and test 899 frames).",
          "- Labels: CoPeD's zero-shot (RAM + Grounding DINO + SAM) masks, 9 indoor classes, unlabeled ignored.",
          "- Depth: ZoeDepth from CoPeD's videos (computed from RGB); recorded RealSense depth is unusable "
          "(saturated 8-bit). IR: left infrared camera. CAML used RGB + depth only.",
          "- Noise: Gaussian sigma 0.12 on ptah's RGB, depth and IR, at train and test (CAML: unspecified "
          "noise on the ground robot's RGB-D).",
          "- Model (open in CAML): one ImageNet ResNet-18 per modality shared by both robots (matches CAML's "
          "25.5M parameters), cross-attention over modalities, FCN head; KD per pixel over all pixels.",
          "- Training as CAML: Adam 1e-3, cosine, batch 32, 200 epochs, alpha 0.9, temperature 4; best "
          "checkpoint by val mIoU with full inputs.",
          "- Not reproduced: the AML and FCN baselines (separately trained single-robot models).", ""]

    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with open(out / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "seed", "checkpoint", "split", *models[0].robots,
                         "miou", "miou_common", "pixel_accuracy"])
        for m in models:
            for seed, run in sorted(m.runs.items()):
                for ck in ("best", "last"):
                    for split, combos in run["metrics"][ck].get("subsets", {}).items():
                        for key, r in combos.items():
                            parts = dict(p.split("=", 1) for p in key.split(" "))
                            writer.writerow([m.exp, m.name, seed, ck, split, *(parts.get(x, "") for x in m.robots),
                                             f"{r['miou']:.5f}", f"{r['miou_common']:.5f}", f"{r['pixel_accuracy']:.5f}"])
    print(f"report -> {out / 'results.md'}  ({found} runs, {sum(1 for _, p in figs if p)} figures)")


if __name__ == "__main__":
    main()
