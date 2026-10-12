"""Check that the raw CoPeD files a sequence needs arrived complete, using a manifest made at the source.

The manifest (coped_collab/transfer/<SEQ>_raw_manifest.json) lists every needed file with its size
and SHA-256, and for each label folder the number and total size of its PNGs.

Examples:
    # on the source machine: write the manifest and the rsync file list
    python tools/verify_raw_data.py --root "E:/CoPed dataset" --sequence NYUARPL --make
    # on the target machine: sizes and file counts (seconds), then optionally full checksums (minutes)
    python tools/verify_raw_data.py --root /data/coped
    python tools/verify_raw_data.py --root /data/coped --hash
    # after exporting: frame counts must equal the source machine's export
    python tools/verify_raw_data.py --root /data/coped --make-export-counts "E:/CoPed dataset/exported/NYUARPL"  # source
    python tools/verify_raw_data.py --root /data/coped --check-export /data/coped/exported/NYUARPL               # target
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TRANSFER_DIR = Path(__file__).resolve().parents[1] / "transfer"


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def needed_files(root: Path, sequence: str) -> tuple[list[Path], list[Path]]:
    """(single files, label folders) of a sequence: its rosbags, its ZoeDepth videos and, per robot,
    pred.json plus the Annotations folder."""
    bags = sorted((root / "sequence").glob(f"*/{sequence}/*.bag"))
    videos = sorted((root / "video" / "depth_video").glob(f"{sequence}_*_depth_*.mp4"))
    label_root = root / "groundtruth_semantics" / sequence
    robots = sorted(p for p in label_root.iterdir() if p.is_dir())
    files = bags + videos + [r / "pred.json" for r in robots]
    return files, [r / "Annotations" for r in robots]


def make(root: Path, sequence: str) -> None:
    files, folders = needed_files(root, sequence)
    if not files:
        raise SystemExit(f"no files for {sequence} under {root}")
    entries = []
    for f in files:
        t = time.time()
        entries.append({"path": f.relative_to(root).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)})
        print(f"  {entries[-1]['path']}  {entries[-1]['bytes'] / 1e9:.2f} GB  ({time.time() - t:.0f}s)", flush=True)
    dirs = []
    for d in folders:
        pngs = sorted(d.glob("*.png"))
        dirs.append({"path": d.relative_to(root).as_posix(), "files": len(pngs),
                     "bytes": sum(p.stat().st_size for p in pngs)})
    total = sum(e["bytes"] for e in entries) + sum(x["bytes"] for x in dirs)
    manifest = {"sequence": sequence, "total_bytes": total, "files": entries, "label_folders": dirs}
    TRANSFER_DIR.mkdir(exist_ok=True)
    out = TRANSFER_DIR / f"{sequence}_raw_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")  # LF on every OS
    # Paths for `rsync -av -r --files-from=...` (folders end with '/', rsync then copies their contents)
    paths = [e["path"] for e in entries] + [x["path"] + "/" for x in dirs]
    (TRANSFER_DIR / f"{sequence}_raw_paths.txt").write_text("\n".join(paths) + "\n", newline="\n")
    print(f"{len(entries)} files + {len(dirs)} label folders, {total / 1e9:.1f} GB -> {out}")


def verify(root: Path, sequence: str, check_hash: bool) -> None:
    manifest = json.loads((TRANSFER_DIR / f"{sequence}_raw_manifest.json").read_text())
    problems = []
    for e in manifest["files"]:
        p = root / e["path"]
        if not p.exists():
            problems.append(f"missing  {e['path']}")
            continue
        size = p.stat().st_size
        if size != e["bytes"]:
            problems.append(f"size     {e['path']}: {size} bytes, expected {e['bytes']}")
            continue
        if check_hash:
            t = time.time()
            ok = sha256(p) == e["sha256"]
            print(f"  {'ok ' if ok else 'BAD'} {e['path']} ({time.time() - t:.0f}s)", flush=True)
            if not ok:
                problems.append(f"checksum {e['path']}")
    for x in manifest["label_folders"]:
        d = root / x["path"]
        pngs = list(d.glob("*.png")) if d.exists() else []
        if len(pngs) != x["files"]:
            problems.append(f"labels   {x['path']}: {len(pngs)} PNGs, expected {x['files']}")
        elif sum(p.stat().st_size for p in pngs) != x["bytes"]:
            problems.append(f"labels   {x['path']}: total size differs")
    n = len(manifest["files"]) + len(manifest["label_folders"])
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} of {n} items are missing or differ")
    print(f"all {n} items present with the right sizes" + (" and checksums" if check_hash else
                                                            " (add --hash to also compare checksums)"))


def export_counts(export_dir: Path) -> dict:
    """Frames per robot and modality, and the consistency checks, from each export_summary.json."""
    out = {}
    for summary in sorted(export_dir.glob("*/export_summary.json")):
        s = json.loads(summary.read_text())
        counts = {m: v["count"] for m, v in s.get("bag", {}).get("modalities", {}).items()}
        for key in ("label", "depth_zoe"):
            if key in s:
                counts[key] = s[key]["count"]
        out[s["robot"]] = {"counts": counts, "checks": s.get("checks", {})}
    return out


def check_export(export_dir: Path, sequence: str) -> None:
    expected = json.loads((TRANSFER_DIR / f"{sequence}_export_counts.json").read_text())
    actual = export_counts(export_dir)
    problems = []
    for robot, exp in expected.items():
        got = actual.get(robot)
        if got is None:
            problems.append(f"{robot}: not exported")
            continue
        for key, n in exp["counts"].items():
            if got["counts"].get(key) != n:
                problems.append(f"{robot}.{key}: {got['counts'].get(key)} frames, expected {n}")
        if got["checks"] != exp["checks"]:
            problems.append(f"{robot}: checks {got['checks']} differ from {exp['checks']}")
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"export differs from the reference export in {len(problems)} place(s)")
    print(f"export matches the reference: {', '.join(f'{r} ' + str(v['counts']) for r, v in expected.items())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="dataset root holding groundtruth_semantics/, sequence/ and video/")
    parser.add_argument("--sequence", default="NYUARPL")
    parser.add_argument("--make", action="store_true", help="write the manifest from the files under --root")
    parser.add_argument("--hash", action="store_true", help="also compare SHA-256 checksums (reads every byte)")
    parser.add_argument("--make-export-counts", type=Path, metavar="EXPORT_SEQ_DIR",
                        help="record frame counts of a finished export, e.g. exported/NYUARPL")
    parser.add_argument("--check-export", type=Path, metavar="EXPORT_SEQ_DIR",
                        help="compare an export's frame counts with the recorded reference")
    args = parser.parse_args()
    if args.make_export_counts:
        counts = export_counts(args.make_export_counts)
        TRANSFER_DIR.mkdir(exist_ok=True)
        out = TRANSFER_DIR / f"{args.sequence}_export_counts.json"
        out.write_text(json.dumps(counts, indent=2) + "\n", newline="\n")
        print(f"{len(counts)} robots -> {out}")
    elif args.check_export:
        check_export(args.check_export, args.sequence)
    elif args.make:
        make(args.root, args.sequence)
    else:
        verify(args.root, args.sequence, args.hash)


if __name__ == "__main__":
    sys.exit(main())
