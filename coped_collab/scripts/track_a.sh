#!/usr/bin/env bash
# Track A (CAML reproduction on CoPeD NYUARPL) on a Linux GPU machine, one stage at a time.
#
#   bash coped_collab/scripts/track_a.sh <stage>
#
# Stages, in order (each is safe to re-run):
#   verify      raw files present with the right sizes (add HASH=1 for checksums)
#   export      rosbags + labels + ZoeDepth videos -> exported/NYUARPL/<robot>/ (all 4 robots)
#   splits      rebuild the Track A splits and confirm they equal the committed ones
#   cache       pre-resize the Track A inputs into cache/track_a/
#   check       data pipeline and model checks
#   smoke       the whole suite as tiny runs (~minutes) into runs/<name>_smoke
#   train       the full suite: 12 configs x 3 seeds (hours; run inside tmux or nohup)
#   efficiency  parameters, message sizes and latency on this GPU
#   report      runs/_report/: results.md, figures, qualitative predictions
#   pack        runs/track_a_results.tar.gz with everything except checkpoints
#   prepare     = verify export splits cache check
#
# Environment: PYTHON (default python), DATA_ROOT (raw dataset root, default: the repo root).
# Extra arguments after smoke/train/efficiency are passed on, e.g.: train --set train.num_workers=2
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/../.." && pwd)"   # repo root = workspace root
CODE="$ROOT/coped_collab"
DATA_ROOT="${DATA_ROOT:-$ROOT}"
read -r -a PY <<< "${PYTHON:-python}"   # may carry flags, e.g. "python -u"
SEQ=NYUARPL
ROBOTS=(ptah race1 race5 wilbur)
MANIFEST=configs/caml/experiments.yaml
cd "$CODE"
EXTRA=("${@:2}")   # anything after the stage goes to the Python tool, e.g. --set train.num_workers=2

say() { printf '\n== %s ==\n' "$*"; }

stage_verify() {
  say "verify raw files under $DATA_ROOT"
  if [[ "${HASH:-0}" == 1 ]]; then
    "${PY[@]}" tools/verify_raw_data.py --root "$DATA_ROOT" --sequence "$SEQ" --hash
  else
    "${PY[@]}" tools/verify_raw_data.py --root "$DATA_ROOT" --sequence "$SEQ"
  fi
}

stage_export() {
  say "export $SEQ (${ROBOTS[*]}) -> $ROOT/exported"
  for robot in "${ROBOTS[@]}"; do
    if [[ -f "$ROOT/exported/$SEQ/$robot/export_summary.json" ]]; then
      echo "$robot already exported, skipping (delete exported/$SEQ/$robot to redo it)"
    else
      # a half-finished export (no summary) is replaced
      "${PY[@]}" tools/export_sequence.py --root "$DATA_ROOT" --sequence "$SEQ" --robots "$robot" \
        --out "$ROOT/exported" --overwrite
    fi
  done
  "${PY[@]}" tools/verify_raw_data.py --root "$DATA_ROOT" --sequence "$SEQ" --check-export "$ROOT/exported/$SEQ"
}

stage_splits() {
  say "rebuild Track A splits"
  "${PY[@]}" tools/build_splits.py --config configs/track_a.yaml
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git -C "$ROOT" diff --quiet -- coped_collab/splits/track_a/train.csv coped_collab/splits/track_a/val.csv \
        coped_collab/splits/track_a/test.csv; then
      echo "splits are identical to the committed ones"
    else
      echo "ERROR: rebuilt splits differ from the committed ones (git diff coped_collab/splits)" >&2
      exit 1
    fi
  fi
}

stage_cache() {
  say "build cache/track_a"
  "${PY[@]}" tools/build_cache.py --config configs/track_a.yaml
}

stage_check() {
  say "data checks"
  "${PY[@]}" tools/check_data.py --config configs/track_a.yaml
  say "model checks"
  "${PY[@]}" tools/check_model.py --configs configs/caml/*.yaml
}

stage_smoke() {
  say "smoke suite"
  "${PY[@]}" tools/run_experiments.py --manifest "$MANIFEST" --optional --seeds 0 --smoke "${EXTRA[@]}"
}

stage_train() {
  say "full suite"
  "${PY[@]}" tools/run_experiments.py --manifest "$MANIFEST" --optional --dry-run "${EXTRA[@]}"
  "${PY[@]}" tools/run_experiments.py --manifest "$MANIFEST" --optional "${EXTRA[@]}"
}

stage_efficiency() {
  say "efficiency"
  "${PY[@]}" tools/measure_efficiency.py --manifest "$MANIFEST" --optional "${EXTRA[@]}"
}

stage_report() {
  say "report"
  "${PY[@]}" tools/summarize.py --manifest "$MANIFEST" --optional
  "${PY[@]}" tools/qualitative.py --manifest "$MANIFEST" --seed 0
}

stage_pack() {
  say "pack results (no checkpoints)"
  local items=()
  for p in runs/_report runs/_efficiency runs/_logs; do [[ -e "$ROOT/$p" ]] && items+=("$p"); done
  for p in "$ROOT"/runs/caml_*; do [[ -d "$p" && "$p" != *_smoke ]] && items+=("runs/$(basename "$p")"); done
  if [[ ${#items[@]} -eq 0 ]]; then echo "nothing to pack yet" >&2; exit 1; fi
  tar -C "$ROOT" --exclude='*.pt' -czf "$ROOT/runs/track_a_results.tar.gz" "${items[@]}"
  ls -lh "$ROOT/runs/track_a_results.tar.gz"
}

case "${1:-}" in
  verify|export|splits|cache|check|smoke|train|efficiency|report|pack) "stage_$1" ;;
  prepare) stage_verify; stage_export; stage_splits; stage_cache; stage_check ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print } /^set -euo/ { exit }' "$SCRIPT"; exit 1 ;;
esac
