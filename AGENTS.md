# AGENTS.md: running Track A on the GPU machine

This file is for **Codex** (and the user) on the Linux GPU machine. It covers setting up the code and data, exporting
the raw CoPeD sequence, and running the full Track A experiment suite. It was prepared on the Windows machine where
the code was written and tested on CPU; everything below has been run there except the GPU training itself.

**Track A** is a faithful reproduction of **CAML** (Collaborative Auxiliary Modality Learning, Liu et al.,
NeurIPS 2025) on the CoPeD NYUARPL sequence.
- **Robots:** the ground robot **ptah**, whose camera view is segmented, collaborates with the drone **race1**.
- **Modalities:** RGB, depth (ZoeDepth) and IR.
- **Models:** a teacher sees all modalities of both robots; students are distilled from it and see fewer.
- **Evaluation:** every model is also tested with one robot or some modalities missing.

More detail: `coped_collab/README.md` (the code) and `coped_collab/splits/track_a/split_report.md` (the data split).

## Rules for the agent
1. **Never modify or delete the raw data:** `groundtruth_semantics/`, `sequence/`, `video/`. Derived data goes only to `exported/`, `cache/` and `runs/`.
2. **Do not change the experimental protocol** without the user's approval. This covers:
   - the `data`, `split`, `noise`, `model`, `train`, `distill` and `eval` settings in `coped_collab/configs/`;
   - the experiment list `configs/caml/experiments.yaml`;
   - the committed split files in `coped_collab/splits/track_a/`.
   Path overrides with `--set data.export_root=...` / `data.cache_dir=...` are fine.
3. **Check each stage before starting the next.** The expected output of every stage is listed below. On any mismatch, stop and report it with the command's output; do not "fix" data or splits to make numbers match.
4. **Never delete `runs/`, and never pass `--overwrite` to training.** Interrupted training continues with the same command. Finished runs are skipped automatically.
5. **Do not commit** data, exports, cache, checkpoints or runs. `.gitignore` already excludes them; don't change it to include them.
6. **Report** by pointing to `runs/_report/results.md`. Mention any missing or failed runs (it lists them at the top).

## Layout
```
<repo root>/                     the git repository; also the workspace root
  AGENTS.md                      this file
  .gitignore, .gitattributes     only code is tracked; text uses LF line endings
  coped_collab/                  all code (tracked)
    configs/track_a.yaml         Track A data, model, training and evaluation settings
    configs/caml/                one config per model + experiments.yaml (the suite: E1-E7, 3 seeds)
    splits/track_a/              train/val/test sample lists (committed; must not change)
    transfer/                    NYUARPL_raw_manifest.json (sizes + SHA-256 of every needed raw file),
                                 NYUARPL_raw_paths.txt (rsync file list), NYUARPL_export_counts.json
    scripts/track_a.sh           the pipeline, one stage per command
    tools/, coped/               scripts and the Python package
  groundtruth_semantics/  sequence/  video/     raw CoPeD data   (copied, not in git)
  exported/                      created by the export stage   (not in git)
  cache/                         created by the cache stage    (not in git)
  runs/                          created by training           (not in git)
```
All configs use paths relative to `coped_collab/`, so this layout needs no path changes.

## 1. One-time setup

### 1a. Code: via git
On the Windows machine (the user does this once), in `E:\CoPed dataset`:
```bash
git init
git add .
git status              # must list only .gitignore, .gitattributes, AGENTS.md and coped_collab/...
git commit -m "Track A: CAML reproduction on CoPeD"
git branch -M main
git remote add origin <private repository URL>
git push -u origin main
```
On the GPU machine:
```bash
git clone <private repository URL> coped && cd coped    # this folder is <repo root> from now on
```

### 1b. Raw data: copy into the repo root (about 51 GB, 16 items)
The needed files are listed in `coped_collab/transfer/NYUARPL_raw_paths.txt`:
- the 4 NYUARPL rosbags (50.1 GB);
- the 4 ZoeDepth depth videos (0.14 GB);
- each robot's `pred.json` and `Annotations/` labels (0.79 GB).

Copy them so that they keep their relative paths under the repo root.

**With rsync.** It resumes and skips finished files. Run it from WSL on the Windows machine; Git Bash has no rsync:
```bash
cd "/mnt/e/CoPed dataset"
rsync -av -r --progress --partial --files-from=coped_collab/transfer/NYUARPL_raw_paths.txt \
  ./ user@gpu-host:/path/to/coped/
```
**Without rsync:** use scp, WinSCP or a shared drive. Copy the listed files and folders keeping the same paths, e.g. `sequence/indoor/NYUARPL/ptah_....bag`.

**Afterwards:** the `verify` stage below confirms everything arrived. Nothing else from the dataset is needed: no other sequences, `image_video`, `semantics_video`, zips, `prepared/` or `exported/`.

If the data has to live elsewhere, set `DATA_ROOT=/path/to/dataset` for the pipeline. The export still goes to `<repo root>/exported`.

### 1c. Python environment
**Python:** 3.10 or newer. `rosbags` needs 3.10+; the code was tested with Python 3.13.

**Tested versions:**

| Package | Version |
|---|---|
| torch | 2.9.1 |
| torchvision | 0.24.1 |
| numpy | 2.2.4 |
| opencv | 4.11 |
| pyyaml | 6.0.2 |
| rosbags | 0.11.5 |
| matplotlib | 3.10.7 |

```bash
python -m venv .venv && source .venv/bin/activate      # or a conda env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match `nvidia-smi`'s CUDA
pip install -r coped_collab/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True
```
**ImageNet weights:** the first model build downloads ResNet-18 weights (45 MB) from download.pytorch.org into `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`. Without internet, copy that file there.

**Disk:** plan for about 100 GB.

| Item | Size |
|---|---|
| Raw data | 51 GB |
| `exported/` | 24 GB |
| `cache/` | 1.4 GB |
| `runs/` with checkpoints | ~15 GB, estimated |

## 2. Pipeline
Run from the repo root, one stage at a time, and compare with the expected output before continuing:
```bash
bash coped_collab/scripts/track_a.sh <stage>      # no argument prints the stage list
```

| Stage | What it does | Expected output |
|---|---|---|
| `verify` | Raw files present with the right sizes. Add `HASH=1` to also compare SHA-256 (a few minutes). | `all 16 items present with the right sizes` |
| `export` | Rosbags, labels and depth videos into per-frame files for all 4 robots (~10 min). | `export matches the reference: ...` (counts below) |
| `splits` | Rebuilds the Track A splits and compares them with the committed ones. | `train: 1345 samples in 3 blocks`, `val: 300 samples in 2 blocks`, `test: 899 samples in 2 blocks`, `splits are identical to the committed ones` |
| `cache` | Pre-resizes Track A inputs; training reads only this. | `train: 1345 samples, 0.74 GB`, `val: 300 samples, 0.17 GB`, `test: 899 samples, 0.50 GB` |
| `check` | Data pipeline and model checks. | `all checks passed`. Then for all 12 model configs: `robot/server split matches = True` and `full and ptah-only match the plain forward = True`. Parameters: teacher RGB+D+IR 36.77M, two-modality models 25.07M, one-modality students 12.29M. |
| `smoke` | The whole suite as tiny runs (a few minutes). | `summary: done 0, trained 12, failed 0, waiting 0` (on a re-run: `done 12, trained 0`) |
| `train` | The full suite: 12 configs x 3 seeds, teachers before students. | Ends with `summary: done 0, trained 36, failed 0, waiting 0`. If interrupted, the same command continues it. |
| `efficiency` | Parameters, message sizes, robot vs server latency on this GPU. | `runs/_efficiency/efficiency_<gpu>.md` |
| `report` | Results report and qualitative figure. | `runs/_report/results.md` plus `fig_*.png` |
| `pack` | Results without checkpoints, to send back. | `runs/track_a_results.tar.gz` |

`prepare` runs `verify`, `export`, `splits`, `cache` and `check` in one go.

**Export frame counts** (they must match exactly; the `export` stage checks them against `transfer/NYUARPL_export_counts.json`):

| Robot | rgb | ir | ir_right | lidar | label | depth_zoe |
|---|---|---|---|---|---|---|
| ptah | 15788 | 15788 | 15788 | 5264 | 15788 | 15788 |
| race1 | 10204 | 10649 | 10650 | — | 10204 | 10204 |
| race5 | 15385 | 15472 | 15472 | — | 15385 | 15385 |
| wilbur | 15575 | 15749 | 15749 | 5248 | 15575 | 15575 |

Track A uses only ptah and race1 (RGB, ZoeDepth, left IR, labels). The other robots and LiDAR are exported for later experiments.

### Running the long stage
`train` takes hours. My estimate is 10-20 min per run on an A6000-class GPU and 36 runs in total; it was not measured on a GPU. Run it so it survives a disconnect:
```bash
tmux new -s tracka 'bash coped_collab/scripts/track_a.sh train 2>&1 | tee runs_train.out'
# or: nohup bash coped_collab/scripts/track_a.sh train > runs_train.out 2>&1 &
```
**Monitoring:**
- Overall progress: `tail -f runs_train.out`.
- One run in detail: `tail -f runs/_logs/<config name>_seed<k>.log`.
- Status of every run: `python coped_collab/tools/run_experiments.py --manifest coped_collab/configs/caml/experiments.yaml --optional --dry-run` (run from the repo root; it prints done / partial / todo).

Each epoch line shows the loss, val mIoU with full inputs, and val mIoU with each robot alone. A run's first epoch time shows the real speed.

## 3. Results
`bash coped_collab/scripts/track_a.sh report` writes `runs/_report/`:
- `results.md`: tables mirroring CAML's Tables 2, 3, 4, 6 and 8 and Fig. 4. It also covers the five input combinations, per-class IoU, efficiency, the protocol and its deviations from CAML.
- `results.json`, `runs.csv`: all numbers.
- `fig_learning_curves.png`, `fig_teacher_combinations.png`, `fig_tradeoff.png`, `fig_qualitative.png`.

`bash coped_collab/scripts/track_a.sh pack` creates `runs/track_a_results.tar.gz`: the report, efficiency, logs and every run's `metrics.json`, `log.csv`, `subset_eval.csv`, `config.yaml` and `model_info.json`. It leaves out the checkpoints (`*.pt`, several GB). Send that archive back.

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `verify`: size mismatch or missing file | Copy incomplete. Re-copy that file; rsync resumes. Use `HASH=1` to rule out corruption. |
| `export` fails on a bag | Bag truncated: run `HASH=1 ... verify`. A robot without `export_summary.json` is re-exported automatically on the next `export`. |
| `export matches the reference` fails | Do not continue. Report the printed differences. |
| `splits` reports different splits | Do not continue and do not commit new splits. Report `git diff --stat coped_collab/splits`. |
| `no cache at .../cache/track_a` | Run the `cache` stage. Or train from the export with `--set data.cache_dir=null` (slower). |
| `cache ... was built for other samples` | The splits changed; see the previous row. |
| ResNet download fails | No internet: place `resnet18-f37072fd.pth` in `~/.cache/torch/hub/checkpoints/`. |
| `DataLoader worker ... killed` or `/dev/shm` errors | Fewer loader workers for every run: `bash coped_collab/scripts/track_a.sh train --set train.num_workers=2` |
| CUDA out of memory | Should not happen at batch 32 on a 24 GB+ GPU. Report it rather than changing `train.batch_size` (protocol). |
| `teacher not finished, skipped` | Its teacher failed or is still missing. Re-run `train`; students run once their teacher is done. |
| `import cv2` fails with libGL errors | `pip uninstall opencv-python && pip install opencv-python-headless`. |
