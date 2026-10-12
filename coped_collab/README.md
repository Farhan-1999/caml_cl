# coped_collab: CAML reproduction on CoPeD (Track A)

A faithful reproduction of **CAML** (Collaborative Auxiliary Modality Learning, Liu et al., NeurIPS 2025) on the
CoPeD NYUARPL sequence. The ground robot **ptah** (the robot whose view is segmented) collaborates with the drone
**race1**. Both robots provide **RGB**, **depth** (ZoeDepth from the released videos) and **IR** (left infrared camera).

- **Teacher:** sees all modalities of both robots.
- **Students:** learn from the teacher by distillation but receive fewer modalities.
- **Inference:** both robots still share features at test time.

## Layout
```
<workspace>/
  coped_collab/            this folder (code, configs, split files)
    coped/                 package: data pipeline, CAML model, losses, metrics, training engine
    configs/track_a.yaml   data, model and training defaults for Track A
    configs/caml/          one config per model + experiments.yaml (the experiment list)
    splits/track_a/        train/val/test sample lists, split report, preview (kept in git)
    tools/                 command-line scripts (see below)
  cache/track_a/           pre-resized inputs and labels (1.4 GB); all that training reads
  exported/NYUARPL/        per-frame export of the rosbags; only needed to rebuild the cache
  runs/                    created by training
```
Relative paths in the configs are resolved from `coped_collab/`. If the data lives elsewhere, add
`--set data.cache_dir=/abs/path/cache/track_a` (and `data.export_root=...`) to any command.

## Setup on the GPU machine
The complete runbook is `../AGENTS.md` at the repository root. It covers git, the raw-data transfer, exporting the bags there, every pipeline stage with its expected output, and troubleshooting. `scripts/track_a.sh` runs the stages. The notes below cover the short path when the cache is copied instead of rebuilt.

1. **Copy:** `coped_collab/` and `cache/track_a/` side by side. The export isn't needed for training.
2. **Python 3.10+:** install the PyTorch build that matches the machine's CUDA from https://pytorch.org/get-started/locally/.
3. **Libraries:** `pip install numpy opencv-python-headless pyyaml`. `rosbags` is only for re-exporting the bags.
4. **ImageNet weights:** the first model build downloads the ResNet-18 weights (45 MB) into `~/.cache/torch/hub/checkpoints/`. Without internet, copy `resnet18-f37072fd.pth` there.

## Running
From inside `coped_collab/`:
```bash
# 1. Check the models build (and download the ImageNet weights)
python tools/check_model.py --configs configs/caml/*.yaml

# 2. End-to-end test of the whole suite: 12 tiny runs into runs/<name>_smoke, ~2 min
python tools/run_experiments.py --manifest configs/caml/experiments.yaml --optional --seeds 0 --smoke

# 3. The real suite: 12 configs x 3 seeds, teachers before students
python tools/run_experiments.py --manifest configs/caml/experiments.yaml --optional --dry-run   # plan + status
nohup python tools/run_experiments.py --manifest configs/caml/experiments.yaml --optional > ../runs/suite.out 2>&1 &

# 4. Model efficiency on this GPU (parameters, message sizes, robot / server latency)
python tools/measure_efficiency.py --manifest configs/caml/experiments.yaml --optional
```
**Interruptions:**
- Running the same command again continues interrupted runs and skips finished ones.
- To watch a run: `tail -f ../runs/_logs/<name>_seed<k>.log`.
- `--only E1 E2` and `--seeds 0` limit what runs.

**Results report (Step 4).** These read only the run folders:
```bash
python tools/summarize.py --manifest configs/caml/experiments.yaml --optional   # tables + figures
python tools/qualitative.py --manifest configs/caml/experiments.yaml --seed 0   # Fig. 5-style predictions
```
- `summarize.py` needs no GPU and no data. It writes `../runs/_report/`: `results.md`, `results.json`, `runs.csv` and the figures.
- **Where to run it:** on the GPU machine, or locally after copying `runs/` back. Copy everything except the `*.pt` checkpoints if you don't need the qualitative figure.
- `qualitative.py` needs the `best.pt` checkpoints of the models it shows, plus the cache. It runs on CPU in about a minute.
- **Testing:** with `--smoke`, both scripts read the smoke runs instead.

`results.md` contains:
1. All models.
2. The main result (CAML Table 6 analogue).
3. Which modalities to share (Table 4).
4. Key comparisons as per-seed paired differences:
   - extra teacher modality (Table 3), pre- vs post-fusion, distillation (Table 2);
   - collaboration gain (Fig. 4);
   - the bandwidth trade-off.
5. The five input combinations.
6. Per-class IoU.
7. Efficiency (Tables 1 and 8). Latency appears once `measure_efficiency.py` has run.
8. Figures, each with a table version:
   - learning curves;
   - the teacher's input combinations over training;
   - accuracy against data sent.
9. The protocol and the deviations from CAML.

**Single runs and evaluation:**
```bash
python tools/train.py --config configs/caml/teacher_rgb_depth_ir.yaml --seed 0
python tools/evaluate.py --run ../runs/caml_student_rgb/seed0 --split test --drop-robots race1
```

## Experiments (`configs/caml/experiments.yaml`)
| ID | Models | Mirrors in CAML |
|---|---|---|
| E1 | Teacher: both robots, RGB + depth + IR | Teacher |
| E2 | Student: both robots share RGB only (from E1) | Table 6, main result |
| E3 | Students: RGB+depth, RGB+IR, depth+IR, depth, IR (from E1) | Table 4, which modality to keep |
| E4 | Pre-fusion teacher + RGB student | Pre-fusion ablation (App. A.3.2) |
| E5 | RGB+depth teacher -> RGB student | CAML's own setup; with E2, Table 3 |
| E6 | Every model tested with each robot alone and with every modality subset (see below) | Fig. 4, fewer agents |
| E7 | RGB student without distillation (optional, `--optional`) | Table 2, RGB-only ablation |

An RGB-only student is the same network under pre- and post-fusion, since there's only one modality to fuse. So E4's student differs from E2's only in which teacher it learns from.

## Evaluating input combinations
After every eval epoch each model is evaluated on val and test for five combinations of what each robot sends. They're listed in `eval.selected_subsets`:

| # | ptah sends | race1 sends | What it shows |
|---|---|---|---|
| 1 | RGB+depth+IR | RGB+depth+IR | Upper bound: everything shared |
| 2 | RGB+depth+IR | nothing | No communication: what collaboration adds |
| 3 | RGB+depth+IR | RGB | ptah's own sensors are free; race1 sends one map instead of three |
| 4 | RGB | RGB | CAML's reduced-modality setting |
| 5 | nothing | RGB+depth+IR | ptah's sensors fail: race1 alone predicts ptah's view |

Each model keeps only its own modalities, and duplicate combinations are dropped. So the teacher, the RGB+depth teacher and the RGB+IR student get 5; the other students get 3.

**Wider evaluation:** set `eval.subsets: all` for every combination (63 for three modalities), or `robots` for together plus each robot alone. Each robot can send any subset of the model's modalities, including nothing.

**How missing inputs are handled:**
- Missing features are zeros, the way CAML runs with fewer agents.
- A modality that no robot sends is left out of the fusion.
- CAML's models are trained on one fixed input set, so these results show how they degrade on inputs they never trained on.

**Selection and cost:**
- The best checkpoint is chosen on val with full inputs only. Test results never influence training.
- The encoders run once per batch and only fusion and decoding repeat per combination, so this is cheap.
- To make it cheaper still, set `eval.subset_splits: [val]` or `train.eval_every: 5` in `configs/track_a.yaml`.
- Any finished run can be evaluated on more combinations later, without retraining: `python tools/evaluate.py --run ../runs/<name>/seed0 --subsets all`.

## What a run produces
Each run writes to `runs/<name>/seed<k>/`:

| File | Contents |
|---|---|
| `config.yaml` | the exact config used |
| `log.csv` | per-epoch loss and val mIoU with full inputs |
| `subset_eval.csv` | per eval epoch, split and input combination: what ptah and race1 sent (e.g. `rgb+ir`, `none`), mIoU, mIoU-common, pixel accuracy |
| `best.pt`, `last.pt` | checkpoints; best = highest val mIoU with full inputs |
| `metrics.json` | for best and last: `val` and `test` (full inputs), `test_without_collaborators` (ptah alone, E6) and `subsets` (val/test per combination, with per-class IoU) |
| `model_info.json` | parameters, per-frame message sizes, seconds per epoch |

## Protocol (details in `splits/track_a/split_report.md`)
- **Window:** only the 207 s while race1 is airborne. Each ptah frame is paired with race1's frames within 50 ms.
- **Splits:** interleaved time blocks; every 100 s gives 70 s train, 10 s val, 20 s test, with 5 s dropped at each boundary.
  - Train: 1,345 samples at 10 Hz.
  - Val: 300 frames; test: 899 frames (every frame).
- **Inputs:** resized to 224 x 224. Gaussian noise (sigma 0.12) is added to ptah's three inputs, as CAML adds noise to the ground robot's RGB-D.
- **Labels:** the 9 indoor CoPeD classes; unlabeled pixels are ignored.
  - `miou` averages the classes present in a split.
  - `miou_common` also leaves out light and table (under 0.1% of pixels).
- **Training (CAML App. A.4-A.5):**
  - Adam with lr 1e-3, cosine schedule, batch 32, 200 epochs.
  - Students use (1 - 0.9) CE + 0.9 * 4^2 * soft-CE against the teacher.
- **Model:**
  - One ImageNet ResNet-18 per modality, shared by both robots.
  - The robots' maps are concatenated and reduced by a 1 x 1 conv.
  - Cross-attention fuses the modalities, then an FCN head and bilinear upsampling.
  - The RGB+depth version has 25.1M parameters, against CAML's reported 25.5M.
