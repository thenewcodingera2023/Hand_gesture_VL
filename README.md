# Hand Gesture Recognition (MV Calculus Class Project)

![CI](https://github.com/thenewcodingera2023/Hand_gesture_VL/actions/workflows/ci.yml/badge.svg)

A 26-class hand gesture classifier built on MediaPipe hand landmarks, engineered into a 279-dimensional feature vector, and read by a 3-hidden-layer PyTorch MLP `R^279 -> R^26`. The system runs in real time on a laptop webcam and doubles as a teaching artefact: every step of training is a concrete instance of multivariable calculus (gradient, Jacobian, chain rule, level sets). The mathematical reading is in [MATH.md](MATH.md).

## Repository Layout

```
Hand_gesture_VL/
|-- src/
|   |-- hagrid_extractor.py     # Stage 1: parse HaGRIDv2 JSON -> raw landmark records
|   |-- preprocessor.py         # per-hand normalization + feature engineering -> (138,)
|   |-- feature_assembler.py    # two-hand combiner -> (279,)
|   |-- synthetic_builder.py    # Stage 1 CLI for build-singles / build-counts / validate
|   |-- dataset.py              # Stage 2: user-aware splits + oversampling
|   |-- models/
|   |   |-- baseline.py         # Stage 3: LR + RBF SVM
|   |   `-- mlp.py              # Stage 4: GestureMLP
|   |-- train.py                # Stage 4 training loop
|   |-- capture.py              # Stage 5: OpenCV + MediaPipe wrapper
|   |-- smoother.py             # Stage 5: sliding-window vote + confidence gate
|   |-- inference.py            # Stage 5: real-time webcam demo
|   |-- evaluate.py             # Stage 6: test-set scorecard
|   `-- mv_visualization.py     # Stage 6: MV-calculus plot + chain-rule helpers
|-- tests/                      # pytest suite (see tests/README.md)
|-- notebooks/                  # 01_data_exploration, 02_training_analysis, 03_mv_visualization
|-- data/
|   |-- labels.json             # 26-class integer label map
|   |-- hagrid_raw/             # downloaded HaGRIDv2 annotations
|   |-- processed/              # Stage 1 .npz outputs
|   `-- splits/                 # Stage 2 train/val/test .npz
|-- runs/
|   |-- baselines.csv           # Stage 3
|   |-- training_log.csv        # Stage 4
|   |-- mlp_best.pt             # Stage 4 best checkpoint
|   `-- evaluation/             # Stage 6 metrics, confusion matrix, MV plots
|-- hagrid_repo/                # git submodule -> github.com/hukenovs/hagrid (downloader, dataset, demos)
|-- tests/smoke_test_mediapipe.py   # standalone MediaPipe sanity check
|-- hand_landmarker.task        # MediaPipe Hand Landmarker model file
|-- requirements.txt
|-- README.md                   # this file
`-- MATH.md                     # multivariable calculus narrative
```

Gitignored (not in version control; produced by the pipeline or downloaded separately): `data/hagrid_raw/`, `data/processed/`, `data/splits/`, `runs/`, `evaluation/`, `venv/`, `hand_landmarker.task`.

## Environment Setup

Python 3.10+ is required.

### Clone (with submodules)

`hagrid_repo/` is a git submodule pointing at [hukenovs/hagrid](https://github.com/hukenovs/hagrid). Clone the project with submodules included so the Stage 1 downloader is available:

```
git clone --recurse-submodules <repo-url>
cd Hand_gesture_VL
```

Already cloned without submodules? Run once:

```
git submodule update --init --recursive
```

### Virtual environment and dependencies

```
python -m venv .venv
.venv\Scripts\activate                  # Windows
# source .venv/bin/activate              # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the dependency graph imports cleanly:

```
python -c "import mediapipe, torch, cv2, sklearn, numpy, pandas, matplotlib, seaborn; print('ok')"
```

### MediaPipe Hand Landmarker model

`hand_landmarker.task` is **not in version control** (~8 MB binary). Download it from Google's official MediaPipe model card and place it at the repo root before running the live demo or the smoke test:

```
# macOS / Linux
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Windows PowerShell
Invoke-WebRequest -OutFile hand_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Confirm the task file loads:

```
python tests/smoke_test_mediapipe.py
```

## Data Acquisition

**Requires network + ~10 GB of disk.** Skip this step if `data/hagrid_raw/annotations.zip` already exists.

HaGRIDv2 annotations ship the pre-extracted MediaPipe landmark arrays we train on; the 1.5 TB image archive is **not** required. Download the 14 relevant class annotation bundles via the vendored downloader:

```
python hagrid_repo/download.py --annotations --targets like dislike fist four ok one palm peace call mute rock stop three two_up --save_path data/hagrid_raw/
```

After download, annotation JSONs should live under `data/hagrid_raw/annotations/annotations/{train,val,test}/<class>.json`.

## Stage 1 — Data Extraction

Three commands write four arrays under `data/processed/`:

```
python -m src.hagrid_extractor --annotations-root data/hagrid_raw/annotations/annotations --out data/processed/hagrid_raw_records.npz --min-records 200000
python -m src.synthetic_builder build-singles --raw data/processed/hagrid_raw_records.npz --features-out data/processed/single_hand_features.npz --assembled-out data/processed/single_hand_assembled.npz
python -m src.synthetic_builder build-counts --features data/processed/single_hand_features.npz --out data/processed/synthetic_two_hand.npz
```

Expected outputs on the current HaGRIDv2 subset:

| File | Rows | Notes |
|---|---|---|
| `data/processed/hagrid_raw_records.npz` | 525,643 | raw `landmarks_raw (N,21,3)` + metadata |
| `data/processed/single_hand_features.npz` | 525,643 | per-hand `(N, 138)` features |
| `data/processed/single_hand_assembled.npz` | 525,643 | single-hand assembled `(N, 279)` rows |
| `data/processed/synthetic_two_hand.npz` | 6,500 | 500 samples per `count_6..count_18` |

Validate Stage 1 outputs:

```
python -m src.synthetic_builder validate --raw data/processed/hagrid_raw_records.npz --features data/processed/single_hand_features.npz --assembled data/processed/single_hand_assembled.npz --synthetic data/processed/synthetic_two_hand.npz
```

## Stage 2 — Dataset Splits

User-aware 80 / 10 / 10 split of HaGRID `user_id`s. Synthetic two-hand rows go to the split where both source users live; counts 6–18 in train are oversampled to 5,000 each with light Gaussian noise (sigma=0.01).

```
python -m src.dataset build --singles data/processed/single_hand_assembled.npz --synthetic data/processed/synthetic_two_hand.npz --out-dir data/splits --seed 20260514 --oversample-target 5000 --aug-sigma 0.01
python -m src.dataset verify --splits-dir data/splits
```

Outputs: `data/splits/{train,val,test}.npz`. Schema (per the codebase map): `X (float32)`, `y (int32)`, plus parallel metadata arrays `user_id`, `source`, `project_label`, `right_label`, `left_label`, `hagrid_split`, `is_synthetic (bool)`, `is_augmented (bool)`, `synth_right_user`, `synth_left_user`, `seed`.

Row counts on the current run: train 491,224 / val 49,853 / test 49,678.

## Stage 3 — Baselines

Logistic Regression on the full train split; RBF SVM via grid-searched C over a stratified subsample.

```
python -m src.models.baseline train-all --splits-dir data/splits --runs-dir runs --seed 20260514
```

Or train individually:

```
python -m src.models.baseline train-lr --splits-dir data/splits --runs-dir runs --seed 20260514
python -m src.models.baseline train-svm --splits-dir data/splits --runs-dir runs --seed 20260514 --subsample 5000 --refit-subsample 10000 --c-grid 0.1 1 10 100 --cv 5
```

Outputs: `runs/baselines.csv`, `runs/baseline_lr.joblib`, `runs/baseline_svm.joblib`. Gate: both val accuracy ≥ 0.85.

Re-score a saved model:

```
python -m src.models.baseline evaluate --splits-dir data/splits --runs-dir runs
```

## Stage 4 — MLP Training

```
python -m src.train --epochs 400 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --patience 25 --min-epochs 100 --scheduler-patience 10 --scheduler-factor 0.5 --min-lr 1e-5 --seed 20260514 --device auto
```

A short sanity-check pass:

```
python -m src.train --smoke
```

Outputs:

- `runs/mlp_best.pt` — best-val-loss checkpoint. Loaded via `torch.load(..., weights_only=False, map_location="cpu")`. Payload includes model + optimizer + scheduler state, the StandardScaler `scaler_mean` / `scaler_scale` (length 279 each), frozen `config`, `seed`, and the integer→name label map.
- `runs/training_log.csv` — 16 columns per epoch: `epoch, train_loss, val_loss, train_acc, val_acc, merged_train_acc, merged_val_acc, val_macro_f1, grad_norm, layer_1_weight_norm .. layer_4_weight_norm, lr, wall_seconds, timestamp`.

Gate: `merged_val_acc ≥ 0.93` AND beats both baselines on raw accuracy and macro F1. The training loop refuses to start without `runs/baselines.csv`.

## Stage 5 — Real-Time Inference

**Requires webcam; skip on headless CI.**

```
python -m src.inference --camera 0 --checkpoint runs/mlp_best.pt --labels data/labels.json --window 7 --threshold 0.75 --no-hand-clear 5 --min-detection-confidence 0.5 --device cpu
```

Behavior:

- Opens an OpenCV window titled **"Gesture Inference"** showing label, smoothed confidence, FPS, and (by default) the MediaPipe hand-landmark skeleton.
- Pass `--no-landmarks` to hide the skeleton overlay.
- **Quit:** press `q` or `Esc`.
- Confidence < `--threshold` (default 0.75) displays `"---"` instead of a class label (from `src/smoother.py`).
- 5 consecutive frames with no detected hand clears the smoother window and silences output.
- The checkpoint's `scaler_mean` / `scaler_scale` are applied to every live feature vector so the live distribution matches training.

**Manual demo for all 26 classes.** Counts `count_2` and `count_5` were dropped because they are structurally identical to the single-hand `peace` and `open_palm` control gestures — see [tasks/peace_count2_collision_fix.md](tasks/peace_count2_collision_fix.md). Hold gestures from `data/labels.json` in order:

| Count | Right hand | Left hand | Class to display |
|---|---|---|---|
| 0 (=fist) | fist | absent | `fist` |
| 1 | one | absent | `count_1` |
| 2 (=peace) | peace | absent | `peace` |
| 3 | three | absent | `count_3` |
| 4 | four | absent | `count_4` |
| 5 (=open_palm) | open_palm | absent | `open_palm` |
| 6 | open_palm | one | `count_6` |
| 7 | open_palm | peace | `count_7` |
| 8 | open_palm | three | `count_8` |
| 9 | open_palm | four | `count_9` |
| 10 | open_palm | open_palm | `count_10` |
| 11 | one | open_palm | `count_11` |
| 12 | peace | open_palm | `count_12` |
| 13 | three | open_palm | `count_13` |
| 14 | four | open_palm | `count_14` |
| 15 | one | one | `count_15` |
| 16 | peace | one | `count_16` |
| 17 | three | one | `count_17` |
| 18 | four | one | `count_18` |

Cycle the 10 control gestures (`thumbs_up, thumbs_down, stop, ok, call, rock, mute, fist, peace, open_palm`) with one hand visible first; then the 13 two-hand counts.

## Stage 6 — Evaluation and MV Visualization

Score the trained checkpoint:

```
python -m src.evaluate --ckpt runs/mlp_best.pt --splits-dir data/splits --labels data/labels.json --output-dir runs/evaluation --batch-size 512 --device cpu
```

Executes the MV-calculus notebook:

```
python -m jupyter nbconvert --to notebook --execute notebooks/03_mv_visualization.ipynb
```

Stage 6 artefacts (all under `runs/evaluation/`):

| File | Source | Purpose |
|---|---|---|
| `test_metrics.json` | `src/evaluate.py` | headline metrics + acceptance gates + per-class + 26×26 CM + latency |
| `per_class_metrics.csv` | `src/evaluate.py` | 10-column per-class breakdown |
| `confusion_matrix.png` / `confusion_matrix_normalized.png` | `src/evaluate.py` | raw + row-normalized matrices |
| `predictions.csv` | `src/evaluate.py` | per-sample top-1 / top-2 |
| `latency.csv` | `src/evaluate.py` | per-class single-sample latency |
| `loss_surface_slice.png` | notebook 03 | 50×50 grid over two scalar entries of `linears[0].weight` |
| `grad_norm_vs_epoch.png` | notebook 03 | `||∇L||` decay across training |
| `pca_input_space.png` | notebook 03 | test set projected onto PC1, PC2 |
| `chain_rule_verification.csv` | notebook 03 | 12 samples, manual vs autograd gradient at one layer-1 weight |

Gates: `merged_accuracy ≥ 0.90`, `macro_f1 ≥ 0.88`, ≥ 10 chain-rule samples passing at tol `1e-5`.

## Notebooks

All three execute end-to-end:

```
python -m jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/02_training_analysis.ipynb
python -m jupyter nbconvert --to notebook --execute notebooks/03_mv_visualization.ipynb
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | class distribution before/after oversampling, leakage verification, PCA scatter on 5k training subsample, feature sanity checks |
| `02_training_analysis.ipynb` | train/val loss + accuracy curves, gradient-norm trajectory, per-layer Frobenius weight norms, MLP-vs-baseline table, checkpoint reload |
| `03_mv_visualization.ipynb` | loss-surface slice, gradient-norm plot, PCA, chain-rule verification |

## Results

Headline numbers come from `runs/evaluation/test_metrics.json` after running `python -m src.evaluate`. The 26-class schema removes the historical `peace == count_2` / `open_palm == count_5` collision, so `raw_accuracy == merged_accuracy` and the raw number is the headline. See [tasks/peace_count2_collision_fix.md](tasks/peace_count2_collision_fix.md) for the migration history (the prior 28-class checkpoint sat at raw 0.82 / merged 0.99 because of the duplicate labels).

| Metric | Source key |
|---|---|
| Test accuracy | `metrics.accuracy` |
| Macro F1 | `metrics.macro_f1` |
| Weighted F1 | `metrics.weighted_f1` |
| Single-sample latency (mean / p95) | `latency.single_sample.mean_ms` / `p95_ms` |
| Batch latency (mean / p95, batch 512) | `latency.batch.mean_ms` / `p95_ms` |
| Chain-rule samples passed | `runs/evaluation/chain_rule_verification.csv` |

Baselines for reference live in `runs/baselines.csv`. After the schema fix, Logistic Regression hits val accuracy ≈ 0.99 and RBF SVM (best `C`) is comparable; the MLP wins on macro F1.

Results depend on the HaGRIDv2 subset used to train this checkpoint, the seed (`20260514`), and — for real-time inference — local webcam conditions. Reproducing on a different machine or seed may shift numbers slightly.

## Troubleshooting

- **`ImportError: mediapipe`** — `pip install -r requirements.txt` again; some platforms need an older Python (3.10 or 3.11) for the MediaPipe wheel.
- **`smoke_test_mediapipe.py` fails to load `hand_landmarker.task`** — re-download the file from the MediaPipe model card; do not commit a corrupted local copy.
- **`runs/baselines.csv` missing when running `src/train.py`** — Stage 4 refuses to start without both baseline rows; run `python -m src.models.baseline train-all` first.
- **Stage 5 shows `---` constantly** — low light or hand too far from camera causes MediaPipe to drop detections, or the gesture isn't one of the 26 trained classes. Both manifest as confidence < 0.75.
- **Stage 5 mislabels two-hand counts** — hold both hands clearly separated; partial occlusion flips MediaPipe handedness and the feature assembler will populate the wrong slot.
- **Slow tests fail** — `tests/test_dataset.py` and `tests/test_mv_visualization.py` have `@pytest.mark.slow` tests that touch `data/splits/` and `runs/mlp_best.pt`; skip them on a clean checkout with `pytest -q -k "not slow"`.

## Mathematical Narrative

The multivariable calculus reading of this system — input space `R^279`, function composition `f = f_4 ∘ f_3 ∘ f_2 ∘ f_1`, Jacobians at each layer, backpropagation as the chain rule (with one numerically verified worked example), and loss-surface geometry — is in [MATH.md](MATH.md).

## License

This project is released under the MIT License — see [LICENSE](LICENSE). The vendored HaGRIDv2 submodule under `hagrid_repo/` is governed by its own license (see [hukenovs/hagrid](https://github.com/hukenovs/hagrid)).
