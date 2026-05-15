"""Stage 4 training loop for the gesture-recognition MLP.

Loads Stage 2 splits, standardizes features with a ``StandardScaler`` fit on
train only, trains :class:`src.models.mlp.GestureMLP` with Adam +
``ReduceLROnPlateau``, early-stops on validation loss, and writes per-epoch
metrics to ``runs/training_log.csv`` plus a best-by-val-loss checkpoint to
``runs/mlp_best.pt``.

Authoritative spec:
    - tasks/gesture_recognition_plan_v2.md §6.2 - §6.4
    - tasks/implementation_stages.md Stage 4
    - tasks/stage4_handoff.md
    - plan at C:/Users/Harry T/.claude/plans/think-carefully-before-planning-cached-hartmanis.md

Re-uses helpers from ``src.models.baseline``:
    - ``DEFAULT_SEED`` (also in ``src.dataset``)
    - ``LABEL_EQUIVALENCE`` and ``_merge_labels`` for the merged-accuracy
      reporting required by Stage 4's pass criterion
    - ``load_split`` + ``validate_split_schema`` for IO and schema checks
    - ``load_label_ids`` for the integer-to-name map saved in the checkpoint

CLI::

    python -m src.train [--epochs 400] [--batch-size 64] [--lr 1e-3]
                        [--weight-decay 1e-4] [--patience 25] [--min-epochs 100]
                        [--scheduler-patience 10] [--scheduler-factor 0.5]
                        [--min-lr 1e-5] [--seed 20260514]
                        [--device auto|cpu|cuda|xpu] [--num-workers 0]
                        [--splits-dir data/splits] [--runs-dir runs]
                        [--ckpt runs/mlp_best.pt] [--log-csv runs/training_log.csv]
                        [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import DEFAULT_SEED
from src.models.baseline import (
    LABEL_EQUIVALENCE,
    _merge_labels,
    load_label_ids,
    load_split,
    validate_split_schema,
)
from src.models.mlp import DROPOUTS, HIDDEN_DIMS, INPUT_DIM, NUM_CLASSES, GestureMLP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_DIM = INPUT_DIM
EXPECTED_NUM_CLASSES = NUM_CLASSES

SPLITS_DIR_DEFAULT = Path("data/splits")
RUNS_DIR_DEFAULT = Path("runs")
TRAINING_LOG_DEFAULT = RUNS_DIR_DEFAULT / "training_log.csv"
CHECKPOINT_DEFAULT = RUNS_DIR_DEFAULT / "mlp_best.pt"
BASELINES_CSV_DEFAULT = RUNS_DIR_DEFAULT / "baselines.csv"
LABELS_JSON_DEFAULT = Path("data/labels.json")

ACCEPTANCE_GATE_MERGED_VAL_ACC = 0.93

# Frozen CSV column order. Stage 6's plots index by these names.
CSV_FIELDS: list[str] = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "merged_train_acc",
    "merged_val_acc",
    "val_macro_f1",
    "grad_norm",
    "layer_1_weight_norm",
    "layer_2_weight_norm",
    "layer_3_weight_norm",
    "layer_4_weight_norm",
    "lr",
    "wall_seconds",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Device + seeds
# ---------------------------------------------------------------------------


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve ``requested`` to a concrete ``torch.device``.

    ``"auto"`` prefers Intel XPU (PyTorch native XPU support), then CUDA, then
    CPU. Otherwise honors the explicit request.
    """
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
        return torch.device("xpu:0")
    if torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_global_seeds(seed: int) -> torch.Generator:
    """Seed ``random``, NumPy, and PyTorch (CPU + CUDA + XPU). Return a seeded
    ``torch.Generator`` for the train ``DataLoader``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
        torch.xpu.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_dataloaders(
    splits_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_batch_size: int = 512,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Return ``(train_loader, val_loader, fitted_scaler)``.

    - Loads ``train.npz`` and ``val.npz`` via ``baseline.load_split`` (so the
      result is ``float32`` X / ``int32`` y).
    - Validates the Stage 2 schema (dim=279, finite, labels in labels.json).
    - Fits ``StandardScaler`` on train only; transforms both.
    - Wraps in ``TensorDataset`` with ``y`` cast to ``int64``.
    - Train loader is shuffled with a seeded ``torch.Generator``; val loader
      is unshuffled with a larger batch for cheaper eval.
    """
    X_train, y_train = load_split("train", splits_dir)
    X_val, y_val = load_split("val", splits_dir)
    validate_split_schema(X_train, y_train)
    validate_split_schema(X_val, y_val)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32, copy=False)
    X_val_s = scaler.transform(X_val).astype(np.float32, copy=False)

    y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
    y_val_t = torch.from_numpy(y_val.astype(np.int64, copy=False))
    X_train_t = torch.from_numpy(X_train_s)
    X_val_t = torch.from_numpy(X_val_s)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(wid: int) -> None:
        np.random.seed(seed + wid)
        random.seed(seed + wid)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader, scaler


# ---------------------------------------------------------------------------
# Model / optimizer / scheduler factories
# ---------------------------------------------------------------------------


def make_model(seed: int) -> GestureMLP:
    """Construct ``GestureMLP`` deterministically.

    Re-seeding here ensures the model parameters are identical across reruns
    even if upstream RNG state was perturbed.
    """
    torch.manual_seed(seed)
    return GestureMLP()


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Adam:
    """``Adam(model.parameters(), lr, weight_decay)`` per plan §6.2."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    opt: torch.optim.Optimizer,
    patience: int,
    factor: float,
    min_lr: float,
) -> ReduceLROnPlateau:
    """``ReduceLROnPlateau(mode='min', ...)`` on val loss per plan §6.2."""
    return ReduceLROnPlateau(
        opt,
        mode="min",
        patience=patience,
        factor=factor,
        min_lr=min_lr,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy as a Python float."""
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of the flattened gradient over all trainable parameters.

    Walks ``model.parameters()`` once; accumulates per-tensor squared sums.
    Returns ``0.0`` when no parameter has a ``.grad`` yet (e.g. before the
    first backward()). Numerically equivalent to::

        torch.cat([p.grad.detach().flatten()
                   for p in model.parameters() if p.grad is not None]).norm()
    """
    total = torch.zeros((), dtype=torch.float64)
    found = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach().to(dtype=torch.float64)
            total += g.pow(2).sum()
            found = True
    if not found:
        return 0.0
    return float(total.sqrt().item())


def compute_layer_weight_norms(model: GestureMLP) -> list[float]:
    """Frobenius norm of each ``nn.Linear.weight`` in forward order.

    Length 4. ``[0]`` is ``Linear(279, 256)``; ``[3]`` is ``Linear(64, 28)``.
    """
    return [float(L.weight.detach().norm().item()) for L in model.linear_layers]


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: GestureMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """One pass over ``loader``. Returns
    ``{'loss', 'acc', 'merged_acc', 'grad_norm'}``.

    ``grad_norm`` is the L2 norm of all parameter gradients **after the last
    batch's ``backward()``** but **before** ``optimizer.step()`` — this is the
    convention required for the Stage 6 ``grad_norm`` plot (gradient at the
    parameter location the next epoch starts from).
    """
    model.train()
    total_loss = 0.0
    total_n = 0
    total_correct = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    last_grad_norm = 0.0
    n_batches = len(loader)
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=False)
        yb = yb.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at batch {i}/{n_batches}: {loss.item()}"
            )
        loss.backward()

        if i == n_batches - 1:
            last_grad_norm = compute_grad_norm(model)

        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == yb).sum().item())
        total_n += bs
        y_true_chunks.append(yb.detach().cpu().numpy())
        y_pred_chunks.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
    merged_acc = (
        float(accuracy_score(_merge_labels(y_true), _merge_labels(y_pred)))
        if total_n > 0 else 0.0
    )

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
        "merged_acc": merged_acc,
        "grad_norm": last_grad_norm,
    }


def evaluate(
    model: GestureMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Full pass over ``loader`` in eval mode (BN running stats; dropout off).

    Returns ``{'loss', 'acc', 'macro_f1', 'merged_acc', 'merged_macro_f1'}``.
    """
    model.eval()
    total_loss = 0.0
    total_n = 0
    total_correct = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=-1)

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((preds == yb).sum().item())
            total_n += bs
            y_true_chunks.append(yb.detach().cpu().numpy())
            y_pred_chunks.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_n > 0 else 0.0
    yt_m = _merge_labels(y_true)
    yp_m = _merge_labels(y_pred)
    merged_acc = float(accuracy_score(yt_m, yp_m)) if total_n > 0 else 0.0
    merged_macro_f1 = float(f1_score(yt_m, yp_m, average="macro", zero_division=0)) if total_n > 0 else 0.0

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
        "macro_f1": macro_f1,
        "merged_acc": merged_acc,
        "merged_macro_f1": merged_macro_f1,
    }


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def append_log_row(
    csv_path: Path, row: dict, fields: list[str] = CSV_FIELDS
) -> None:
    """Append one epoch row to ``csv_path``; write header on first write.

    Mirrors :func:`src.models.baseline.write_baseline_metrics`: parent dir
    created automatically, ``DictWriter(extrasaction="ignore")``, raises if
    any required field is missing.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    missing = [k for k in fields if k not in row]
    if missing:
        raise ValueError(f"row missing required keys {missing}: {row}")
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: GestureMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: StandardScaler,
    epoch: int,
    metrics: dict,
    config: dict,
    seed: int,
    labels: dict | None = None,
) -> None:
    """Persist a full training state dict to ``path``.

    Payload keys (stage4_handoff.md §6.2): ``model_state_dict``,
    ``optimizer_state_dict``, ``scheduler_state_dict``, ``epoch``,
    ``val_loss``, ``val_acc``, ``merged_val_acc``, ``macro_f1``,
    ``scaler_mean``, ``scaler_scale``, ``config``, ``seed``, ``labels``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "val_loss": float(metrics.get("val_loss", float("nan"))),
        "val_acc": float(metrics.get("val_acc", float("nan"))),
        "merged_val_acc": float(metrics.get("merged_val_acc", float("nan"))),
        "macro_f1": float(metrics.get("val_macro_f1", float("nan"))),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float32).tolist(),
        "config": dict(config),
        "seed": int(seed),
        "labels": dict(labels) if labels is not None else None,
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def _parse_merged_acc_from_notes(notes: str) -> float | None:
    """Extract ``merged_accuracy=<float>`` from a ``runs/baselines.csv`` row's
    ``notes`` field. Returns ``None`` if absent."""
    if not isinstance(notes, str):
        return None
    for part in notes.split(";"):
        part = part.strip()
        if part.startswith("merged_accuracy="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def load_baseline_metrics(path: Path = BASELINES_CSV_DEFAULT) -> dict[str, dict]:
    """Read ``runs/baselines.csv`` and return ``{model_name: {acc, macro_f1,
    merged_acc}}`` using the latest row per model.

    Raises :class:`RuntimeError` if the file is missing or empty — the
    Stage 4 acceptance gate requires beating both Stage 3 baselines, so this
    is treated as a blocker rather than silently skipping the comparison.
    """
    if not path.is_file():
        raise RuntimeError(
            f"Stage 3 baselines not found at {path} — "
            f"run `python -m src.models.baseline train-all` first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty; Stage 3 baselines must be available.")
    out: dict[str, dict] = {}
    for model_name in ("logistic_regression", "svm_rbf"):
        rows = df[df["model"] == model_name]
        if rows.empty:
            continue
        latest = rows.iloc[-1]
        out[model_name] = {
            "acc": float(latest["accuracy"]),
            "macro_f1": float(latest["macro_f1"]),
            "merged_acc": _parse_merged_acc_from_notes(latest.get("notes", "")),
        }
    if not out:
        raise RuntimeError(
            f"{path} has no logistic_regression / svm_rbf rows; Stage 3 must run first."
        )
    return out


def _baseline_table(mlp_metrics: dict, baselines: dict[str, dict]) -> str:
    """Pretty comparison table for the end-of-training summary."""
    lr = baselines.get("logistic_regression", {})
    sv = baselines.get("svm_rbf", {})

    def _fmt(x: object) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "  n/a "
        return f"{float(x):.4f}"

    lines = [
        "Stage 4 vs Stage 3 baselines (val split)",
        f"  {'metric':<14}{'MLP':>10}{'LR':>10}{'SVM':>10}",
        f"  {'raw_acc':<14}{_fmt(mlp_metrics.get('val_acc')):>10}"
        f"{_fmt(lr.get('acc')):>10}{_fmt(sv.get('acc')):>10}",
        f"  {'macro_f1':<14}{_fmt(mlp_metrics.get('val_macro_f1')):>10}"
        f"{_fmt(lr.get('macro_f1')):>10}{_fmt(sv.get('macro_f1')):>10}",
        f"  {'merged_acc':<14}{_fmt(mlp_metrics.get('merged_val_acc')):>10}"
        f"{_fmt(lr.get('merged_acc')):>10}{_fmt(sv.get('merged_acc')):>10}",
    ]
    return "\n".join(lines)


def _gate_message(mlp_metrics: dict, baselines: dict[str, dict]) -> tuple[str, bool]:
    """Build the GATE PASSED/FAILED summary string and a bool."""
    merged = mlp_metrics.get("merged_val_acc", float("nan"))
    raw = mlp_metrics.get("val_acc", float("nan"))
    f1 = mlp_metrics.get("val_macro_f1", float("nan"))
    lr_acc = baselines.get("logistic_regression", {}).get("acc", float("nan"))
    sv_acc = baselines.get("svm_rbf", {}).get("acc", float("nan"))
    lr_f1 = baselines.get("logistic_regression", {}).get("macro_f1", float("nan"))
    sv_f1 = baselines.get("svm_rbf", {}).get("macro_f1", float("nan"))
    base_acc = max(v for v in (lr_acc, sv_acc) if not math.isnan(v)) if (lr_acc or sv_acc) else float("nan")
    base_f1 = max(v for v in (lr_f1, sv_f1) if not math.isnan(v)) if (lr_f1 or sv_f1) else float("nan")

    merged_pass = merged >= ACCEPTANCE_GATE_MERGED_VAL_ACC
    beats_acc = (not math.isnan(base_acc)) and raw > base_acc
    beats_f1 = (not math.isnan(base_f1)) and f1 > base_f1
    all_pass = bool(merged_pass and beats_acc and beats_f1)
    tag = "GATE PASSED" if all_pass else "GATE FAILED"
    msg = (
        f"{tag}: merged_val_acc={merged:.4f} (>= {ACCEPTANCE_GATE_MERGED_VAL_ACC}: "
        f"{merged_pass}); raw_val_acc={raw:.4f} > max(LR,SVM)={base_acc:.4f}: "
        f"{beats_acc}; macro_f1={f1:.4f} > max(LR,SVM)={base_f1:.4f}: {beats_f1}"
    )
    return msg, all_pass


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict:
    """Full training run. Returns a dict of final metrics + paths."""
    seed = int(args.seed)
    set_global_seeds(seed)

    device = pick_device(args.device)
    print(f"[train] device={device}", flush=True)

    # Confirm Stage 3 baselines are available before spending time training.
    baselines = load_baseline_metrics(args.baselines_csv)
    print(
        f"[train] baselines loaded: "
        f"LR acc={baselines['logistic_regression']['acc']:.4f}, "
        f"SVM acc={baselines['svm_rbf']['acc']:.4f}",
        flush=True,
    )

    train_loader, val_loader, scaler = build_dataloaders(
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )
    if args.smoke:
        # Smoke run: cap epochs at 3, but use the full splits so plumbing is
        # exercised end-to-end against real data shapes.
        args.max_epochs = min(args.max_epochs, 3)
        args.min_epochs_before_early_stop = 0
        print("[train] SMOKE MODE: max_epochs=3, early-stop warm-up disabled", flush=True)

    model = make_model(seed).to(device)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer,
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss()

    config = {
        "input_dim": INPUT_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropouts": list(DROPOUTS),
        "num_classes": NUM_CLASSES,
    }
    try:
        labels = {int(v): k for k, v in json.loads(LABELS_JSON_DEFAULT.read_text(encoding="utf-8")).items()} if LABELS_JSON_DEFAULT.is_file() else None
    except Exception:  # pragma: no cover
        labels = None

    # Reset training log to a clean file so smoke + full runs don't interleave.
    if args.log_csv.is_file():
        args.log_csv.unlink()

    print(
        f"[train] n_train={len(train_loader.dataset)} n_val={len(val_loader.dataset)} "
        f"batch_size={args.batch_size} max_epochs={args.max_epochs} "
        f"patience={args.patience} min_epochs={args.min_epochs_before_early_stop} "
        f"lr={args.lr} weight_decay={args.weight_decay} seed={seed}",
        flush=True,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    patience_counter = 0
    t_start = time.perf_counter()

    for epoch in range(args.max_epochs):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step(va["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])
        layer_norms = compute_layer_weight_norms(model)
        wall = time.perf_counter() - t_start

        row = {
            "epoch": int(epoch),
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "train_acc": float(tr["acc"]),
            "val_acc": float(va["acc"]),
            "merged_train_acc": float(tr["merged_acc"]),
            "merged_val_acc": float(va["merged_acc"]),
            "val_macro_f1": float(va["macro_f1"]),
            "grad_norm": float(tr["grad_norm"]),
            "layer_1_weight_norm": float(layer_norms[0]),
            "layer_2_weight_norm": float(layer_norms[1]),
            "layer_3_weight_norm": float(layer_norms[2]),
            "layer_4_weight_norm": float(layer_norms[3]),
            "lr": lr_now,
            "wall_seconds": float(wall),
            "timestamp": _now_iso(),
        }
        append_log_row(args.log_csv, row)

        print(
            f"epoch={epoch:3d}  train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"raw_acc={va['acc']:.4f}  merged_acc={va['merged_acc']:.4f}  "
            f"macro_f1={va['macro_f1']:.4f}  grad_norm={tr['grad_norm']:.4f}  lr={lr_now:.2e}",
            flush=True,
        )

        improved = va["loss"] < best_val_loss - 1e-6
        if improved:
            best_val_loss = va["loss"]
            best_epoch = epoch
            best_metrics = {
                "val_loss": float(va["loss"]),
                "val_acc": float(va["acc"]),
                "merged_val_acc": float(va["merged_acc"]),
                "val_macro_f1": float(va["macro_f1"]),
            }
            patience_counter = 0
            save_checkpoint(
                args.ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=best_metrics,
                config=config,
                seed=seed,
                labels=labels,
            )
        else:
            patience_counter += 1
            if (
                patience_counter >= args.patience
                and epoch >= args.min_epochs_before_early_stop
            ):
                print(
                    f"[train] early stop at epoch {epoch} "
                    f"(no val_loss improvement for {patience_counter} epochs; "
                    f"best epoch={best_epoch}, best val_loss={best_val_loss:.4f})",
                    flush=True,
                )
                break

    # End-of-training summary.
    print("", flush=True)
    print(_baseline_table(best_metrics, baselines), flush=True)
    msg, gate_passed = _gate_message(best_metrics, baselines)
    print(msg, flush=True)

    return {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_metrics.get("val_acc", float("nan"))),
        "best_merged_val_acc": float(best_metrics.get("merged_val_acc", float("nan"))),
        "best_macro_f1": float(best_metrics.get("val_macro_f1", float("nan"))),
        "gate_passed": bool(gate_passed),
        "ckpt": str(args.ckpt),
        "log_csv": str(args.log_csv),
        "baselines": baselines,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4: train the GestureMLP and log gradient/weight norms.",
    )
    p.add_argument("--epochs", dest="max_epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", dest="min_epochs_before_early_stop", type=int, default=100)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cpu", "cuda", "xpu"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    p.add_argument("--ckpt", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--log-csv", type=Path, default=TRAINING_LOG_DEFAULT)
    p.add_argument("--baselines-csv", type=Path, default=BASELINES_CSV_DEFAULT)
    p.add_argument("--smoke", action="store_true",
                   help="Cap epochs at 3 and disable early-stop warm-up.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
