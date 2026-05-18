"""Stage 6 test-set evaluation for the gesture-recognition MLP.

Loads ``data/splits/test.npz`` and ``runs/mlp_best.pt``, applies the same
``StandardScaler`` used during Stage 4 training, runs the model in eval mode,
and emits a scorecard under ``runs/evaluation/``:

    - test_metrics.json
    - per_class_metrics.csv
    - confusion_matrix.png  +  confusion_matrix_normalized.png
    - predictions.csv
    - latency.csv

Reports raw + merged metrics. After the 26-class schema correction (see
``tasks/peace_count2_collision_fix.md``) ``LABEL_EQUIVALENCE`` is empty and
``merged_accuracy == accuracy``. The Stage 6 acceptance gate is on
``merged_accuracy >= 0.90`` and ``macro_f1 >= 0.88``.

Authoritative spec:
    - tasks/gesture_recognition_plan_v2.md §6.4
    - tasks/implementation_stages.md Stage 6
    - plan at C:/Users/Harry T/.claude/plans/you-are-claude-opus-snappy-ritchie.md

CLI::

    python -m src.evaluate [--ckpt runs/mlp_best.pt] [--splits-dir data/splits]
                           [--labels data/labels.json] [--output-dir runs/evaluation]
                           [--batch-size 512] [--latency-repeats 50]
                           [--warmup-batches 3] [--device auto|cpu|cuda|xpu]
                           [--seed 20260514]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless rendering for CLI runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from src.models.baseline import (
    LABEL_EQUIVALENCE,
    _merge_labels,
    load_label_ids,
    load_split,
    validate_split_schema,
)
from src.models.mlp import INPUT_DIM, NUM_CLASSES, GestureMLP, assert_labels_consistent
from src.train import DEFAULT_SEED, pick_device, set_global_seeds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS_DIR_DEFAULT = Path("data/splits")
RUNS_DIR_DEFAULT = Path("runs")
EVAL_DIR_DEFAULT = RUNS_DIR_DEFAULT / "evaluation"
CHECKPOINT_DEFAULT = RUNS_DIR_DEFAULT / "mlp_best.pt"
LABELS_JSON_DEFAULT = Path("data/labels.json")

EXPECTED_FEATURE_DIM = INPUT_DIM
EXPECTED_NUM_CLASSES = NUM_CLASSES

ACCEPTANCE_TEST_ACC = 0.90  # gated on merged_accuracy
ACCEPTANCE_MACRO_F1 = 0.88

DEFAULT_BATCH_SIZE = 512
DEFAULT_LATENCY_REPEATS = 50
DEFAULT_WARMUP_BATCHES = 3
DEFAULT_SINGLE_SAMPLE_REPEATS = 2000
DEFAULT_SINGLE_SAMPLE_WARMUP = 20
PER_CLASS_LATENCY_CAP = 200  # max samples per class for per-class timing

SCHEMA_VERSION = 1

PER_CLASS_CSV_FIELDS: list[str] = [
    "label_id",
    "label_name",
    "support",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
]

PREDICTIONS_CSV_FIELDS: list[str] = [
    "sample_index",
    "y_true_id",
    "y_true_name",
    "y_pred_id",
    "y_pred_name",
    "correct",
    "top1_prob",
    "top2_id",
    "top2_prob",
]

LATENCY_CSV_FIELDS: list[str] = [
    "label_id",
    "label_name",
    "support",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "batch_size",
    "device",
]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_test_split(
    splits_dir: Path = SPLITS_DIR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` from ``data/splits/test.npz`` after schema validation."""
    X, y = load_split("test", splits_dir)
    validate_split_schema(X, y)
    return X, y


def validate_test_schema(
    X: np.ndarray,
    y: np.ndarray,
    expected_dim: int = EXPECTED_FEATURE_DIM,
    num_classes: int = EXPECTED_NUM_CLASSES,
) -> None:
    """Re-export of ``baseline.validate_split_schema`` for a direct test target."""
    validate_split_schema(X, y, expected_dim=expected_dim, num_classes=num_classes)


def _scaler_from_arrays(mean: np.ndarray, scale: np.ndarray) -> StandardScaler:
    """Reconstitute a fitted ``StandardScaler`` from the checkpoint arrays.

    Stage 4 stores ``scaler_mean`` and ``scaler_scale`` as Python lists alongside
    the model state; rebuilding the sklearn object here lets evaluation reuse the
    same ``.transform`` code path.
    """
    mean_arr = np.asarray(mean, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    if mean_arr.shape != (EXPECTED_FEATURE_DIM,):
        raise ValueError(
            f"scaler_mean length {mean_arr.shape} != ({EXPECTED_FEATURE_DIM},)"
        )
    if scale_arr.shape != (EXPECTED_FEATURE_DIM,):
        raise ValueError(
            f"scaler_scale length {scale_arr.shape} != ({EXPECTED_FEATURE_DIM},)"
        )
    sc = StandardScaler()
    sc.mean_ = mean_arr
    sc.scale_ = scale_arr
    sc.var_ = scale_arr ** 2
    sc.n_features_in_ = EXPECTED_FEATURE_DIM
    sc.n_samples_seen_ = 0
    return sc


def load_model_checkpoint(
    checkpoint_path: Path = CHECKPOINT_DEFAULT,
    labels_path: Path = LABELS_JSON_DEFAULT,
    device: torch.device | None = None,
) -> tuple[GestureMLP, StandardScaler, dict[int, str], dict]:
    """Load the Stage 4 checkpoint and return a ready-to-evaluate bundle.

    Returns ``(model, scaler, label_map, raw_ckpt)``. ``model`` is in eval mode
    on ``device``; ``scaler`` exposes the same ``.transform`` as the one fit in
    ``src.train.build_dataloaders``; ``label_map`` is ``{int -> name}``.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {checkpoint_path}. "
            f"Run `python -m src.train` first."
        )
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ck.get("config", {})
    if int(cfg.get("input_dim", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"checkpoint config.input_dim={cfg.get('input_dim')} != {EXPECTED_FEATURE_DIM}"
        )
    if int(cfg.get("num_classes", -1)) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"checkpoint config.num_classes={cfg.get('num_classes')} != {EXPECTED_NUM_CLASSES}"
        )

    model = GestureMLP(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropouts=tuple(cfg["dropouts"]),
        num_classes=int(cfg["num_classes"]),
    )
    model.load_state_dict(ck["model_state_dict"])
    if device is None:
        device = pick_device("auto")
    model.to(device)
    model.eval()

    scaler = _scaler_from_arrays(ck["scaler_mean"], ck["scaler_scale"])

    if ck.get("labels") is not None:
        label_map = {int(k): str(v) for k, v in ck["labels"].items()}
    else:
        if not labels_path.is_file():
            raise FileNotFoundError(f"labels.json not found at {labels_path}")
        with labels_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        label_map = {int(v): str(k) for k, v in raw.items()}

    # Sanity: every id 0..N-1 must be in the map.
    missing = sorted(set(range(EXPECTED_NUM_CLASSES)) - set(label_map.keys()))
    if missing:
        raise ValueError(f"label_map missing ids: {missing}")

    return model, scaler, label_map, ck


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_batches(
    model: GestureMLP,
    X_std: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(preds, probs)`` from a forward pass over already-scaled ``X``.

    Caller is responsible for standardization via the checkpoint's scaler.
    Runs in eval mode under ``torch.inference_mode()``.
    """
    if device is None:
        device = next(model.parameters()).device
    if X_std.dtype != np.float32:
        X_std = X_std.astype(np.float32, copy=False)
    model.eval()

    n = X_std.shape[0]
    preds = np.empty((n,), dtype=np.int64)
    probs = np.empty((n, EXPECTED_NUM_CLASSES), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(X_std[start:end]).to(device)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"non-finite logits in batch [{start}:{end}); checkpoint may be corrupt"
            )
        p = F.softmax(logits, dim=-1)
        preds[start:end] = logits.argmax(dim=-1).detach().cpu().numpy()
        probs[start:end] = p.detach().cpu().numpy()

    return preds, probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_ids: list[int]
) -> dict:
    """Raw + merged accuracy / macro / weighted F1, per-class breakdown, CM."""
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).astype(np.int64, copy=False)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    row_sums = cm.sum(axis=1)
    per_class_accuracy: list[float | None] = []
    for i in range(len(label_ids)):
        if row_sums[i] == 0:
            per_class_accuracy.append(None)
        else:
            per_class_accuracy.append(float(cm[i, i]) / float(row_sums[i]))

    yt_m = _merge_labels(y_true)
    yp_m = _merge_labels(y_pred)
    merged_ids = sorted({int(v) for v in LABEL_EQUIVALENCE.values()} |
                        (set(int(v) for v in np.unique(yt_m).tolist())) |
                        (set(int(v) for v in np.unique(yp_m).tolist())))
    merged_accuracy = float(accuracy_score(yt_m, yp_m))
    merged_macro_f1 = float(
        f1_score(yt_m, yp_m, labels=merged_ids, average="macro", zero_division=0)
    )
    merged_weighted_f1 = float(
        f1_score(yt_m, yp_m, labels=merged_ids, average="weighted", zero_division=0)
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "merged_accuracy": merged_accuracy,
        "merged_macro_f1": merged_macro_f1,
        "merged_weighted_f1": merged_weighted_f1,
        "per_class_accuracy": per_class_accuracy,
        "per_class_support": [int(s) for s in support],
        "per_class_precision": [float(p) for p in precision],
        "per_class_recall": [float(r) for r in recall],
        "per_class_f1": [float(v) for v in f1],
        "confusion_matrix": cm.astype(np.int64),
        "label_ids": list(label_ids),
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":  # pragma: no cover - depends on hardware
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):  # pragma: no cover
        try:
            torch.xpu.synchronize()
        except Exception:
            pass


@torch.inference_mode()
def compute_prediction_latency(
    model: GestureMLP,
    X_std: np.ndarray,
    y: np.ndarray,
    label_map: dict[int, str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    repeats: int = DEFAULT_LATENCY_REPEATS,
    warmup: int = DEFAULT_WARMUP_BATCHES,
    device: torch.device | None = None,
    seed: int = DEFAULT_SEED,
    single_sample_repeats: int = DEFAULT_SINGLE_SAMPLE_REPEATS,
    single_sample_warmup: int = DEFAULT_SINGLE_SAMPLE_WARMUP,
    per_class_cap: int = PER_CLASS_LATENCY_CAP,
) -> dict:
    """Measure batch + single-sample + per-class forward-pass latency.

    Excludes disk IO, scaling, ``.to(device)`` of inputs (one-shot move), and
    argmax. Returns the latency JSON fragment plus per-class rows.
    """
    if device is None:
        device = next(model.parameters()).device
    if X_std.dtype != np.float32:
        X_std = X_std.astype(np.float32, copy=False)

    n = X_std.shape[0]
    if n == 0:
        raise ValueError("cannot measure latency on empty input")

    rng = np.random.default_rng(seed)
    model.eval()

    # ---- Batch latency ---------------------------------------------------
    bs = min(batch_size, n)
    # One-shot move of a single batch buffer; we time only the forward pass.
    sample_indices = rng.integers(0, n, size=bs)
    xb = torch.from_numpy(X_std[sample_indices]).to(device)

    # Warmup.
    for _ in range(max(0, warmup)):
        _ = model(xb)
    _synchronize(device)

    batch_times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        _synchronize(device)
        t0 = time.perf_counter_ns()
        _ = model(xb)
        _synchronize(device)
        t1 = time.perf_counter_ns()
        batch_times_ms.append((t1 - t0) / 1e6)

    batch_arr = np.asarray(batch_times_ms, dtype=np.float64)
    batch_summary = {
        "batch_size": int(bs),
        "mean_ms": float(batch_arr.mean()),
        "median_ms": float(np.median(batch_arr)),
        "p95_ms": float(np.percentile(batch_arr, 95)),
        "per_sample_mean_ms": float(batch_arr.mean() / max(bs, 1)),
        "n_repeats": int(len(batch_arr)),
        "warmup_batches": int(warmup),
    }

    # ---- Single-sample latency ------------------------------------------
    single_idx = rng.integers(0, n, size=1)[0]
    x_single = torch.from_numpy(X_std[single_idx : single_idx + 1]).to(device)
    for _ in range(max(0, single_sample_warmup)):
        _ = model(x_single)
    _synchronize(device)
    single_times_ms: list[float] = []
    for _ in range(max(1, single_sample_repeats)):
        _synchronize(device)
        t0 = time.perf_counter_ns()
        _ = model(x_single)
        _synchronize(device)
        t1 = time.perf_counter_ns()
        single_times_ms.append((t1 - t0) / 1e6)
    single_arr = np.asarray(single_times_ms, dtype=np.float64)
    single_summary = {
        "mean_ms": float(single_arr.mean()),
        "median_ms": float(np.median(single_arr)),
        "p95_ms": float(np.percentile(single_arr, 95)),
        "n_repeats": int(len(single_arr)),
        "warmup_batches": int(single_sample_warmup),
    }

    # ---- Per-class latency ----------------------------------------------
    per_class_rows: list[dict] = []
    for cls_id in sorted(label_map.keys()):
        mask = (y == cls_id)
        support_total = int(mask.sum())
        cls_idx = np.where(mask)[0]
        if cls_idx.size == 0:
            per_class_rows.append({
                "label_id": int(cls_id),
                "label_name": label_map[cls_id],
                "support": 0,
                "mean_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "batch_size": 1,
                "device": str(device),
            })
            continue

        if cls_idx.size > per_class_cap:
            cls_idx = rng.choice(cls_idx, size=per_class_cap, replace=False)

        times_ms: list[float] = []
        for idx in cls_idx:
            xs = torch.from_numpy(X_std[idx : idx + 1]).to(device)
            _synchronize(device)
            t0 = time.perf_counter_ns()
            _ = model(xs)
            _synchronize(device)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)
        arr = np.asarray(times_ms, dtype=np.float64)
        mean_ms = float(arr.mean())
        median_ms = float(np.median(arr)) if arr.size >= 5 else None
        p95_ms = float(np.percentile(arr, 95)) if arr.size >= 5 else None
        per_class_rows.append({
            "label_id": int(cls_id),
            "label_name": label_map[cls_id],
            "support": int(support_total),
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "batch_size": 1,
            "device": str(device),
        })

    return {
        "device": str(device),
        "n_samples": int(n),
        "batch": batch_summary,
        "single_sample": single_summary,
        "per_class": per_class_rows,
    }


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------


def save_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_path: Path,
    normalize: bool = False,
    title: str = "Test Confusion Matrix",
) -> None:
    """Render a 28x28 confusion matrix as PNG (raw counts or row-normalised)."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square; got shape {cm.shape}")
    if len(label_names) != cm.shape[0]:
        raise ValueError(
            f"label_names length {len(label_names)} != cm.shape[0] {cm.shape[0]}"
        )

    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
            data = np.where(row_sums > 0, cm.astype(np.float64) / row_sums, 0.0)
        fmt = "{:.2f}"
        cmap = "Blues"
    else:
        data = cm
        fmt = "{:d}"
        cmap = "Blues"

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))
    im = ax.imshow(data, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(label_names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells; skip zero cells in the normalised view for legibility.
    threshold = data.max() / 2.0 if data.size and data.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if normalize and v == 0:
                continue
            if not normalize and v == 0:
                continue
            ax.text(
                j, i, fmt.format(v),
                ha="center", va="center", fontsize=6,
                color="white" if v > threshold else "black",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _to_json_safe(obj):
    """Convert numpy / torch scalars + arrays into plain Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def write_metrics_json(metrics: dict, output_path: Path) -> None:
    """Atomic write of the headline metrics JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_json_safe(metrics)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_path.parent),
        prefix=".tmp_metrics_",
        suffix=".json",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=False, allow_nan=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, output_path)


def write_per_class_csv(per_class_rows: list[dict], output_path: Path) -> None:
    """Write the per-class metrics + latency CSV with a frozen header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=PER_CLASS_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in per_class_rows:
            missing = [k for k in PER_CLASS_CSV_FIELDS if k not in row]
            if missing:
                raise ValueError(
                    f"per-class row missing required keys {missing}: {row}"
                )
            out = dict(row)
            # Blank-out None for CSV friendliness.
            for k, v in list(out.items()):
                if v is None:
                    out[k] = ""
            writer.writerow(out)


def write_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_map: dict[int, str],
    output_path: Path,
) -> None:
    """Stream per-sample predictions to disk."""
    if probs.ndim != 2 or probs.shape[1] != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"probs shape {probs.shape} != (n, {EXPECTED_NUM_CLASSES})"
        )
    n = y_true.shape[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top2 = np.argsort(-probs, axis=1)[:, :2]
    top1_id = top2[:, 0]
    top2_id = top2[:, 1]
    top1_prob = probs[np.arange(n), top1_id]
    top2_prob = probs[np.arange(n), top2_id]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=PREDICTIONS_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for i in range(n):
            yt = int(y_true[i])
            yp = int(y_pred[i])
            writer.writerow({
                "sample_index": int(i),
                "y_true_id": yt,
                "y_true_name": label_map.get(yt, str(yt)),
                "y_pred_id": yp,
                "y_pred_name": label_map.get(yp, str(yp)),
                "correct": int(yp == yt),
                "top1_prob": float(top1_prob[i]),
                "top2_id": int(top2_id[i]),
                "top2_prob": float(top2_prob[i]),
            })


def write_latency_csv(per_class_rows: list[dict], output_path: Path) -> None:
    """Write per-class latency rows (one per label id, including support=0)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=LATENCY_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in per_class_rows:
            out = dict(row)
            for k, v in list(out.items()):
                if v is None:
                    out[k] = ""
            writer.writerow(out)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gate_message(merged_acc: float, macro_f1: float) -> tuple[str, bool]:
    acc_pass = merged_acc >= ACCEPTANCE_TEST_ACC
    f1_pass = macro_f1 >= ACCEPTANCE_MACRO_F1
    all_pass = bool(acc_pass and f1_pass)
    tag = "GATE PASSED" if all_pass else "GATE FAILED"
    msg = (
        f"{tag}: merged_test_acc={merged_acc:.4f} (>= {ACCEPTANCE_TEST_ACC}: {acc_pass}); "
        f"macro_f1={macro_f1:.4f} (>= {ACCEPTANCE_MACRO_F1}: {f1_pass})"
    )
    return msg, all_pass


def run_evaluation(
    checkpoint_path: Path = CHECKPOINT_DEFAULT,
    splits_dir: Path = SPLITS_DIR_DEFAULT,
    labels_path: Path = LABELS_JSON_DEFAULT,
    output_dir: Path = EVAL_DIR_DEFAULT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    latency_repeats: int = DEFAULT_LATENCY_REPEATS,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    device: str = "auto",
    seed: int = DEFAULT_SEED,
    measure_latency: bool = True,
) -> dict:
    """Full Stage 6 evaluation pipeline. Returns the metrics payload."""
    set_global_seeds(int(seed))
    assert_labels_consistent(labels_path)
    dev = pick_device(device)
    print(f"[evaluate] device={dev}", flush=True)

    model, scaler, label_map, ck = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        labels_path=labels_path,
        device=dev,
    )
    print(
        f"[evaluate] checkpoint epoch={ck['epoch']} "
        f"val_loss={ck['val_loss']:.4f} val_acc={ck['val_acc']:.4f} "
        f"merged_val_acc={ck['merged_val_acc']:.4f}",
        flush=True,
    )

    X, y = load_test_split(splits_dir)
    print(
        f"[evaluate] test rows={X.shape[0]} feature_dim={X.shape[1]} "
        f"classes={len(np.unique(y))}",
        flush=True,
    )

    X_std = scaler.transform(X).astype(np.float32, copy=False)
    preds, probs = predict_batches(model, X_std, batch_size=batch_size, device=dev)

    label_ids = list(range(EXPECTED_NUM_CLASSES))
    metrics = compute_classification_metrics(y, preds, label_ids)
    label_names = [label_map[i] for i in label_ids]

    # Latency (always per-class; row count = 28).
    if measure_latency:
        latency = compute_prediction_latency(
            model=model,
            X_std=X_std,
            y=y,
            label_map=label_map,
            batch_size=batch_size,
            repeats=latency_repeats,
            warmup=warmup_batches,
            device=dev,
            seed=int(seed),
        )
    else:
        latency = {
            "device": str(dev),
            "n_samples": int(X.shape[0]),
            "batch": None,
            "single_sample": None,
            "per_class": [
                {
                    "label_id": int(i),
                    "label_name": label_map[i],
                    "support": int((y == i).sum()),
                    "mean_ms": None,
                    "median_ms": None,
                    "p95_ms": None,
                    "batch_size": 1,
                    "device": str(dev),
                }
                for i in label_ids
            ],
        }

    # Build the per-class CSV rows (join metrics + latency).
    latency_by_id = {row["label_id"]: row for row in latency["per_class"]}
    per_class_rows: list[dict] = []
    for i, lid in enumerate(label_ids):
        lat = latency_by_id.get(lid, {})
        per_class_rows.append({
            "label_id": int(lid),
            "label_name": label_names[i],
            "support": int(metrics["per_class_support"][i]),
            "accuracy": (
                metrics["per_class_accuracy"][i]
                if metrics["per_class_accuracy"][i] is not None
                else None
            ),
            "precision": float(metrics["per_class_precision"][i]),
            "recall": float(metrics["per_class_recall"][i]),
            "f1": float(metrics["per_class_f1"][i]),
            "latency_mean_ms": lat.get("mean_ms"),
            "latency_median_ms": lat.get("median_ms"),
            "latency_p95_ms": lat.get("p95_ms"),
        })

    # Warnings.
    low_support = [
        int(lid) for i, lid in enumerate(label_ids)
        if metrics["per_class_support"][i] < 10
    ]
    warnings_list: list[str] = []
    if low_support:
        warnings_list.append(
            f"classes with support < 10: {low_support} "
            f"(synthetic two-hand counts; per-class metrics are noisy)"
        )
    # In the 26-class schema LABEL_EQUIVALENCE is empty and merged == raw; the
    # historical warning is no longer applicable. Kept the field for log
    # consumers that expect a `warnings` list.

    gate_msg, gate_passed = _gate_message(
        metrics["merged_accuracy"], metrics["macro_f1"]
    )

    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(ck["epoch"]),
        "checkpoint_val_loss": float(ck["val_loss"]),
        "checkpoint_val_acc": float(ck["val_acc"]),
        "checkpoint_merged_val_acc": float(ck["merged_val_acc"]),
        "checkpoint_macro_f1": float(ck.get("macro_f1", float("nan"))),
        "splits_dir": str(splits_dir),
        "labels_path": str(labels_path),
        "feature_dim": int(EXPECTED_FEATURE_DIM),
        "num_classes": int(EXPECTED_NUM_CLASSES),
        "n_samples": int(X.shape[0]),
        "device": str(dev),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "merged_accuracy": metrics["merged_accuracy"],
            "merged_macro_f1": metrics["merged_macro_f1"],
            "merged_weighted_f1": metrics["merged_weighted_f1"],
        },
        "per_class": [
            {
                "label_id": int(lid),
                "label_name": label_names[i],
                "support": int(metrics["per_class_support"][i]),
                "accuracy": metrics["per_class_accuracy"][i],
                "precision": float(metrics["per_class_precision"][i]),
                "recall": float(metrics["per_class_recall"][i]),
                "f1": float(metrics["per_class_f1"][i]),
            }
            for i, lid in enumerate(label_ids)
        ],
        "confusion_matrix": metrics["confusion_matrix"].astype(np.int64).tolist(),
        "label_ids": label_ids,
        "label_names": label_names,
        "acceptance": {
            "test_accuracy_gate": ACCEPTANCE_TEST_ACC,
            "macro_f1_gate": ACCEPTANCE_MACRO_F1,
            "test_accuracy_passed": bool(
                metrics["merged_accuracy"] >= ACCEPTANCE_TEST_ACC
            ),
            "macro_f1_passed": bool(metrics["macro_f1"] >= ACCEPTANCE_MACRO_F1),
            "all_passed": gate_passed,
            "gate_message": gate_msg,
        },
        "warnings": warnings_list,
        "latency": {
            "device": latency["device"],
            "batch": latency.get("batch"),
            "single_sample": latency.get("single_sample"),
            "per_class_summary": str(output_dir / "latency.csv"),
        },
    }

    # Write all artefacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    per_class_path = output_dir / "per_class_metrics.csv"
    cm_path = output_dir / "confusion_matrix.png"
    cm_norm_path = output_dir / "confusion_matrix_normalized.png"
    preds_path = output_dir / "predictions.csv"
    latency_path = output_dir / "latency.csv"

    write_metrics_json(metrics_payload, metrics_path)
    write_per_class_csv(per_class_rows, per_class_path)
    save_confusion_matrix(
        metrics["confusion_matrix"], label_names, cm_path,
        normalize=False, title="Test Confusion Matrix (raw counts)",
    )
    save_confusion_matrix(
        metrics["confusion_matrix"], label_names, cm_norm_path,
        normalize=True, title="Test Confusion Matrix (row-normalised)",
    )
    write_predictions_csv(y, preds, probs, label_map, preds_path)
    write_latency_csv(latency["per_class"], latency_path)

    # Summary.
    print("", flush=True)
    print(
        f"[evaluate] raw_acc={metrics['accuracy']:.4f}  "
        f"merged_acc={metrics['merged_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  weighted_f1={metrics['weighted_f1']:.4f}",
        flush=True,
    )
    if latency.get("batch") is not None:
        print(
            f"[evaluate] batch latency mean={latency['batch']['mean_ms']:.3f}ms  "
            f"per-sample mean={latency['batch']['per_sample_mean_ms']*1000:.3f}us  "
            f"single-sample mean={latency['single_sample']['mean_ms']:.3f}ms",
            flush=True,
        )
    print(gate_msg, flush=True)

    return {
        "metrics_payload": metrics_payload,
        "paths": {
            "metrics_json": str(metrics_path),
            "per_class_csv": str(per_class_path),
            "confusion_matrix_png": str(cm_path),
            "confusion_matrix_normalized_png": str(cm_norm_path),
            "predictions_csv": str(preds_path),
            "latency_csv": str(latency_path),
        },
        "gate_passed": gate_passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stage 6: score the trained Stage 4 MLP on the held-out test split "
            "and write the evaluation scorecard under runs/evaluation/."
        ),
    )
    p.add_argument("--ckpt", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p.add_argument("--labels", type=Path, default=LABELS_JSON_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=EVAL_DIR_DEFAULT)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    p.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    p.add_argument(
        "--device", type=str, default="auto",
        choices=("auto", "cpu", "cuda", "xpu"),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--no-latency", action="store_true",
        help="Skip latency measurement (faster smoke checks).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_evaluation(
        checkpoint_path=args.ckpt,
        splits_dir=args.splits_dir,
        labels_path=args.labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        latency_repeats=args.latency_repeats,
        warmup_batches=args.warmup_batches,
        device=args.device,
        seed=args.seed,
        measure_latency=not args.no_latency,
    )


if __name__ == "__main__":
    main()
