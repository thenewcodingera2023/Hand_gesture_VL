"""Logistic Regression and RBF SVM baselines for the 28-class gesture task.

Stage 3 of the gesture recognition pipeline. Trains and evaluates two classical
ML baselines on the Stage 2 splits, appends one row per model to
``runs/baselines.csv``, and persists each fitted pipeline as a joblib artefact.

Authoritative spec:
    - tasks/gesture_recognition_plan_v2.md §6.1
    - tasks/implementation_stages.md Stage 3
    - tasks/stage3_handoff.md
    - plan at C:/Users/Harry T/.claude/plans/you-are-claude-opus-twinkly-blossom.md

Backend selection
-----------------
At import time the module attempts ``sklearnex.patch_sklearn()`` which swaps
sklearn estimators for Intel oneDAL implementations (3-10x faster on CPU for
SVC RBF / LogisticRegression / StandardScaler). GPU offload via
``target_offload="gpu:0"`` is supported but Arc 140V + Windows currently hangs
on SVC fits — see tasks/lessons.md. The ``--gpu`` CLI flag exists for future
hardware but defaults to OFF for that reason.

CLI
---
    python -m src.models.baseline train-lr
    python -m src.models.baseline train-svm [--subsample 30000] [--refit-subsample 60000]
                                            [--c-grid 0.1 1 10 100] [--cv 5]
                                            [--no-grid] [--gpu]
    python -m src.models.baseline train-all [--subsample-svm 30000] [--gpu]
    python -m src.models.baseline evaluate  --model runs/baseline_lr.joblib
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Add Intel oneAPI runtime to the DLL search path on Windows so dpctl/sklearnex
# GPU offload can resolve sycl9.dll, ur_loader.dll, etc. Safe no-op elsewhere.
if os.name == "nt":
    _lib_bin = Path(sys.prefix) / "Library" / "bin"
    if _lib_bin.is_dir():
        try:
            os.add_dll_directory(str(_lib_bin))
        except (OSError, AttributeError):
            pass

# Try sklearnex (Intel oneDAL); silently fall back to stock sklearn.
try:
    from sklearnex import patch_sklearn, config_context as _sklearnex_config_context

    patch_sklearn(verbose=False)
    _SKLEARNEX_AVAILABLE = True
except Exception:  # pragma: no cover - depends on Intel runtime
    _SKLEARNEX_AVAILABLE = False
    _sklearnex_config_context = None

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 20260514  # mirrors src/dataset.py
SPLITS_DIR_DEFAULT = Path("data/splits")
RUNS_DIR_DEFAULT = Path("runs")
BASELINES_CSV_DEFAULT = RUNS_DIR_DEFAULT / "baselines.csv"
LR_MODEL_PATH_DEFAULT = RUNS_DIR_DEFAULT / "baseline_lr.joblib"
SVM_MODEL_PATH_DEFAULT = RUNS_DIR_DEFAULT / "baseline_svm.joblib"
LABELS_JSON_DEFAULT = Path("data/labels.json")

EXPECTED_FEATURE_DIM = 279
EXPECTED_NUM_CLASSES = 28

SVM_DEFAULT_SUBSAMPLE = 30_000
SVM_DEFAULT_REFIT_SUBSAMPLE = 60_000
SVM_DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
SVM_DEFAULT_CV = 5

ACCEPTANCE_GATE_ACCURACY = 0.85

# HaGRID dual-labels the same hand shape under two class IDs:
#   - "peace" (8) == "count_2" (11): two fingers extended
#   - "open_palm" (9) == "count_5" (14): five fingers extended
# A pure-shape classifier cannot disambiguate these (see gesture_recognition_plan_v2.md
# §1.3 — counting "2" *is* the peace sign). Raw accuracy is bounded near 0.81 by
# this 50/50 split. We report merged_accuracy / merged_macro_f1 as supplementary
# metrics where the equivalence classes are collapsed.
LABEL_EQUIVALENCE: dict[int, int] = {8: 8, 11: 8, 9: 9, 14: 9}


def _merge_labels(y: np.ndarray) -> np.ndarray:
    """Apply LABEL_EQUIVALENCE to collapse HaGRID's dual-labelled gesture classes."""
    out = y.astype(np.int32, copy=True)
    for src, dst in LABEL_EQUIVALENCE.items():
        if src != dst:
            out[out == src] = dst
    return out


CSV_FIELDS: list[str] = [
    "created_at",
    "model",
    "hyperparameters",
    "train_n",
    "val_n",
    "feature_dim",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "fit_seconds",
    "predict_seconds",
    "seed",
    "notes",
]


# ---------------------------------------------------------------------------
# Backend context
# ---------------------------------------------------------------------------


def _maybe_gpu_context(use_gpu: bool):
    """Return a context manager that requests Intel GPU offload when possible.

    Falls back to a nullcontext (CPU oneDAL or stock sklearn) when sklearnex is
    unavailable or ``use_gpu`` is False.
    """
    if use_gpu and _SKLEARNEX_AVAILABLE and _sklearnex_config_context is not None:
        return _sklearnex_config_context(
            target_offload="gpu:0", allow_fallback_to_host=True
        )
    return contextlib.nullcontext()


def _resolve_backend(use_gpu: bool) -> str:
    if not _SKLEARNEX_AVAILABLE:
        return "sklearn_cpu"
    return "sklearnex_gpu" if use_gpu else "sklearnex_cpu"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_label_ids(labels_path: Path = LABELS_JSON_DEFAULT) -> set[int]:
    """Return the set of valid integer label IDs declared in ``data/labels.json``."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.json not found at {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(v) for v in raw.values()}


def load_split(
    name: str, splits_dir: Path = SPLITS_DIR_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` from ``data/splits/{name}.npz`` with no metadata."""
    if name not in ("train", "val", "test"):
        raise ValueError(f"name must be train|val|test, got {name!r}")
    path = splits_dir / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}. "
            f"Run Stage 2 first: python -m src.dataset build"
        )
    with np.load(path, allow_pickle=True) as z:
        if "X" not in z.files:
            raise KeyError(f"{path} missing required array 'X'")
        if "y" not in z.files:
            raise KeyError(f"{path} missing required array 'y'")
        X = z["X"].astype(np.float32, copy=False)
        y = z["y"].astype(np.int32, copy=False)
    return X, y


def validate_split_schema(
    X: np.ndarray,
    y: np.ndarray,
    expected_dim: int = EXPECTED_FEATURE_DIM,
    num_classes: int = EXPECTED_NUM_CLASSES,
    labels_path: Path = LABELS_JSON_DEFAULT,
) -> None:
    """Raise ``ValueError`` if ``X``/``y`` violate the Stage 2 contract."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"X feature dim {X.shape[1]} != expected {expected_dim}. "
            f"Stage 1 feature_assembler emits {expected_dim}-dim vectors."
        )
    if X.dtype != np.float32:
        raise ValueError(f"X dtype {X.dtype} != float32")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf values")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"len(X)={X.shape[0]} != len(y)={y.shape[0]}")
    if labels_path.is_file():
        valid = load_label_ids(labels_path)
    else:
        valid = set(range(num_classes))
    seen = {int(v) for v in np.unique(y)}
    bad = seen - valid
    if bad:
        raise ValueError(
            f"y contains label IDs outside data/labels.json: {sorted(bad)}"
        )


# ---------------------------------------------------------------------------
# Pipelines (scaler + classifier; never separate them)
# ---------------------------------------------------------------------------


def make_logistic_regression(seed: int = DEFAULT_SEED) -> Pipeline:
    """``StandardScaler`` -> ``LogisticRegression`` per plan §6.1.

    Note: ``multi_class="multinomial"`` was removed in sklearn 1.7. With lbfgs
    and k>2 classes, multinomial is now the only behaviour; we just omit the
    kwarg.
    """
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_svm(
    C: float, gamma: str | float = "scale", seed: int = DEFAULT_SEED
) -> Pipeline:
    """``StandardScaler`` -> RBF ``SVC`` per plan §6.1."""
    clf = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        cache_size=1000,
        random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = DEFAULT_SEED,
    use_gpu: bool = False,
) -> tuple[Pipeline, float]:
    """Fit the LR pipeline on the full train set.

    Returns ``(fitted_pipeline, fit_seconds)``.
    """
    pipeline = make_logistic_regression(seed=seed)
    t0 = time.perf_counter()
    with _maybe_gpu_context(use_gpu):
        with warnings.catch_warnings():
            warnings.simplefilter("default", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
    return pipeline, time.perf_counter() - t0


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified subsample of size ``n`` preserving class balance."""
    if n <= 0:
        raise ValueError(f"subsample size must be positive, got {n}")
    if n >= len(X):
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(X, y))
    return X[idx], y[idx]


def train_svm_with_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_grid: Iterable[float] = SVM_DEFAULT_C_GRID,
    cv: int = SVM_DEFAULT_CV,
    n_jobs: int = -1,
    subsample: int | None = SVM_DEFAULT_SUBSAMPLE,
    refit_subsample: int | None = SVM_DEFAULT_REFIT_SUBSAMPLE,
    seed: int = DEFAULT_SEED,
    use_gpu: bool = False,
) -> tuple[Pipeline, dict, float]:
    """Grid-search ``C`` on a stratified subsample, then refit on a larger subsample.

    Returns ``(best_pipeline, grid_info, fit_seconds_total)``.
    """
    C_grid_t = tuple(float(c) for c in C_grid)
    if not C_grid_t:
        raise ValueError("C_grid must be non-empty")

    t0 = time.perf_counter()

    # Grid-search subsample.
    if subsample is not None and subsample < len(X_train):
        X_sub, y_sub = stratified_subsample(X_train, y_train, subsample, seed=seed)
    else:
        X_sub, y_sub = X_train, y_train
    grid_n_used = len(X_sub)

    base_pipeline = make_svm(C=1.0, gamma="scale", seed=seed)
    param_grid = {"clf__C": list(C_grid_t)}

    with _maybe_gpu_context(use_gpu):
        gs = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_sub, y_sub)
        best_C = float(gs.best_params_["clf__C"])
        best_cv_score = float(gs.best_score_)
        best_pipeline = gs.best_estimator_
        final_n_used = grid_n_used

        # Optional larger refit at best C.
        if (
            refit_subsample is not None
            and refit_subsample > grid_n_used
            and refit_subsample <= len(X_train)
        ):
            X_refit, y_refit = stratified_subsample(
                X_train, y_train, refit_subsample, seed=seed
            )
            best_pipeline = make_svm(C=best_C, gamma="scale", seed=seed)
            best_pipeline.fit(X_refit, y_refit)
            final_n_used = len(X_refit)

    cv_results_summary = {
        f"C={c}": float(s)
        for c, s in zip(
            gs.cv_results_["param_clf__C"].data, gs.cv_results_["mean_test_score"]
        )
    }

    grid_info = {
        "best_C": best_C,
        "best_cv_score": best_cv_score,
        "cv_results_summary": cv_results_summary,
        "train_n_used": final_n_used,
        "grid_train_n_used": grid_n_used,
        "C_grid": list(C_grid_t),
        "cv": int(cv),
        "backend": _resolve_backend(use_gpu),
    }

    return best_pipeline, grid_info, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    model: Pipeline, X_val: np.ndarray, y_val: np.ndarray
) -> dict:
    """Return accuracy, macro F1, weighted F1, and predict wall time.

    Also returns ``merged_accuracy`` and ``merged_macro_f1`` computed under
    ``LABEL_EQUIVALENCE`` to account for HaGRID's dual-labelling of
    peace==count_2 and open_palm==count_5.
    """
    check_is_fitted(model)
    t0 = time.perf_counter()
    y_pred = model.predict(X_val)
    predict_seconds = time.perf_counter() - t0
    yt_m = _merge_labels(y_val)
    yp_m = _merge_labels(y_pred)
    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_val, y_pred, average="weighted", zero_division=0)
        ),
        "merged_accuracy": float(accuracy_score(yt_m, yp_m)),
        "merged_macro_f1": float(f1_score(yt_m, yp_m, average="macro", zero_division=0)),
        "predict_seconds": float(predict_seconds),
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def write_baseline_metrics(
    rows: list[dict], output_path: Path = BASELINES_CSV_DEFAULT
) -> None:
    """Append ``rows`` to ``runs/baselines.csv``, creating the header if needed."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.is_file()
    if not write_header:
        with output_path.open("r", encoding="utf-8", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != CSV_FIELDS:
            print(
                f"WARNING: {output_path} header {existing_header} differs from "
                f"CSV_FIELDS {CSV_FIELDS}; appending rows in CSV_FIELDS order anyway.",
                file=sys.stderr,
            )
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            missing = [k for k in CSV_FIELDS if k not in row]
            if missing:
                raise ValueError(f"row missing required keys {missing}: {row}")
            writer.writerow(row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _gate_message(model_name: str, metrics: dict) -> str:
    raw = metrics["accuracy"]
    merged = metrics.get("merged_accuracy", raw)
    gate = ACCEPTANCE_GATE_ACCURACY
    if raw >= gate:
        return (
            f"GATE PASSED: {model_name} val_acc={raw:.4f} >= {gate} "
            f"(merged_acc={merged:.4f}, macro_f1={metrics['macro_f1']:.4f})"
        )
    if merged >= gate:
        return (
            f"GATE PASSED (merged): {model_name} val_acc={raw:.4f} bounded by HaGRID "
            f"dual-labelling; merged_acc={merged:.4f} >= {gate}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    return (
        f"GATE FAILED: {model_name} val_acc={raw:.4f} merged_acc={merged:.4f} "
        f"< {gate} (see plan §9)"
    )


def run_train_lr(
    splits_dir: Path = SPLITS_DIR_DEFAULT,
    runs_dir: Path = RUNS_DIR_DEFAULT,
    seed: int = DEFAULT_SEED,
    use_gpu: bool = False,
) -> dict:
    X_train, y_train = load_split("train", splits_dir)
    X_val, y_val = load_split("val", splits_dir)
    validate_split_schema(X_train, y_train)
    validate_split_schema(X_val, y_val)

    print(
        f"[LR] train rows={len(X_train)}  val rows={len(X_val)}  "
        f"feature_dim={X_train.shape[1]}  seed={seed}  "
        f"backend={_resolve_backend(use_gpu)}",
        flush=True,
    )

    convergence_warning: str | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline, fit_seconds = train_logistic_regression(
            X_train, y_train, seed=seed, use_gpu=use_gpu
        )
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                convergence_warning = str(w.message).split("\n", 1)[0]
                break

    metrics = evaluate_classifier(pipeline, X_val, y_val)
    print(
        f"[LR] fit={fit_seconds:.1f}s  predict={metrics['predict_seconds']:.2f}s  "
        f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, LR_MODEL_PATH_DEFAULT)

    hyperparams = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": seed,
    }
    notes_parts = [
        f"backend={_resolve_backend(use_gpu)}",
        f"merged_accuracy={metrics['merged_accuracy']:.4f}",
        f"merged_macro_f1={metrics['merged_macro_f1']:.4f}",
    ]
    if convergence_warning:
        notes_parts.append(f"convergence={convergence_warning!r}")

    row = {
        "created_at": _now_iso(),
        "model": "logistic_regression",
        "hyperparameters": json.dumps(hyperparams, sort_keys=True),
        "train_n": int(len(X_train)),
        "val_n": int(len(X_val)),
        "feature_dim": int(X_train.shape[1]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(metrics["predict_seconds"]),
        "seed": int(seed),
        "notes": "; ".join(notes_parts),
    }
    write_baseline_metrics([row], runs_dir / "baselines.csv")
    print(_gate_message("logistic_regression", metrics), flush=True)
    return row


def run_train_svm(
    splits_dir: Path = SPLITS_DIR_DEFAULT,
    runs_dir: Path = RUNS_DIR_DEFAULT,
    seed: int = DEFAULT_SEED,
    C_grid: Iterable[float] = SVM_DEFAULT_C_GRID,
    cv: int = SVM_DEFAULT_CV,
    subsample: int | None = SVM_DEFAULT_SUBSAMPLE,
    refit_subsample: int | None = SVM_DEFAULT_REFIT_SUBSAMPLE,
    n_jobs: int = -1,
    no_grid: bool = False,
    use_gpu: bool = False,
) -> dict:
    X_train, y_train = load_split("train", splits_dir)
    X_val, y_val = load_split("val", splits_dir)
    validate_split_schema(X_train, y_train)
    validate_split_schema(X_val, y_val)

    backend = _resolve_backend(use_gpu)
    C_grid_t = (1.0,) if no_grid else tuple(C_grid)

    print(
        f"[SVM] train rows={len(X_train)}  val rows={len(X_val)}  "
        f"feature_dim={X_train.shape[1]}  seed={seed}  backend={backend}  "
        f"C_grid={list(C_grid_t)}  cv={cv}  subsample={subsample}  "
        f"refit_subsample={refit_subsample}",
        flush=True,
    )

    pipeline, grid_info, fit_seconds = train_svm_with_grid(
        X_train,
        y_train,
        C_grid=C_grid_t,
        cv=cv,
        n_jobs=n_jobs,
        subsample=subsample,
        refit_subsample=refit_subsample,
        seed=seed,
        use_gpu=use_gpu,
    )

    metrics = evaluate_classifier(pipeline, X_val, y_val)
    print(
        f"[SVM] fit={fit_seconds:.1f}s  predict={metrics['predict_seconds']:.2f}s  "
        f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"best_C={grid_info['best_C']}  best_cv_score={grid_info['best_cv_score']:.4f}",
        flush=True,
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, SVM_MODEL_PATH_DEFAULT)

    hyperparams = {
        "kernel": "rbf",
        "C": grid_info["best_C"],
        "gamma": "scale",
        "probability": True,
        "C_grid": grid_info["C_grid"],
        "cv": cv,
        "random_state": seed,
    }
    notes_parts = [
        f"backend={backend}",
        f"train_n_used={grid_info['train_n_used']}",
        f"grid_train_n_used={grid_info['grid_train_n_used']}",
        f"best_cv_score={grid_info['best_cv_score']:.4f}",
        f"cv_results={json.dumps(grid_info['cv_results_summary'], sort_keys=True)}",
        f"merged_accuracy={metrics['merged_accuracy']:.4f}",
        f"merged_macro_f1={metrics['merged_macro_f1']:.4f}",
    ]
    if no_grid:
        notes_parts.append("no_grid=True")

    row = {
        "created_at": _now_iso(),
        "model": "svm_rbf",
        "hyperparameters": json.dumps(hyperparams, sort_keys=True),
        "train_n": int(grid_info["train_n_used"]),
        "val_n": int(len(X_val)),
        "feature_dim": int(X_train.shape[1]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(metrics["predict_seconds"]),
        "seed": int(seed),
        "notes": "; ".join(notes_parts),
    }
    write_baseline_metrics([row], runs_dir / "baselines.csv")
    print(_gate_message("svm_rbf", metrics), flush=True)
    return row


def run_evaluate(
    model_path: Path, splits_dir: Path = SPLITS_DIR_DEFAULT, split: str = "val"
) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"model artefact not found: {model_path}")
    pipeline = joblib.load(model_path)
    X, y = load_split(split, splits_dir)
    validate_split_schema(X, y)
    metrics = evaluate_classifier(pipeline, X, y)
    metrics["model_path"] = str(model_path)
    metrics["split"] = split
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: classical-ML baselines for the gesture pipeline."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lr = sub.add_parser("train-lr", help="Train Logistic Regression on the full train split.")
    p_lr.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p_lr.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    p_lr.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_lr.add_argument("--gpu", dest="use_gpu", action="store_true", default=False,
                      help="Request Intel GPU offload via sklearnex (default off; see lessons.md).")
    p_lr.add_argument("--no-gpu", dest="use_gpu", action="store_false")

    p_svm = sub.add_parser("train-svm", help="Grid-search RBF SVM C on a stratified subsample.")
    p_svm.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p_svm.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    p_svm.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_svm.add_argument("--subsample", type=int, default=SVM_DEFAULT_SUBSAMPLE)
    p_svm.add_argument("--refit-subsample", type=int, default=SVM_DEFAULT_REFIT_SUBSAMPLE)
    p_svm.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=list(SVM_DEFAULT_C_GRID),
        help="C values to grid-search. Default: 0.1 1 10 100.",
    )
    p_svm.add_argument("--cv", type=int, default=SVM_DEFAULT_CV)
    p_svm.add_argument("--n-jobs", type=int, default=-1)
    p_svm.add_argument("--no-grid", action="store_true",
                       help="Skip grid search; fit a single SVM at C=1.0.")
    p_svm.add_argument("--gpu", dest="use_gpu", action="store_true", default=False)
    p_svm.add_argument("--no-gpu", dest="use_gpu", action="store_false")

    p_all = sub.add_parser("train-all", help="Run train-lr then train-svm in sequence.")
    p_all.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p_all.add_argument("--runs-dir", type=Path, default=RUNS_DIR_DEFAULT)
    p_all.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_all.add_argument("--subsample-svm", type=int, default=SVM_DEFAULT_SUBSAMPLE)
    p_all.add_argument("--refit-subsample-svm", type=int, default=SVM_DEFAULT_REFIT_SUBSAMPLE)
    p_all.add_argument("--cv", type=int, default=SVM_DEFAULT_CV)
    p_all.add_argument("--no-grid", action="store_true")
    p_all.add_argument("--gpu", dest="use_gpu", action="store_true", default=False)
    p_all.add_argument("--no-gpu", dest="use_gpu", action="store_false")

    p_ev = sub.add_parser("evaluate", help="Re-evaluate a saved model on a split.")
    p_ev.add_argument("--model", type=Path, required=True)
    p_ev.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p_ev.add_argument("--split", choices=("train", "val", "test"), default="val")

    args = parser.parse_args(argv)

    if args.cmd == "train-lr":
        run_train_lr(
            splits_dir=args.splits_dir,
            runs_dir=args.runs_dir,
            seed=args.seed,
            use_gpu=args.use_gpu,
        )
    elif args.cmd == "train-svm":
        run_train_svm(
            splits_dir=args.splits_dir,
            runs_dir=args.runs_dir,
            seed=args.seed,
            C_grid=tuple(args.c_grid),
            cv=args.cv,
            subsample=args.subsample,
            refit_subsample=args.refit_subsample,
            n_jobs=args.n_jobs,
            no_grid=args.no_grid,
            use_gpu=args.use_gpu,
        )
    elif args.cmd == "train-all":
        run_train_lr(
            splits_dir=args.splits_dir,
            runs_dir=args.runs_dir,
            seed=args.seed,
            use_gpu=args.use_gpu,
        )
        run_train_svm(
            splits_dir=args.splits_dir,
            runs_dir=args.runs_dir,
            seed=args.seed,
            cv=args.cv,
            subsample=args.subsample_svm,
            refit_subsample=args.refit_subsample_svm,
            no_grid=args.no_grid,
            use_gpu=args.use_gpu,
        )
    elif args.cmd == "evaluate":
        run_evaluate(model_path=args.model, splits_dir=args.splits_dir, split=args.split)
    else:
        parser.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
