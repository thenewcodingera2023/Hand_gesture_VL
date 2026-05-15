"""Unit tests for src/models/baseline.py — Stage 3 baseline classifiers.

Tests run on a small synthetic 28-class dataset (sklearn.datasets.make_classification)
so they finish in seconds. They do NOT load the real data/splits/*.npz files
(those are validated by the CLI gate at training time).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from src.models import baseline as bl


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO_ROOT / "data" / "labels.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """Tiny 28-class, 279-dim dataset that fits LR/SVM in <2 s."""
    X, y = make_classification(
        n_samples=560,
        n_features=bl.EXPECTED_FEATURE_DIM,
        n_informative=60,
        n_redundant=20,
        n_classes=bl.EXPECTED_NUM_CLASSES,
        n_clusters_per_class=1,
        random_state=bl.DEFAULT_SEED,
    )
    rng = np.random.default_rng(bl.DEFAULT_SEED)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    X_tr = X[idx[:cut]].astype(np.float32)
    y_tr = y[idx[:cut]].astype(np.int32)
    X_va = X[idx[cut:]].astype(np.float32)
    y_va = y[idx[cut:]].astype(np.int32)
    return X_tr, y_tr, X_va, y_va


@pytest.fixture
def valid_arrays():
    """A small (X, y) pair that satisfies the schema."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, bl.EXPECTED_FEATURE_DIM)).astype(np.float32)
    y = (rng.integers(0, bl.EXPECTED_NUM_CLASSES, size=50)).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_validate_split_schema_accepts_valid(valid_arrays):
    X, y = valid_arrays
    bl.validate_split_schema(X, y)


def test_validate_split_schema_rejects_wrong_dim(valid_arrays):
    X, y = valid_arrays
    with pytest.raises(ValueError, match="feature dim"):
        bl.validate_split_schema(X[:, :278], y)
    X_wide = np.concatenate([X, np.zeros((X.shape[0], 1), dtype=np.float32)], axis=1)
    with pytest.raises(ValueError, match="feature dim"):
        bl.validate_split_schema(X_wide, y)


def test_validate_split_schema_rejects_nan(valid_arrays):
    X, y = valid_arrays
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        bl.validate_split_schema(X_bad, y)


def test_validate_split_schema_rejects_bad_labels(valid_arrays):
    X, y = valid_arrays
    y_bad = y.copy()
    y_bad[0] = 28
    with pytest.raises(ValueError, match="label IDs"):
        bl.validate_split_schema(X_bad := X, y_bad)
    y_neg = y.copy()
    y_neg[0] = -1
    with pytest.raises(ValueError, match="label IDs"):
        bl.validate_split_schema(X, y_neg)


def test_validate_split_schema_rejects_length_mismatch(valid_arrays):
    X, y = valid_arrays
    with pytest.raises(ValueError, match="len"):
        bl.validate_split_schema(X, y[:-1])


def test_validate_split_schema_rejects_wrong_dtype(valid_arrays):
    X, y = valid_arrays
    with pytest.raises(ValueError, match="dtype"):
        bl.validate_split_schema(X.astype(np.float64), y)


def test_load_label_ids_returns_0_to_27():
    assert bl.load_label_ids(LABELS_PATH) == set(range(28))


def test_load_split_rejects_missing_x_or_y(tmp_path):
    # Missing 'X'
    np.savez(tmp_path / "train.npz", y=np.zeros(3, dtype=np.int32))
    with pytest.raises(KeyError, match="'X'"):
        bl.load_split("train", tmp_path)
    # Missing 'y'
    np.savez(tmp_path / "val.npz", X=np.zeros((3, 279), dtype=np.float32))
    with pytest.raises(KeyError, match="'y'"):
        bl.load_split("val", tmp_path)


def test_load_split_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Stage 2"):
        bl.load_split("train", tmp_path)


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def test_make_logistic_regression_pipeline_shape():
    p = bl.make_logistic_regression()
    assert isinstance(p, Pipeline)
    assert list(p.named_steps.keys()) == ["scaler", "clf"]
    assert isinstance(p.named_steps["scaler"], StandardScaler)
    clf = p.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.max_iter == 2000
    assert clf.C == 1.0
    assert clf.solver == "lbfgs"
    assert clf.penalty == "l2"


def test_make_svm_pipeline_shape():
    p = bl.make_svm(C=2.5, gamma="scale")
    assert isinstance(p, Pipeline)
    assert list(p.named_steps.keys()) == ["scaler", "clf"]
    assert isinstance(p.named_steps["scaler"], StandardScaler)
    clf = p.named_steps["clf"]
    assert isinstance(clf, SVC)
    assert clf.kernel == "rbf"
    assert clf.C == 2.5
    assert clf.probability is True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_logistic_regression_fits(synthetic_data):
    X_tr, y_tr, X_va, _ = synthetic_data
    pipeline, fit_seconds = bl.train_logistic_regression(X_tr, y_tr, use_gpu=False)
    check_is_fitted(pipeline)
    assert fit_seconds >= 0.0
    pred = pipeline.predict(X_va)
    assert pred.shape == (len(X_va),)


def test_train_svm_with_grid_returns_fitted(synthetic_data):
    X_tr, y_tr, _, _ = synthetic_data
    pipeline, grid_info, fit_seconds = bl.train_svm_with_grid(
        X_tr,
        y_tr,
        C_grid=(1.0, 10.0),
        cv=2,
        n_jobs=1,
        subsample=200,
        refit_subsample=None,
        use_gpu=False,
    )
    check_is_fitted(pipeline)
    assert fit_seconds >= 0.0
    assert grid_info["best_C"] in (1.0, 10.0)
    assert "cv_results_summary" in grid_info
    assert grid_info["train_n_used"] <= len(X_tr)
    assert isinstance(grid_info["backend"], str)


def test_stratified_subsample_deterministic(synthetic_data):
    X_tr, y_tr, _, _ = synthetic_data
    X_a, y_a = bl.stratified_subsample(X_tr, y_tr, n=200, seed=42)
    X_b, y_b = bl.stratified_subsample(X_tr, y_tr, n=200, seed=42)
    assert np.array_equal(X_a, X_b)
    assert np.array_equal(y_a, y_b)
    X_c, _ = bl.stratified_subsample(X_tr, y_tr, n=200, seed=43)
    assert not np.array_equal(X_a, X_c)


def test_stratified_subsample_preserves_classes(synthetic_data):
    X_tr, y_tr, _, _ = synthetic_data
    _, y_sub = bl.stratified_subsample(X_tr, y_tr, n=len(X_tr) // 2)
    assert set(np.unique(y_sub).tolist()) == set(np.unique(y_tr).tolist())


def test_stratified_subsample_passthrough_when_n_exceeds_len(synthetic_data):
    X_tr, y_tr, _, _ = synthetic_data
    X_out, y_out = bl.stratified_subsample(X_tr, y_tr, n=len(X_tr) + 10)
    assert X_out is X_tr
    assert y_out is y_tr


def test_stratified_subsample_rejects_zero():
    X = np.zeros((10, 279), dtype=np.float32)
    y = np.arange(10, dtype=np.int32) % 3
    with pytest.raises(ValueError):
        bl.stratified_subsample(X, y, n=0)


def test_scaler_fit_only_on_train(synthetic_data):
    X_tr, y_tr, X_va, _ = synthetic_data
    pipeline, _ = bl.train_logistic_regression(X_tr, y_tr, use_gpu=False)
    learned_mean = pipeline.named_steps["scaler"].mean_
    np.testing.assert_allclose(learned_mean, X_tr.mean(axis=0), rtol=1e-5, atol=1e-5)
    # Mean should NOT match val mean (no leakage).
    assert not np.allclose(learned_mean, X_va.mean(axis=0), rtol=1e-3)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_classifier_returns_metrics_in_range(synthetic_data):
    X_tr, y_tr, X_va, y_va = synthetic_data
    pipeline, _ = bl.train_logistic_regression(X_tr, y_tr, use_gpu=False)
    metrics = bl.evaluate_classifier(pipeline, X_va, y_va)
    expected_keys = {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "merged_accuracy",
        "merged_macro_f1",
        "predict_seconds",
    }
    assert set(metrics.keys()) == expected_keys
    for k in ("accuracy", "macro_f1", "weighted_f1", "merged_accuracy", "merged_macro_f1"):
        assert 0.0 <= metrics[k] <= 1.0
    # Merged accuracy >= raw accuracy (merging only collapses confusable classes).
    assert metrics["merged_accuracy"] >= metrics["accuracy"] - 1e-9
    assert metrics["predict_seconds"] >= 0.0


def test_merge_labels_collapses_dual_labels():
    import numpy as np

    y = np.array([8, 11, 9, 14, 0, 27], dtype=np.int32)
    merged = bl._merge_labels(y)
    assert merged.tolist() == [8, 8, 9, 9, 0, 27]


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def _row(accuracy: float = 0.9, model: str = "logistic_regression") -> dict:
    return {
        "created_at": "2026-05-14T00:00:00+00:00",
        "model": model,
        "hyperparameters": json.dumps({"C": 1.0}),
        "train_n": 100,
        "val_n": 50,
        "feature_dim": 279,
        "accuracy": accuracy,
        "macro_f1": 0.8,
        "weighted_f1": 0.85,
        "fit_seconds": 1.0,
        "predict_seconds": 0.1,
        "seed": 20260514,
        "notes": "test",
    }


def test_write_baseline_metrics_creates_csv_with_header(tmp_path):
    out = tmp_path / "subdir" / "baselines.csv"  # tests mkdir parents=True
    bl.write_baseline_metrics([_row()], out)
    assert out.is_file()
    with out.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == bl.CSV_FIELDS
    assert len(rows) == 1


def test_write_baseline_metrics_appends_without_duplicating_header(tmp_path):
    out = tmp_path / "baselines.csv"
    bl.write_baseline_metrics([_row(accuracy=0.9)], out)
    bl.write_baseline_metrics([_row(accuracy=0.92, model="svm_rbf")], out)
    with out.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # 1 header + 2 data rows
    assert len(lines) == 3
    assert lines[0] == ",".join(bl.CSV_FIELDS)


def test_write_baseline_metrics_rejects_missing_keys(tmp_path):
    out = tmp_path / "baselines.csv"
    bad = _row()
    bad.pop("accuracy")
    with pytest.raises(ValueError, match="missing required keys"):
        bl.write_baseline_metrics([bad], out)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


def test_smoke_train_lr_end_to_end(synthetic_data, tmp_path):
    """Train LR -> evaluate -> serialize -> reload -> re-evaluate."""
    X_tr, y_tr, X_va, y_va = synthetic_data
    pipeline, fit_seconds = bl.train_logistic_regression(X_tr, y_tr, use_gpu=False)
    metrics = bl.evaluate_classifier(pipeline, X_va, y_va)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    out = tmp_path / "lr.joblib"
    import joblib

    joblib.dump(pipeline, out)
    reloaded = joblib.load(out)
    metrics_reloaded = bl.evaluate_classifier(reloaded, X_va, y_va)
    assert metrics_reloaded["accuracy"] == pytest.approx(metrics["accuracy"])
    assert fit_seconds >= 0.0
