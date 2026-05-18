"""Stage 6 evaluation tests.

All fast — every test uses synthetic NPZ + a freshly-built checkpoint via
``src.train.save_checkpoint`` so the file never touches ``data/splits/*.npz``
or ``runs/mlp_best.pt``. Mirrors the convention in ``tests/test_model.py``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src import evaluate as ev
from src import train as train_mod
from src.models.mlp import DROPOUTS, HIDDEN_DIMS, INPUT_DIM, NUM_CLASSES, GestureMLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_synthetic_npz(
    path: Path, n: int, dim: int, num_classes: int, seed: int = 0,
    cover_all_classes: bool = True,
) -> None:
    """Write a Stage-2-style NPZ with ``X`` (float32) and ``y`` (int32)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    if cover_all_classes and n >= num_classes:
        y = np.tile(np.arange(num_classes, dtype=np.int32),
                    int(np.ceil(n / num_classes)))[:n]
        # Shuffle deterministically so the first batch isn't strictly ordered.
        idx = rng.permutation(n)
        y = y[idx]
    else:
        y = rng.integers(0, num_classes, size=n).astype(np.int32)
    np.savez(path, X=X, y=y)


def _build_checkpoint_at(path: Path, seed: int = 0) -> dict:
    """Create a fresh GestureMLP, save a Stage-4-shaped checkpoint to ``path``,
    return the labels map used."""
    torch.manual_seed(seed)
    model = GestureMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = train_mod.make_scheduler(opt, patience=10, factor=0.5, min_lr=1e-5)

    scaler = type("S", (), {})()
    scaler.mean_ = np.zeros(INPUT_DIM, dtype=np.float32)
    scaler.scale_ = np.ones(INPUT_DIM, dtype=np.float32)

    cfg = {
        "input_dim": INPUT_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropouts": list(DROPOUTS),
        "num_classes": NUM_CLASSES,
    }
    # Derive the {id: name} map from the canonical data/labels.json so any
    # schema change (e.g. the 28-class -> 26-class collision fix) flows through.
    with open(Path("data/labels.json"), "r", encoding="utf-8") as f:
        labels = {int(v): str(k) for k, v in json.load(f).items()}
    assert len(labels) == NUM_CLASSES, (
        f"labels.json has {len(labels)} entries; NUM_CLASSES={NUM_CLASSES}"
    )
    train_mod.save_checkpoint(
        path=path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
        epoch=42,
        metrics={"val_loss": 0.5, "val_acc": 0.5, "merged_val_acc": 0.5,
                 "val_macro_f1": 0.5},
        config=cfg, seed=seed, labels=labels,
    )
    return labels


@pytest.fixture
def synthetic_test_split(tmp_path: Path) -> Path:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_synthetic_npz(
        splits_dir / "test.npz", n=200, dim=INPUT_DIM,
        num_classes=NUM_CLASSES, seed=1,
    )
    return splits_dir


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "mlp_best.pt"
    _build_checkpoint_at(path, seed=0)
    return path


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_load_test_split_rejects_missing_X(tmp_path):
    bad = tmp_path / "test.npz"
    np.savez(bad, y=np.zeros(5, dtype=np.int32))
    with pytest.raises(KeyError):
        ev.load_test_split(tmp_path)


def test_load_test_split_rejects_wrong_feature_dim(tmp_path):
    _write_synthetic_npz(
        tmp_path / "test.npz", n=10, dim=INPUT_DIM - 1,
        num_classes=NUM_CLASSES, seed=2,
    )
    with pytest.raises(ValueError, match="feature dim"):
        ev.load_test_split(tmp_path)


def test_validate_test_schema_rejects_invalid_label_ids(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=20).astype(np.int32)
    y[0] = 99
    np.savez(tmp_path / "test.npz", X=X, y=y)
    with pytest.raises(ValueError, match="outside data/labels.json"):
        ev.load_test_split(tmp_path)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def test_load_model_checkpoint_returns_eval_model(synthetic_checkpoint):
    model, scaler, label_map, ck = ev.load_model_checkpoint(
        checkpoint_path=synthetic_checkpoint,
        labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )
    assert model.training is False
    assert scaler.mean_.shape == (INPUT_DIM,)
    assert scaler.scale_.shape == (INPUT_DIM,)
    assert set(label_map.keys()) == set(range(NUM_CLASSES))
    assert label_map[8] == "peace"
    assert label_map[9] == "open_palm"
    assert ck["epoch"] == 42


def test_load_model_checkpoint_rejects_dim_mismatch(tmp_path):
    """A tampered checkpoint config raises ValueError."""
    path = tmp_path / "mlp_best.pt"
    _build_checkpoint_at(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["config"]["input_dim"] = 100  # tamper
    torch.save(ck, path)
    with pytest.raises(ValueError, match="input_dim"):
        ev.load_model_checkpoint(
            checkpoint_path=path,
            labels_path=Path("data/labels.json"),
            device=torch.device("cpu"),
        )


def test_load_model_checkpoint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.load_model_checkpoint(
            checkpoint_path=tmp_path / "does_not_exist.pt",
            labels_path=Path("data/labels.json"),
            device=torch.device("cpu"),
        )


def test_load_model_checkpoint_rejects_wrong_scaler_length(tmp_path):
    path = tmp_path / "mlp_best.pt"
    _build_checkpoint_at(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["scaler_mean"] = list(np.zeros(50, dtype=np.float32))
    torch.save(ck, path)
    with pytest.raises(ValueError, match="scaler_mean"):
        ev.load_model_checkpoint(
            checkpoint_path=path,
            labels_path=Path("data/labels.json"),
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_batches_returns_correct_shapes(synthetic_checkpoint):
    model, _, _, _ = ev.load_model_checkpoint(
        synthetic_checkpoint, labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, INPUT_DIM)).astype(np.float32)
    preds, probs = ev.predict_batches(model, X, batch_size=16, device=torch.device("cpu"))
    assert preds.shape == (50,)
    assert probs.shape == (50, NUM_CLASSES)
    assert preds.dtype == np.int64
    sums = probs.sum(axis=1)
    assert np.allclose(sums, np.ones(50), atol=1e-4)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_classification_metrics_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=300).astype(np.int64)
    y_pred = rng.integers(0, NUM_CLASSES, size=300).astype(np.int64)
    m = ev.compute_classification_metrics(y_true, y_pred, list(range(NUM_CLASSES)))
    for k in ("accuracy", "macro_f1", "weighted_f1",
              "merged_accuracy", "merged_macro_f1", "merged_weighted_f1"):
        assert 0.0 <= m[k] <= 1.0
    assert m["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)
    assert m["confusion_matrix"].dtype == np.int64


def test_compute_classification_metrics_perfect():
    y_true = np.tile(np.arange(NUM_CLASSES, dtype=np.int64), 5)
    y_pred = y_true.copy()
    m = ev.compute_classification_metrics(y_true, y_pred, list(range(NUM_CLASSES)))
    assert math.isclose(m["accuracy"], 1.0, abs_tol=1e-12)
    assert math.isclose(m["macro_f1"], 1.0, abs_tol=1e-12)
    assert math.isclose(m["weighted_f1"], 1.0, abs_tol=1e-12)
    assert all(pa == 1.0 for pa in m["per_class_accuracy"])


def test_per_class_metrics_cover_all_labels_and_handle_zero_support():
    rng = np.random.default_rng(0)
    # Last class (NUM_CLASSES - 1) has no test samples; class 0 has many.
    y_true = np.concatenate([
        np.zeros(40, dtype=np.int64),
        rng.integers(1, NUM_CLASSES - 1, size=40).astype(np.int64),
    ])
    y_pred = rng.integers(0, NUM_CLASSES, size=80).astype(np.int64)
    m = ev.compute_classification_metrics(y_true, y_pred, list(range(NUM_CLASSES)))
    assert len(m["per_class_accuracy"]) == NUM_CLASSES
    assert len(m["per_class_support"]) == NUM_CLASSES
    last = NUM_CLASSES - 1
    assert m["per_class_support"][last] == 0
    assert m["per_class_accuracy"][last] is None


def test_confusion_matrix_shape_and_labels_order():
    y_true = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    y_pred = np.array([0, 0, 2, 1, 0], dtype=np.int64)
    m = ev.compute_classification_metrics(y_true, y_pred, list(range(NUM_CLASSES)))
    cm = m["confusion_matrix"]
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    # Class 0: 2 correct, 0 wrong -> row sums of unique classes match input.
    assert cm[0, 0] == 2
    assert cm[1, 0] == 1  # one class-1 was predicted as 0
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_latency_returns_positive_finite_timings(synthetic_checkpoint):
    model, _, label_map, _ = ev.load_model_checkpoint(
        synthetic_checkpoint, labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=64).astype(np.int32)
    lat = ev.compute_prediction_latency(
        model, X, y, label_map,
        batch_size=16, repeats=4, warmup=1,
        single_sample_repeats=20, single_sample_warmup=2,
        per_class_cap=10, device=torch.device("cpu"),
    )
    b = lat["batch"]
    assert b["mean_ms"] > 0 and math.isfinite(b["mean_ms"])
    assert b["median_ms"] > 0
    assert b["p95_ms"] >= b["median_ms"] - 1e-9
    s = lat["single_sample"]
    assert s["mean_ms"] > 0 and math.isfinite(s["mean_ms"])
    assert len(lat["per_class"]) == NUM_CLASSES
    # At least one class with support should have a numeric mean.
    has_numeric = any(
        row["mean_ms"] is not None and row["mean_ms"] > 0
        for row in lat["per_class"]
    )
    assert has_numeric


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------


def test_save_confusion_matrix_writes_png(tmp_path):
    cm = np.eye(NUM_CLASSES, dtype=np.int64) * 10
    names = [f"c{i}" for i in range(NUM_CLASSES)]
    raw_path = tmp_path / "cm.png"
    norm_path = tmp_path / "cm_norm.png"
    ev.save_confusion_matrix(cm, names, raw_path, normalize=False)
    ev.save_confusion_matrix(cm, names, norm_path, normalize=True)
    assert raw_path.is_file() and raw_path.stat().st_size > 0
    assert norm_path.is_file() and norm_path.stat().st_size > 0


def test_write_metrics_json_roundtrip(tmp_path):
    payload = {
        "schema_version": 1,
        "metrics": {"accuracy": 0.5, "macro_f1": 0.4},
        "confusion_matrix": np.eye(NUM_CLASSES, dtype=np.int64).tolist(),
        "value_with_nan": float("nan"),
    }
    path = tmp_path / "metrics.json"
    ev.write_metrics_json(payload, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert isinstance(loaded["metrics"]["accuracy"], float)
    assert loaded["value_with_nan"] is None
    assert len(loaded["confusion_matrix"]) == NUM_CLASSES


def test_write_per_class_csv_columns(tmp_path):
    rows = []
    for i in range(NUM_CLASSES):
        rows.append({
            "label_id": i, "label_name": f"c{i}", "support": i,
            "accuracy": 0.5 if i > 0 else None, "precision": 0.4,
            "recall": 0.3, "f1": 0.35,
            "latency_mean_ms": 0.1, "latency_median_ms": 0.09,
            "latency_p95_ms": 0.2,
        })
    path = tmp_path / "per_class.csv"
    ev.write_per_class_csv(rows, path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == ev.PER_CLASS_CSV_FIELDS
    assert len(body) == NUM_CLASSES
    # None-valued accuracy serialized as blank string.
    assert body[0][3] == ""


def test_write_per_class_csv_rejects_missing_keys(tmp_path):
    bad = [{"label_id": 0}]
    with pytest.raises(ValueError, match="missing required keys"):
        ev.write_per_class_csv(bad, tmp_path / "bad.csv")


def test_label_mapping_matches_labels_json(synthetic_checkpoint):
    _, _, label_map, _ = ev.load_model_checkpoint(
        synthetic_checkpoint, labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )
    # peace and open_palm are the canonical 1:1 labels after the 26-class
    # collision fix (count_2 / count_5 no longer exist as separate ids).
    assert label_map[8] == "peace"
    assert label_map[9] == "open_palm"
    assert "count_2" not in label_map.values()
    assert "count_5" not in label_map.values()


# ---------------------------------------------------------------------------
# End-to-end (synthetic)
# ---------------------------------------------------------------------------


def test_run_evaluation_writes_all_artefacts(
    synthetic_checkpoint, synthetic_test_split, tmp_path
):
    out = tmp_path / "eval_out"
    result = ev.run_evaluation(
        checkpoint_path=synthetic_checkpoint,
        splits_dir=synthetic_test_split,
        labels_path=Path("data/labels.json"),
        output_dir=out,
        batch_size=32,
        latency_repeats=3,
        warmup_batches=1,
        device="cpu",
        seed=0,
        measure_latency=True,
    )
    paths = result["paths"]
    for key in (
        "metrics_json", "per_class_csv", "confusion_matrix_png",
        "confusion_matrix_normalized_png", "predictions_csv", "latency_csv",
    ):
        assert Path(paths[key]).is_file(), f"missing artefact: {key}"

    metrics = json.loads(Path(paths["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 1
    assert metrics["feature_dim"] == INPUT_DIM
    assert metrics["num_classes"] == NUM_CLASSES
    assert metrics["n_samples"] == 200
    assert len(metrics["confusion_matrix"]) == NUM_CLASSES
    assert len(metrics["confusion_matrix"][0]) == NUM_CLASSES
    assert "all_passed" in metrics["acceptance"]
    assert "gate_message" in metrics["acceptance"]
