"""Unit tests for ``src/models/mlp.py`` and ``src/train.py``.

Stage 4 focused tests: output shape, softmax sums to one, loss decrease,
gradient/weight norm correctness, dropout behaviour, CSV logging, schema
validation, and checkpoint round-trip. All tests run on synthetic data so the
file finishes in seconds and never touches ``data/splits/*.npz``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import train as train_mod
from src.models import baseline as bl
from src.models.mlp import (
    DROPOUTS,
    HIDDEN_DIMS,
    INPUT_DIM,
    NUM_CLASSES,
    GestureMLP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch():
    """Small synthetic (X, y) batch."""
    torch.manual_seed(0)
    X = torch.randn(8, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (8,), dtype=torch.long)
    return X, y


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GestureMLP()


def _write_synthetic_npz(path: Path, n: int, dim: int, num_classes: int,
                          seed: int = 0) -> None:
    """Write a Stage-2-style NPZ with ``X`` (float32) and ``y`` (int32)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n).astype(np.int32)
    np.savez(path, X=X, y=y)


# ---------------------------------------------------------------------------
# Model: shape + numerics
# ---------------------------------------------------------------------------


def test_forward_shape(model, batch):
    X, _ = batch
    out = model(X)
    assert out.shape == (8, NUM_CLASSES)


def test_forward_logits_finite(model, batch):
    X, _ = batch
    out = model(X)
    assert torch.isfinite(out).all()


def test_softmax_sums_to_one(model, batch):
    X, _ = batch
    model.eval()  # turn off dropout so the assertion is exact
    probs = F.softmax(model(X), dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_cross_entropy_accepts_logits(model, batch):
    X, y = batch
    loss = nn.CrossEntropyLoss()(model(X), y)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_one_step_changes_parameters(model, batch):
    X, y = batch
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.linear_layers[0].weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    opt.step()
    after = model.linear_layers[0].weight.detach()
    assert (after - before).abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Gradient / weight norms
# ---------------------------------------------------------------------------


def test_grad_norm_zero_before_backward(model):
    """No backward yet -> compute_grad_norm returns 0.0 (no grads attached)."""
    assert train_mod.compute_grad_norm(model) == 0.0


def test_grad_norm_nonzero_at_init(model, batch):
    X, y = batch
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    assert train_mod.compute_grad_norm(model) > 0


def test_compute_grad_norm_matches_manual(model, batch):
    X, y = batch
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    manual = torch.cat(
        [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    ).norm().item()
    ours = train_mod.compute_grad_norm(model)
    assert math.isclose(ours, manual, rel_tol=1e-5, abs_tol=1e-6)


def test_weight_norms_returns_four_positive_values(model):
    norms = train_mod.compute_layer_weight_norms(model)
    assert len(norms) == 4
    assert all(np.isfinite(n) for n in norms)
    assert all(n > 0 for n in norms)


def test_compute_layer_weight_norms_changes_after_step(model, batch):
    """A non-zero gradient step must move at least one weight matrix norm."""
    X, y = batch
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = train_mod.compute_layer_weight_norms(model)
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    opt.step()
    after = train_mod.compute_layer_weight_norms(model)
    assert any(abs(b - a) > 0 for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_kaiming_init_applied(model):
    """Kaiming-uniform for ReLU spans [-sqrt(6/fan_in), sqrt(6/fan_in)].

    A sanity bound: mean absolute value falls comfortably within that range.
    """
    w0 = model.linear_layers[0].weight
    fan_in = w0.shape[1]  # 279
    bound = math.sqrt(6.0 / fan_in)
    assert 0 < w0.abs().mean().item() < bound


def test_bias_initialised_to_zero(model):
    assert all(L.bias.abs().sum().item() == 0 for L in model.linear_layers)


# ---------------------------------------------------------------------------
# Dropout / BN behaviour
# ---------------------------------------------------------------------------


def test_dropout_active_in_train_mode_inactive_in_eval(model):
    """Dropout makes two train-mode forwards differ but eval-mode forwards
    identical (BN running stats already initialised at construction)."""
    X = torch.randn(16, INPUT_DIM)
    model.train()
    a = model(X)
    b = model(X)
    assert not torch.allclose(a, b)
    model.eval()
    c = model(X)
    d = model(X)
    assert torch.allclose(c, d)


def test_loss_decreases_over_steps(batch):
    """Adam(lr=1e-3) over 30 steps must drive loss strictly down from the
    initial value on a fixed synthetic batch."""
    torch.manual_seed(0)
    m = GestureMLP()
    X = torch.randn(32, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (32,), dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m.train()
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = crit(m(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_rejects_wrong_number_of_hidden_dims():
    with pytest.raises(ValueError, match="3 hidden layers"):
        GestureMLP(hidden_dims=(128, 64))


def test_rejects_invalid_dropout_probability():
    with pytest.raises(ValueError, match="dropout probabilities"):
        GestureMLP(dropouts=(0.3, 0.3, 1.5))


# ---------------------------------------------------------------------------
# DataLoader + schema (tmp_path NPZ; no real splits)
# ---------------------------------------------------------------------------


def test_build_dataloaders_yields_float32_and_long(tmp_path):
    _write_synthetic_npz(tmp_path / "train.npz", n=200, dim=INPUT_DIM,
                          num_classes=NUM_CLASSES, seed=1)
    _write_synthetic_npz(tmp_path / "val.npz", n=80, dim=INPUT_DIM,
                          num_classes=NUM_CLASSES, seed=2)
    train_loader, val_loader, scaler = train_mod.build_dataloaders(
        splits_dir=tmp_path, batch_size=16, num_workers=0, seed=0,
    )
    xb, yb = next(iter(train_loader))
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.int64
    assert xb.shape[1] == INPUT_DIM
    # Scaler fit on train only — mean/scale must be 279-long.
    assert scaler.mean_.shape == (INPUT_DIM,)
    assert scaler.scale_.shape == (INPUT_DIM,)
    # Val loader iterates without error.
    xv, yv = next(iter(val_loader))
    assert xv.dtype == torch.float32
    assert yv.dtype == torch.int64


def test_build_dataloaders_rejects_wrong_feature_dim(tmp_path):
    _write_synthetic_npz(tmp_path / "train.npz", n=50, dim=INPUT_DIM - 1,
                          num_classes=NUM_CLASSES, seed=3)
    _write_synthetic_npz(tmp_path / "val.npz", n=20, dim=INPUT_DIM - 1,
                          num_classes=NUM_CLASSES, seed=4)
    with pytest.raises(ValueError, match="feature dim"):
        train_mod.build_dataloaders(splits_dir=tmp_path, batch_size=16,
                                     num_workers=0, seed=0)


def test_build_dataloaders_rejects_invalid_label_ids(tmp_path):
    """A label outside ``data/labels.json`` -> schema rejection."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((50, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=50).astype(np.int32)
    y[0] = 99  # invalid id
    np.savez(tmp_path / "train.npz", X=X, y=y)
    _write_synthetic_npz(tmp_path / "val.npz", n=20, dim=INPUT_DIM,
                          num_classes=NUM_CLASSES, seed=6)
    with pytest.raises(ValueError, match="outside data/labels.json"):
        train_mod.build_dataloaders(splits_dir=tmp_path, batch_size=16,
                                     num_workers=0, seed=0)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def _make_log_row(epoch: int = 0) -> dict:
    return {
        "epoch": epoch,
        "train_loss": 1.0,
        "val_loss": 1.1,
        "train_acc": 0.5,
        "val_acc": 0.49,
        "merged_train_acc": 0.55,
        "merged_val_acc": 0.54,
        "val_macro_f1": 0.45,
        "grad_norm": 0.7,
        "layer_1_weight_norm": 1.2,
        "layer_2_weight_norm": 1.1,
        "layer_3_weight_norm": 0.9,
        "layer_4_weight_norm": 0.6,
        "lr": 1e-3,
        "wall_seconds": 1.0,
        "timestamp": "2026-05-14T00:00:00+00:00",
    }


def test_append_log_row_writes_required_columns(tmp_path):
    path = tmp_path / "training_log.csv"
    train_mod.append_log_row(path, _make_log_row(0))
    train_mod.append_log_row(path, _make_log_row(1))
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == train_mod.CSV_FIELDS
    assert len(rows) == 2


def test_append_log_row_rejects_missing_columns(tmp_path):
    path = tmp_path / "training_log.csv"
    bad = _make_log_row()
    bad.pop("grad_norm")
    with pytest.raises(ValueError, match="missing required keys"):
        train_mod.append_log_row(path, bad)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    m = GestureMLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = train_mod.make_scheduler(opt, patience=10, factor=0.5, min_lr=1e-5)

    # A "fitted" scaler: provide mean/scale arrays directly.
    scaler = type("S", (), {})()
    scaler.mean_ = np.zeros(INPUT_DIM, dtype=np.float32)
    scaler.scale_ = np.ones(INPUT_DIM, dtype=np.float32)

    cfg = {
        "input_dim": INPUT_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropouts": list(DROPOUTS),
        "num_classes": NUM_CLASSES,
    }
    metrics = {"val_loss": 1.0, "val_acc": 0.5, "merged_val_acc": 0.6, "val_macro_f1": 0.4}

    path = tmp_path / "mlp_best.pt"
    train_mod.save_checkpoint(
        path=path, model=m, optimizer=opt, scheduler=sched, scaler=scaler,
        epoch=7, metrics=metrics, config=cfg, seed=0, labels={0: "a"},
    )
    assert path.is_file()

    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 7
    assert ck["config"]["input_dim"] == INPUT_DIM
    assert ck["config"]["num_classes"] == NUM_CLASSES

    # Rebuild a fresh model from config + state dict, compare forward output.
    m2 = GestureMLP(
        input_dim=ck["config"]["input_dim"],
        hidden_dims=tuple(ck["config"]["hidden_dims"]),
        dropouts=tuple(ck["config"]["dropouts"]),
        num_classes=ck["config"]["num_classes"],
    )
    m2.load_state_dict(ck["model_state_dict"])
    m.eval(); m2.eval()
    X = torch.randn(4, INPUT_DIM)
    with torch.no_grad():
        assert torch.allclose(m(X), m2(X), atol=1e-6)
