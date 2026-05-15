"""Stage 6 MV-visualisation helper tests.

Fast tests use a freshly-constructed ``GestureMLP`` (random init) + synthetic
data. The single ``@pytest.mark.slow`` test hits the real test split +
checkpoint and is deselected by default.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import mv_visualization as mv
from src.models.mlp import INPUT_DIM, NUM_CLASSES, GestureMLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return GestureMLP()


@pytest.fixture
def small_batch():
    torch.manual_seed(1)
    X = torch.randn(64, INPUT_DIM, dtype=torch.float32)
    y = torch.randint(0, NUM_CLASSES, (64,), dtype=torch.long)
    return X, y


# ---------------------------------------------------------------------------
# Plot 1: loss surface
# ---------------------------------------------------------------------------


def test_loss_surface_grid_shape(small_model, small_batch):
    X, y = small_batch
    payload = mv.compute_loss_surface_slice(
        small_model, X, y, weight_indices=((5, 0), (5, 1)),
        grid_size=10, radius=0.5, device=torch.device("cpu"),
    )
    assert payload["loss_grid"].shape == (10, 10)
    assert payload["loss_grid"].dtype == np.float32
    # Trained loss should be a finite scalar inside the grid range.
    assert math.isfinite(payload["trained_loss"])
    grid_min = float(payload["loss_grid"].min())
    grid_max = float(payload["loss_grid"].max())
    assert grid_min - 1e-3 <= payload["trained_loss"] <= grid_max + 1e-3


def test_loss_surface_restores_weights(small_model, small_batch):
    X, y = small_batch
    before = small_model.linears[0].weight.detach().clone()
    _ = mv.compute_loss_surface_slice(
        small_model, X, y, weight_indices=((5, 0), (5, 1)),
        grid_size=6, radius=0.5,
    )
    after = small_model.linears[0].weight.detach()
    assert torch.allclose(before, after, atol=0.0)


def test_loss_surface_rejects_bad_args(small_model, small_batch):
    X, y = small_batch
    with pytest.raises(ValueError, match="grid_size"):
        mv.compute_loss_surface_slice(small_model, X, y, ((0, 0), (0, 1)),
                                       grid_size=1)
    with pytest.raises(ValueError, match="radius"):
        mv.compute_loss_surface_slice(small_model, X, y, ((0, 0), (0, 1)),
                                       grid_size=5, radius=0.0)


def test_plot_loss_surface_returns_figure(small_model, small_batch, tmp_path):
    X, y = small_batch
    payload = mv.compute_loss_surface_slice(
        small_model, X, y, weight_indices=((5, 0), (5, 1)),
        grid_size=8, radius=0.5,
    )
    out = tmp_path / "loss_surface.png"
    fig = mv.plot_loss_surface(payload, output_path=out, kind="contourf")
    assert fig is not None
    assert out.is_file() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Plot 2: gradient norm
# ---------------------------------------------------------------------------


def _write_synthetic_training_log(path: Path, n: int = 20) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for e in range(n):
        rows.append({
            "epoch": e,
            "train_loss": 1.0 - e * 0.02,
            "val_loss": 1.0 - e * 0.02 + rng.standard_normal() * 0.01,
            "train_acc": 0.5 + e * 0.01,
            "val_acc": 0.5 + e * 0.01,
            "merged_train_acc": 0.6 + e * 0.01,
            "merged_val_acc": 0.6 + e * 0.01,
            "val_macro_f1": 0.4 + e * 0.01,
            "grad_norm": max(0.001, 5.0 / (e + 1)),
            "layer_1_weight_norm": 20.0 - e * 0.1,
            "layer_2_weight_norm": 15.0 - e * 0.1,
            "layer_3_weight_norm": 10.0 - e * 0.1,
            "layer_4_weight_norm": 8.0 - e * 0.1,
            "lr": 1e-3,
            "wall_seconds": e * 60.0,
            "timestamp": f"2026-05-14T00:{e:02d}:00+00:00",
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_plot_gradient_norm_returns_figure_and_df(tmp_path):
    log = tmp_path / "training_log.csv"
    _write_synthetic_training_log(log)
    out = tmp_path / "grad_norm.png"
    fig, df = mv.plot_gradient_norm(log, output_path=out)
    assert fig is not None
    assert out.is_file() and out.stat().st_size > 0
    assert {"epoch", "grad_norm", "val_loss"} <= set(df.columns)


def test_plot_gradient_norm_rejects_missing_columns(tmp_path):
    log = tmp_path / "bad_log.csv"
    pd.DataFrame({"epoch": [0, 1], "val_loss": [1.0, 0.5]}).to_csv(log, index=False)
    with pytest.raises(KeyError, match="grad_norm"):
        mv.plot_gradient_norm(log)


# ---------------------------------------------------------------------------
# Plot 3: PCA
# ---------------------------------------------------------------------------


def test_compute_input_pca_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=200).astype(np.int64)
    payload = mv.compute_input_pca(X, y, subsample=100, seed=0)
    assert payload["coords"].shape == (100, 2)
    assert payload["explained_variance_ratio"].shape == (2,)
    evr = payload["explained_variance_ratio"]
    assert 0.0 <= evr[0] <= 1.0
    assert 0.0 <= evr[1] <= 1.0
    assert float(evr.sum()) <= 1.0 + 1e-6


def test_compute_input_pca_deterministic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=200).astype(np.int64)
    a = mv.compute_input_pca(X, y, subsample=100, seed=0)
    b = mv.compute_input_pca(X, y, subsample=100, seed=0)
    assert np.array_equal(a["coords"], b["coords"])
    assert np.array_equal(a["labels"], b["labels"])


def test_compute_input_pca_uses_training_scaler():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, INPUT_DIM)).astype(np.float32) * 3.0 + 2.0
    y = rng.integers(0, NUM_CLASSES, size=150).astype(np.int64)
    mean = np.full(INPUT_DIM, 2.0, dtype=np.float32)
    scale = np.full(INPUT_DIM, 3.0, dtype=np.float32)
    payload = mv.compute_input_pca(
        X, y, standardize_with=(mean, scale), subsample=100, seed=0,
    )
    # Standardised X should be near zero-mean unit-variance; PCA coords
    # should be on the order of single digits, not tens.
    assert payload["coords"].max() < 100.0


# ---------------------------------------------------------------------------
# Plot 4: chain rule
# ---------------------------------------------------------------------------


def test_select_active_first_layer_neuron_deterministic(small_model):
    torch.manual_seed(7)
    X = torch.randn(128, INPUT_DIM, dtype=torch.float32)
    a = mv.select_active_first_layer_neuron(small_model, X, seed=0)
    b = mv.select_active_first_layer_neuron(small_model, X, seed=0)
    assert a == b


def test_chain_rule_controlled_toy_model():
    """A linear -> ReLU -> linear toy model: manual chain rule is trivial and
    must match autograd to ~1e-7."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 2, bias=False),
    )
    model.eval()
    # Force at least one active neuron for an interesting check.
    with torch.no_grad():
        model[0].weight.data = torch.tensor(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float32
        )
        model[2].weight.data = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
    crit = nn.CrossEntropyLoss()

    for x_val in (torch.tensor([[0.5, 0.5, 0.5]]),
                  torch.tensor([[1.0, 2.0, -0.5]]),
                  torch.tensor([[0.3, 0.1, 0.2]])):
        x = x_val
        y_true = torch.tensor([0], dtype=torch.long)
        # Forward.
        z = model[0](x)
        a = model[2](F.relu(z))
        loss = crit(a, y_true)
        # Manual derivative for W^{(1)}[0, 0]:
        # dL/dW0[0,0] = x[0] * (1 if z[0]>0 else 0) * dL/da^(1)_0
        (grad_a,) = torch.autograd.grad(loss, z, retain_graph=True)
        # The "post-ReLU activation" gradient is grad_a only after we
        # multiply by the ReLU gate. autograd.grad on z already pushes
        # through the ReLU, so re-derive cleanly: use the post-ReLU as
        # the captured tensor.
        a1 = F.relu(z)
        a1_id = a1.detach().clone().requires_grad_(True)
        a_logits = model[2](a1_id)
        loss2 = crit(a_logits, y_true)
        (grad_a1,) = torch.autograd.grad(loss2, a1_id)
        upstream = float(grad_a1[0, 0].item())
        gate = 1.0 if float(z[0, 0].item()) > 0 else 0.0
        manual = float(x[0, 0].item()) * gate * upstream
        # Now compare to autograd's W gradient.
        model.zero_grad(set_to_none=True)
        loss_ref = crit(model[2](F.relu(model[0](x))), y_true)
        loss_ref.backward()
        autograd_val = float(model[0].weight.grad[0, 0].item())
        assert abs(manual - autograd_val) < 1e-6, (manual, autograd_val)


def test_chain_rule_with_bn_real_arch(small_model):
    torch.manual_seed(11)
    X = torch.randn(64, INPUT_DIM, dtype=torch.float32)
    y = torch.randint(0, NUM_CLASSES, (64,), dtype=torch.long)
    i, j = mv.select_active_first_layer_neuron(small_model, X)
    saw_active = False
    for k in range(5):
        row = mv.verify_chain_rule_single_sample(
            small_model, X[k : k + 1], int(y[k].item()),
            neuron_idx=i, input_idx=j,
        )
        assert row["abs_error"] < 1e-5, row
        if row["gate_active"]:
            saw_active = True
    assert saw_active, "selection rule failed to find an active gate"


def test_chain_rule_batch_requires_ten_passes(small_model):
    torch.manual_seed(13)
    X = torch.randn(64, INPUT_DIM, dtype=torch.float32)
    y = torch.randint(0, NUM_CLASSES, (64,), dtype=torch.long)
    i, j = mv.select_active_first_layer_neuron(small_model, X)
    indices = list(range(12))
    df = mv.verify_chain_rule_batch(
        small_model, X, y, indices, neuron_idx=i, input_idx=j,
    )
    assert len(df) == 12
    assert int(df["passed"].sum()) >= 10


def test_verify_chain_rule_batch_raises_when_under_tol(small_model):
    torch.manual_seed(13)
    X = torch.randn(64, INPUT_DIM, dtype=torch.float32)
    y = torch.randint(0, NUM_CLASSES, (64,), dtype=torch.long)
    i, j = mv.select_active_first_layer_neuron(small_model, X)
    with pytest.raises(AssertionError, match="chain-rule verification"):
        mv.verify_chain_rule_batch(
            small_model, X, y, list(range(12)),
            neuron_idx=i, input_idx=j, tol=1e-30,
        )


def test_verify_chain_rule_single_sample_rejects_wrong_shape(small_model):
    with pytest.raises(ValueError, match="shape"):
        mv.verify_chain_rule_single_sample(
            small_model, torch.randn(2, INPUT_DIM), y_true=0,
            neuron_idx=5, input_idx=0,
        )


# ---------------------------------------------------------------------------
# Slow: real checkpoint + real test split
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_chain_rule_real_checkpoint_smoke():
    ckpt_path = Path("runs/mlp_best.pt")
    test_path = Path("data/splits/test.npz")
    if not (ckpt_path.is_file() and test_path.is_file()):
        pytest.skip("real checkpoint / test split not present")

    from src.evaluate import load_model_checkpoint
    model, scaler, label_map, _ = load_model_checkpoint(
        ckpt_path, labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )

    with np.load(test_path, allow_pickle=True) as z:
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)

    X_std = scaler.transform(X).astype(np.float32)
    X_t = torch.from_numpy(X_std)
    y_t = torch.from_numpy(y)

    probe = X_t[:512]
    i, j = mv.select_active_first_layer_neuron(model, probe)
    indices = mv.pick_active_sample_indices(model, X_t, neuron_idx=i, n_samples=12)
    df = mv.verify_chain_rule_batch(
        model, X_t, y_t, indices, neuron_idx=i, input_idx=j,
        label_map=label_map,
    )
    assert int(df["passed"].sum()) >= 10
    assert int((df["gate_active"]).sum()) >= 8
