"""Tests for src.loss_landscape.

Fast tests use a freshly-constructed ``GestureMLP`` + synthetic data — no
real checkpoint or split required. The single ``@pytest.mark.slow`` test
exercises the real ``runs/mlp_best.pt`` + ``data/splits/val.npz`` with a
small grid.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import loss_landscape as ll
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


@pytest.fixture
def synthetic_val_npz(tmp_path: Path) -> Path:
    """Write a small synthetic val.npz that mimics the Stage 2 schema."""
    rng = np.random.default_rng(42)
    n_per_class = 12
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    y = np.repeat(classes, n_per_class).astype(np.int32)
    n = y.shape[0]
    X = rng.standard_normal((n, INPUT_DIM)).astype(np.float32)
    is_synth = np.zeros(n, dtype=bool)
    is_aug = np.zeros(n, dtype=bool)
    path = tmp_path / "val.npz"
    np.savez(
        path,
        X=X,
        y=y,
        is_synthetic=is_synth,
        is_augmented=is_aug,
    )
    return path


# ---------------------------------------------------------------------------
# Direction generation
# ---------------------------------------------------------------------------


def test_directions_shape_and_keys(small_model):
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=123)
    expected = dict(small_model.named_parameters())
    assert set(d1) == set(expected)
    assert set(d2) == set(expected)
    for name, p in expected.items():
        assert d1[name].shape == p.shape
        assert d2[name].shape == p.shape


def test_directions_filter_normalized_rows(small_model):
    d1, _ = ll.generate_filter_normalized_directions(small_model, seed=7)
    named = dict(small_model.named_parameters())
    for k in range(4):
        name = f"linears.{k}.weight"
        d_rows = d1[name].norm(dim=1)
        w_rows = named[name].detach().norm(dim=1)
        # Every row's L2 norm of d must match the trained row's L2 norm.
        assert torch.allclose(d_rows, w_rows, atol=1e-5, rtol=1e-4), (
            f"linear {k}: max diff "
            f"{(d_rows - w_rows).abs().max().item()}"
        )


def test_directions_bn_params_are_zero(small_model):
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=11)
    bn_keys = [n for n, _ in small_model.named_parameters() if n.startswith("bns.")]
    assert bn_keys, "GestureMLP should have BN affine parameters"
    for name in bn_keys:
        assert torch.equal(d1[name], torch.zeros_like(d1[name]))
        assert torch.equal(d2[name], torch.zeros_like(d2[name]))


def test_directions_determinism(small_model):
    d1a, d2a = ll.generate_filter_normalized_directions(small_model, seed=2026)
    d1b, d2b = ll.generate_filter_normalized_directions(small_model, seed=2026)
    for name in d1a:
        assert torch.equal(d1a[name], d1b[name])
        assert torch.equal(d2a[name], d2b[name])


def test_directions_zero_for_dead_rows(small_model):
    # Zero out two rows of linears[0].weight to simulate dead neurons. Filter
    # normalisation should give the same zero rows in d1, d2.
    with torch.no_grad():
        small_model.linears[0].weight[3].zero_()
        small_model.linears[0].weight[17].zero_()
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=99)
    for d in (d1, d2):
        assert torch.allclose(
            d["linears.0.weight"][3], torch.zeros_like(d["linears.0.weight"][3])
        )
        assert torch.allclose(
            d["linears.0.weight"][17], torch.zeros_like(d["linears.0.weight"][17])
        )
    # Sanity-check the helper surfaces the count.
    ratios, dead = ll._per_linear_row_norm_ratios(small_model, d1)
    assert dead[0] == 2
    assert dead[1] == dead[2] == dead[3] == 0
    # Alive-row ratio still 1.0 despite the dead rows.
    assert ratios[0] == pytest.approx(1.0, abs=1e-4)


def test_directions_differ_across_seeds(small_model):
    d1a, _ = ll.generate_filter_normalized_directions(small_model, seed=1)
    d1b, _ = ll.generate_filter_normalized_directions(small_model, seed=2)
    # At least one non-BN parameter must differ.
    differ = False
    for name in d1a:
        if name.startswith("bns."):
            continue
        if not torch.equal(d1a[name], d1b[name]):
            differ = True
            break
    assert differ, "different seeds should produce different directions"


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def test_evaluate_radius_zero_grid_is_anchor(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    # radius=0 isn't allowed (>0 required), but a tiny radius with grid_size=3
    # and central cell should match anchor exactly.
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=3, radius=1e-6,
        device=torch.device("cpu"),
    )
    grid = payload["loss_grid"]
    # All cells should be near the anchor (tiny radius).
    assert np.allclose(grid, payload["anchor_loss"], atol=1e-3)


def test_evaluate_smoke_5x5(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=5, radius=0.1,
        device=torch.device("cpu"),
        eval_split="synthetic",
    )
    grid = payload["loss_grid"]
    assert grid.shape == (5, 5)
    assert np.isfinite(grid).all()
    assert grid.dtype == np.float32
    # Central cell of an odd grid should equal anchor_loss within 1e-4.
    mid = 5 // 2
    assert abs(float(grid[mid, mid]) - payload["anchor_loss"]) < 1e-4
    # Axes symmetric around 0.
    assert payload["a_axis"][0] == pytest.approx(-0.1)
    assert payload["a_axis"][-1] == pytest.approx(0.1)
    # Provenance fields populated.
    assert isinstance(payload["state_dict_sha256"], str)
    assert payload["state_dict_sha256"]
    assert payload["eval_split"] == "synthetic"
    assert payload["n_eval"] == X.shape[0]
    assert payload["direction_norm_ratios"].shape == (4,)
    # Filter-normalisation invariant: alive-row ratio is exactly 1.
    assert np.allclose(payload["direction_norm_ratios"], 1.0, atol=1e-4)
    # Fresh GestureMLP starts with no dead rows.
    assert payload["dead_rows_per_linear"].shape == (4,)
    assert (payload["dead_rows_per_linear"] == 0).all()


def test_evaluate_state_restoration(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    before_hash = ll._state_dict_sha256(small_model)
    snapshots = {
        name: p.detach().clone() for name, p in small_model.named_parameters()
    }
    _ = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=5, radius=0.3,
        device=torch.device("cpu"),
    )
    after_hash = ll._state_dict_sha256(small_model)
    assert before_hash == after_hash
    for name, p in small_model.named_parameters():
        assert torch.equal(p.detach(), snapshots[name])


def test_evaluate_rejects_bad_args(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    with pytest.raises(ValueError, match="grid_size"):
        ll.evaluate_loss_landscape(small_model, X, y, d1, d2,
                                   grid_size=2, radius=0.1)
    with pytest.raises(ValueError, match="radius"):
        ll.evaluate_loss_landscape(small_model, X, y, d1, d2,
                                   grid_size=5, radius=0.0)
    with pytest.raises(ValueError, match="rows"):
        ll.evaluate_loss_landscape(small_model, X[:32], y, d1, d2,
                                   grid_size=5, radius=0.1)


def test_evaluate_rejects_missing_direction_keys(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    d2_bad = {k: v for k, v in d2.items() if not k.startswith("linears.3.")}
    with pytest.raises(ValueError, match="missing keys"):
        ll.evaluate_loss_landscape(small_model, X, y, d1, d2_bad,
                                   grid_size=5, radius=0.1)


def test_evaluate_large_radius_warns(small_model, small_batch):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    with pytest.warns(RuntimeWarning, match="radius"):
        ll.evaluate_loss_landscape(small_model, X, y, d1, d2,
                                   grid_size=3, radius=2.0)


# ---------------------------------------------------------------------------
# Eval-subset selector
# ---------------------------------------------------------------------------


def test_select_eval_subset_stratified(synthetic_val_npz):
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    scale = np.ones(INPUT_DIM, dtype=np.float32)
    X_std, y = ll.select_eval_subset(
        synthetic_val_npz, mean, scale,
        n_samples=2 * NUM_CLASSES, seed=0,
    )
    assert X_std.shape == (2 * NUM_CLASSES, INPUT_DIM)
    assert X_std.dtype == torch.float32
    assert y.dtype == torch.int64
    # Every class must appear at least once in the stratified subset.
    assert set(int(v) for v in y.unique().tolist()) == set(range(NUM_CLASSES))


def test_select_eval_subset_determinism(synthetic_val_npz):
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    scale = np.ones(INPUT_DIM, dtype=np.float32)
    X1, y1 = ll.select_eval_subset(synthetic_val_npz, mean, scale,
                                   n_samples=56, seed=123)
    X2, y2 = ll.select_eval_subset(synthetic_val_npz, mean, scale,
                                   n_samples=56, seed=123)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_select_eval_subset_rejects_too_many(synthetic_val_npz):
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    scale = np.ones(INPUT_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match="n_samples"):
        ll.select_eval_subset(synthetic_val_npz, mean, scale,
                              n_samples=10_000, seed=0)


def test_select_eval_subset_applies_scaler(synthetic_val_npz):
    # Use a scaler that shifts by 5 and scales by 2; standardised values
    # should match the manual transform.
    mean = 5.0 * np.ones(INPUT_DIM, dtype=np.float32)
    scale = 2.0 * np.ones(INPUT_DIM, dtype=np.float32)
    X_std, _ = ll.select_eval_subset(synthetic_val_npz, mean, scale,
                                     n_samples=NUM_CLASSES * 2, seed=0)
    # Mean of standardised data should be roughly (raw_mean - 5) / 2.
    # Synthetic raw mean is ~0, so standardised mean ~ -2.5.
    assert float(X_std.mean().item()) == pytest.approx(-2.5, abs=0.6)


# ---------------------------------------------------------------------------
# NPZ caching
# ---------------------------------------------------------------------------


def test_save_load_landscape_npz_roundtrip(small_model, small_batch, tmp_path):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=5, radius=0.1, eval_split="synthetic", seed=5,
        device=torch.device("cpu"),
    )
    path = tmp_path / "landscape.npz"
    ll.save_landscape_npz(payload, path)
    assert path.is_file()
    restored = ll.load_landscape_npz(path)
    np.testing.assert_allclose(restored["a_axis"], payload["a_axis"])
    np.testing.assert_allclose(restored["b_axis"], payload["b_axis"])
    np.testing.assert_allclose(restored["loss_grid"], payload["loss_grid"])
    assert restored["anchor_loss"] == pytest.approx(payload["anchor_loss"])
    assert restored["state_dict_sha256"] == payload["state_dict_sha256"]
    assert restored["eval_split"] == payload["eval_split"]
    assert restored["seed"] == payload["seed"]
    np.testing.assert_array_equal(
        restored["dead_rows_per_linear"], payload["dead_rows_per_linear"]
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def test_plot_contour_writes_png(small_model, small_batch, tmp_path):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=5, radius=0.1, device=torch.device("cpu"),
    )
    out = tmp_path / "contour.png"
    fig = ll.plot_landscape_contour(payload, output_path=out)
    assert out.is_file()
    assert out.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_surface_writes_png(small_model, small_batch, tmp_path):
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=5, radius=0.1, device=torch.device("cpu"),
    )
    out = tmp_path / "surface.png"
    fig = ll.plot_landscape_surface(payload, output_path=out)
    assert out.is_file()
    assert out.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-D descent path
# ---------------------------------------------------------------------------


def _bowl_payload() -> dict:
    """A synthetic convex bowl ``L(a,b) = a^2 + b^2`` on a 21x21 grid in
    ``[-1, 1]^2``. Lets us assert exact behaviour of the descent (monotone
    decreasing loss, converges to the centre)."""
    a = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    b = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    A, B = np.meshgrid(a, b, indexing="ij")
    Z = (A * A + B * B).astype(np.float32)
    return {
        "a_axis": a, "b_axis": b, "loss_grid": Z,
        "anchor_loss": 0.0, "radius": 1.0, "grid_size": 21,
        "n_eval": 1, "eval_split": "synthetic", "seed": 0,
        "state_dict_sha256": "x" * 64,
        "direction_norm_ratios": np.ones(4),
        "dead_rows_per_linear": np.zeros(4, dtype=np.int64),
        "note": "bowl",
    }


def test_descent_path_monotone_decreasing():
    pl = _bowl_payload()
    out = ll.compute_descent_path_2d(
        pl, start=(-0.8, 0.8), learning_rate=0.1, n_steps=50, tolerance=1e-6,
    )
    path = out["path"]
    losses = path[:, 2]
    # On a quadratic bowl the descent should be monotone non-increasing.
    diffs = np.diff(losses)
    assert (diffs <= 1e-9).all(), f"non-monotone: max diff {diffs.max()}"
    # Final point should be very near the origin.
    assert abs(path[-1, 0]) < 0.1
    assert abs(path[-1, 1]) < 0.1
    assert path[-1, 2] < 0.05


def test_descent_path_records_start():
    pl = _bowl_payload()
    out = ll.compute_descent_path_2d(
        pl, start=(0.5, -0.5), learning_rate=0.05, n_steps=10,
    )
    assert out["start"] == (0.5, -0.5)
    assert out["path"][0, 0] == pytest.approx(0.5)
    assert out["path"][0, 1] == pytest.approx(-0.5)
    assert out["learning_rate"] == 0.05
    assert out["n_steps_taken"] == out["path"].shape[0] - 1


def test_descent_path_clamps_to_grid():
    """Aggressive learning rate would overshoot the bowl in one step; the
    helper must clamp every iterate inside the grid."""
    pl = _bowl_payload()
    out = ll.compute_descent_path_2d(
        pl, start=(-0.9, 0.9), learning_rate=5.0, n_steps=20,
    )
    p = out["path"]
    assert (p[:, 0] >= pl["a_axis"][0] - 1e-9).all()
    assert (p[:, 0] <= pl["a_axis"][-1] + 1e-9).all()
    assert (p[:, 1] >= pl["b_axis"][0] - 1e-9).all()
    assert (p[:, 1] <= pl["b_axis"][-1] + 1e-9).all()


def test_descent_path_rejects_bad_args():
    pl = _bowl_payload()
    with pytest.raises(ValueError, match="learning_rate"):
        ll.compute_descent_path_2d(pl, start=(0.0, 0.0), learning_rate=0.0)
    with pytest.raises(ValueError, match="n_steps"):
        ll.compute_descent_path_2d(pl, start=(0.0, 0.0), n_steps=0)
    with pytest.raises(ValueError, match="outside the grid"):
        ll.compute_descent_path_2d(pl, start=(5.0, 0.0))


def test_descent_path_determinism():
    pl = _bowl_payload()
    a = ll.compute_descent_path_2d(pl, start=(-0.7, 0.4), learning_rate=0.1, n_steps=30)
    b = ll.compute_descent_path_2d(pl, start=(-0.7, 0.4), learning_rate=0.1, n_steps=30)
    np.testing.assert_array_equal(a["path"], b["path"])


def test_plot_overlay_with_descent_path(small_model, small_batch, tmp_path):
    """Both plot helpers accept ``descent_path`` and still write a PNG."""
    X, y = small_batch
    d1, d2 = ll.generate_filter_normalized_directions(small_model, seed=5)
    payload = ll.evaluate_loss_landscape(
        small_model, X, y, d1, d2,
        grid_size=7, radius=0.2, device=torch.device("cpu"),
    )
    descent = ll.compute_descent_path_2d(
        payload, start=(-0.1, 0.1), learning_rate=0.01, n_steps=5,
    )
    out_c = tmp_path / "contour_with_path.png"
    out_s = tmp_path / "surface_with_path.png"
    fig_c = ll.plot_landscape_contour(payload, output_path=out_c, descent_path=descent)
    fig_s = ll.plot_landscape_surface(payload, output_path=out_s, descent_path=descent)
    assert out_c.stat().st_size > 0
    assert out_s.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(fig_c)
    plt.close(fig_s)


# ---------------------------------------------------------------------------
# Slow: real checkpoint + real val split
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_checkpoint_smoke():
    ckpt_path = Path("runs/mlp_best.pt")
    val_path = Path("data/splits/val.npz")
    if not (ckpt_path.is_file() and val_path.is_file()):
        pytest.skip("real checkpoint / val split not present")

    from src.evaluate import load_model_checkpoint

    model, scaler, _label_map, ck = load_model_checkpoint(
        ckpt_path,
        labels_path=Path("data/labels.json"),
        device=torch.device("cpu"),
    )
    X_std, y = ll.select_eval_subset(
        val_path,
        scaler_mean=ck["scaler_mean"],
        scaler_scale=ck["scaler_scale"],
        n_samples=500,
        seed=20260514,
    )
    d1, d2 = ll.generate_filter_normalized_directions(model, seed=20260514)
    payload = ll.evaluate_loss_landscape(
        model, X_std, y, d1, d2,
        grid_size=11, radius=0.5,
        device=torch.device("cpu"),
        eval_split="val",
        seed=20260514,
    )
    grid = payload["loss_grid"]
    assert grid.shape == (11, 11)
    assert np.isfinite(grid).all()
    # Filter-norm sanity: alive-row ratio is ~1.0 even when there are dead rows.
    assert np.allclose(payload["direction_norm_ratios"], 1.0, atol=1e-3)
    # Surface dead-row count for the audit trail.
    assert payload["dead_rows_per_linear"].shape == (4,)
    assert (payload["dead_rows_per_linear"] >= 0).all()
    # Anchor should be near the minimum of this random subspace slice — at
    # most a small fraction above the absolute minimum on the grid.
    mid = grid.shape[0] // 2
    anchor = float(grid[mid, mid])
    assert anchor == pytest.approx(payload["anchor_loss"], abs=1e-4)
    # On a well-trained model, theta_best should lie at or near a minimum
    # of any random 2-d slice — the centre cell should be within 0.5 of the
    # grid minimum (very loose; the slice may extend uphill in any direction).
    assert anchor - float(grid.min()) < 0.5
