"""Stage 6 multivariable-calculus visualisation helpers.

Pure functions consumed by ``notebooks/03_mv_visualization.ipynb``:

    1. ``compute_loss_surface_slice`` / ``plot_loss_surface``  — MV 2.6 level sets
    2. ``plot_gradient_norm``                                  — critical-point approach
    3. ``compute_input_pca`` / ``plot_input_pca``              — R^{279} structure
    4. ``select_active_first_layer_neuron``
       + ``verify_chain_rule_single_sample``
       + ``verify_chain_rule_batch``                           — chain-rule verification

Every helper returns the underlying arrays (or a DataFrame) alongside any
``matplotlib`` figure, so the helpers can be unit-tested without rendering.

Authoritative spec:
    - tasks/gesture_recognition_plan_v2.md §6.4
    - tasks/implementation_stages.md Stage 6
    - plan at C:/Users/Harry T/.claude/plans/you-are-claude-opus-snappy-ritchie.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.mlp import INPUT_DIM, NUM_CLASSES, GestureMLP
from src.train import DEFAULT_SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOSS_GRID_SIZE = 50
DEFAULT_LOSS_RADIUS = 0.5  # in units of |trained value| (or 1e-3 floor)
DEFAULT_PROBE_BATCH_SIZE = 512
DEFAULT_PCA_SUBSAMPLE = 5000
CHAIN_RULE_TOL_DEFAULT = 1e-5
CHAIN_RULE_REQUIRED_PASSES = 10


# ---------------------------------------------------------------------------
# Plot 1: Loss surface slice
# ---------------------------------------------------------------------------


def compute_loss_surface_slice(
    model: GestureMLP,
    X_std: torch.Tensor,
    y: torch.Tensor,
    weight_indices: tuple[tuple[int, int], tuple[int, int]],
    grid_size: int = DEFAULT_LOSS_GRID_SIZE,
    radius: float = DEFAULT_LOSS_RADIUS,
    device: torch.device | None = None,
) -> dict:
    """Compute L(w_a, w_b) on a grid centred at the trained values.

    Two scalar weights in ``model.linears[0].weight`` are varied (indexed by
    ``weight_indices = ((i_a, j_a), (i_b, j_b))``); all other parameters are
    held at their trained values. Uses ``model.eval()`` so BatchNorm uses
    running statistics (the slice is deterministic and pedagogically clean).
    Modifies the weights in-place inside ``torch.no_grad()`` and restores them
    in a ``try/finally`` block — no state-dict copies needed.
    """
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    (ia, ja), (ib, jb) = weight_indices
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    W = model.linears[0].weight  # shape (256, 279)
    trained_w1 = float(W[ia, ja].item())
    trained_w2 = float(W[ib, jb].item())

    scale1 = max(abs(trained_w1), 1e-3) * radius
    scale2 = max(abs(trained_w2), 1e-3) * radius
    w1_axis = trained_w1 + np.linspace(-1.0, 1.0, grid_size) * scale1
    w2_axis = trained_w2 + np.linspace(-1.0, 1.0, grid_size) * scale2

    X = X_std.to(device)
    yt = y.to(device).long()
    loss_grid = np.empty((grid_size, grid_size), dtype=np.float32)

    try:
        with torch.no_grad():
            # Trained-point loss for the contour overlay.
            logits = model(X)
            trained_loss = float(F.cross_entropy(logits, yt).item())

            for a in range(grid_size):
                W[ia, ja] = float(w1_axis[a])
                for b in range(grid_size):
                    W[ib, jb] = float(w2_axis[b])
                    logits = model(X)
                    loss_grid[a, b] = float(F.cross_entropy(logits, yt).item())
    finally:
        # Always restore both scalars.
        with torch.no_grad():
            W[ia, ja] = trained_w1
            W[ib, jb] = trained_w2
        if was_training:
            model.train()

    return {
        "w1_axis": w1_axis.astype(np.float64),
        "w2_axis": w2_axis.astype(np.float64),
        "loss_grid": loss_grid,
        "trained_w1": trained_w1,
        "trained_w2": trained_w2,
        "trained_loss": trained_loss,
        "weight_indices": [(ia, ja), (ib, jb)],
        "n_eval": int(X.shape[0]),
        "note": (
            "Trajectory overlay omitted: training_log.csv logs Frobenius norms "
            "only, not per-weight trajectories. Stage 6 does not retrain."
        ),
    }


def plot_loss_surface(
    payload: dict,
    output_path: Path | None = None,
    kind: str = "contourf",
    title: str = "Loss surface slice (first-layer 2-weight subspace)",
) -> Figure:
    """Render the loss surface payload from :func:`compute_loss_surface_slice`."""
    w1 = payload["w1_axis"]
    w2 = payload["w2_axis"]
    Z = payload["loss_grid"]
    (ia, ja), (ib, jb) = payload["weight_indices"]
    W1, W2 = np.meshgrid(w1, w2, indexing="ij")

    if kind == "surface":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.85)
        ax.set_xlabel(f"W1[{ia},{ja}]")
        ax.set_ylabel(f"W1[{ib},{jb}]")
        ax.set_zlabel("loss")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        c = ax.contourf(W1, W2, Z, levels=30, cmap="viridis")
        cs = ax.contour(W1, W2, Z, levels=10, colors="black",
                        linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
        ax.plot(payload["trained_w1"], payload["trained_w2"],
                "r*", markersize=12, label="trained")
        ax.set_xlabel(f"W1[{ia},{ja}]")
        ax.set_ylabel(f"W1[{ib},{jb}]")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(c, ax=ax, label="cross-entropy loss")

    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Gradient norm vs. epoch
# ---------------------------------------------------------------------------


def plot_gradient_norm(
    training_log_csv: Path,
    output_path: Path | None = None,
    use_log_y: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """Plot ``grad_norm`` against ``epoch`` from ``runs/training_log.csv``."""
    df = pd.read_csv(training_log_csv)
    for col in ("epoch", "grad_norm", "val_loss"):
        if col not in df.columns:
            raise KeyError(f"{training_log_csv} missing required column '{col}'")

    best_epoch = int(df["val_loss"].idxmin())

    # Monotone fraction over final 20 epochs (matches notebooks/02_*).
    tail = df.tail(min(20, len(df)))
    diffs = np.diff(tail["grad_norm"].to_numpy())
    monotone_frac = float((diffs <= 0).mean()) if diffs.size else float("nan")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["grad_norm"], label="grad_norm", linewidth=1.1)
    if use_log_y and (df["grad_norm"] > 0).all():
        ax.set_yscale("log")
    ax.axvline(best_epoch, linestyle="--", color="C3", alpha=0.7,
               label=f"best val_loss @ epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("||grad L||_2  (last batch of epoch)")
    ax.set_title("Gradient norm vs. epoch — approach to a critical point of L")
    ax.legend(loc="best", fontsize=9)
    ax.text(
        0.02, 0.05,
        f"final-20-epoch monotone fraction = {monotone_frac:.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig, df


# ---------------------------------------------------------------------------
# Plot 3: PCA of input space
# ---------------------------------------------------------------------------


def compute_input_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    standardize_with: tuple[np.ndarray, np.ndarray] | None = None,
    subsample: int | None = DEFAULT_PCA_SUBSAMPLE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Standardise (with the training scaler) + (stratified) subsample + PCA.

    Pass ``standardize_with = (mean, scale)`` from the checkpoint to avoid
    refitting a scaler on test data; otherwise the input is treated as already
    standardised.
    """
    if standardize_with is not None:
        mean, scale = standardize_with
        X_std = (X - np.asarray(mean, dtype=np.float32)) / np.asarray(
            scale, dtype=np.float32
        )
        X_std = X_std.astype(np.float32, copy=False)
    else:
        X_std = X.astype(np.float32, copy=False)

    if subsample is not None and subsample < X_std.shape[0]:
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=subsample, random_state=seed
        )
        idx, _ = next(splitter.split(X_std, y))
        X_sub = X_std[idx]
        y_sub = y[idx]
    else:
        X_sub, y_sub = X_std, y

    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(X_sub)

    return {
        "coords": coords.astype(np.float32),
        "labels": np.asarray(y_sub).astype(np.int64),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
        "n_components": int(n_components),
        "subsample_size": int(X_sub.shape[0]),
        "seed": int(seed),
    }


def _build_28_class_colors() -> np.ndarray:
    """Return a (28, 4) RGBA colour table by blending tab20 + tab20b."""
    cmap_a = plt.get_cmap("tab20")
    cmap_b = plt.get_cmap("tab20b")
    rows = [cmap_a(i) for i in range(20)] + [cmap_b(i) for i in range(8)]
    return np.array(rows)


def plot_input_pca(
    payload: dict,
    label_map: dict[int, str],
    output_path: Path | None = None,
) -> Figure:
    """Scatter PC1 vs PC2 coloured by class label."""
    coords = payload["coords"]
    labels = payload["labels"]
    evr = payload["explained_variance_ratio"]
    colors = _build_28_class_colors()

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in sorted(set(int(c) for c in np.unique(labels))):
        m = labels == cls
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=6, alpha=0.5,
            color=colors[cls % len(colors)],
            label=label_map.get(cls, str(cls)),
            linewidths=0,
        )
    ax.set_xlabel(f"PC1 (var={evr[0]:.3f})")
    ax.set_ylabel(f"PC2 (var={evr[1]:.3f})")
    ax.set_title(
        f"Test input space PCA (R^279 -> R^2)  -  "
        f"PC1+PC2 explain {evr.sum():.3f} of variance"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Chain rule verification
# ---------------------------------------------------------------------------


def _eval_forward_capture(
    model: GestureMLP, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a manual forward pass capturing (z1, u1, a1, logits) for the chain
    rule trace. Assumes ``model.eval()`` so BatchNorm uses running stats and
    Dropout is identity.
    """
    z1 = model.linears[0](x)
    u1 = model.bns[0](z1)
    a1 = F.relu(u1)
    z2 = model.linears[1](a1)
    u2 = model.bns[1](z2)
    a2 = F.relu(u2)
    z3 = model.linears[2](a2)
    u3 = model.bns[2](z3)
    a3 = F.relu(u3)
    logits = model.linears[3](a3)
    return z1, u1, a1, logits


def select_active_first_layer_neuron(
    model: GestureMLP,
    X_std: torch.Tensor,
    min_active_fraction: float = 0.5,
    seed: int = DEFAULT_SEED,
) -> tuple[int, int]:
    """Pick a ``(neuron_idx, input_idx)`` whose chain-rule contribution is
    non-trivial on a probe batch.

    Selection rule (§12 of the plan):
      1. Forward the probe batch through ``linears[0] -> bns[0] -> relu``.
      2. Pick the neuron with the highest mean post-ReLU activation;
         tiebreak by lowest index.
      3. Require ``(a1 > 0).float().mean() >= min_active_fraction`` for the
         chosen neuron; otherwise fall back to the next most-active neuron.
      4. Pick the input dimension with the largest mean ``|x_j|`` on the
         probe batch; tiebreak by lowest index.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, _, a1, _ = _eval_forward_capture(model, X_std)
        act_fraction = (a1 > 0).float().mean(dim=0).cpu().numpy()  # (256,)
        # Stable sort by (-fraction, index) -> tiebreak by lowest index.
        order = np.lexsort((np.arange(act_fraction.size), -act_fraction))
        chosen_neuron = None
        for i in order:
            if act_fraction[i] >= min_active_fraction:
                chosen_neuron = int(i)
                break
        if chosen_neuron is None:
            # Fall back to the most-active neuron even if below the threshold.
            chosen_neuron = int(order[0])
        # Input dim with largest mean |x_j| on the probe batch.
        mean_abs_x = X_std.detach().abs().mean(dim=0).cpu().numpy()
        input_order = np.lexsort((np.arange(mean_abs_x.size), -mean_abs_x))
        chosen_input = int(input_order[0])
    finally:
        if was_training:
            model.train()
    return chosen_neuron, chosen_input


def pick_active_sample_indices(
    model: GestureMLP,
    X_std: torch.Tensor,
    neuron_idx: int,
    n_samples: int = 12,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Return ``n_samples`` row indices into ``X_std`` where ``neuron_idx`` of
    layer 1 fires (post-ReLU activation > 0).

    Iterates ``X_std`` in chunks to avoid materialising 49k×256 in memory all
    at once. If fewer than ``n_samples`` rows activate the neuron, falls back
    to padding with deterministic samples (gate=0 rows still pass verification
    but show only the trivial 0=0 product).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            chunk = 2048
            active_mask = torch.zeros(X_std.shape[0], dtype=torch.bool)
            for start in range(0, X_std.shape[0], chunk):
                end = min(start + chunk, X_std.shape[0])
                z1 = model.linears[0](X_std[start:end])
                u1 = model.bns[0](z1)
                active_mask[start:end] = u1[:, neuron_idx] > 0
    finally:
        if was_training:
            model.train()

    rng = np.random.default_rng(seed)
    active_idx = np.where(active_mask.numpy())[0]
    if active_idx.size >= n_samples:
        chosen = rng.choice(active_idx, size=n_samples, replace=False)
        return sorted(int(i) for i in chosen)

    # Fallback: take all active, pad with evenly-spaced indices.
    needed = n_samples - active_idx.size
    pad = np.linspace(0, X_std.shape[0] - 1, needed, dtype=int)
    combined = sorted(set(int(i) for i in active_idx) | set(int(i) for i in pad))
    return combined[:n_samples]


def verify_chain_rule_single_sample(
    model: GestureMLP,
    x_std: torch.Tensor,
    y_true: int,
    neuron_idx: int,
    input_idx: int,
    criterion: torch.nn.Module | None = None,
) -> dict:
    """Verify ``dL/dW^{(1)}_{i,j}`` for one sample.

    ``x_std`` must be standardized and shaped ``(1, 279)`` or ``(279,)``.
    Returns a dict matching the chain_rule_verification.csv schema.
    """
    if x_std.dim() == 1:
        x_std = x_std.unsqueeze(0)
    if x_std.shape != (1, INPUT_DIM):
        raise ValueError(
            f"x_std must be shape (1, {INPUT_DIM}); got {tuple(x_std.shape)}"
        )

    device = next(model.parameters()).device
    x = x_std.to(device).detach().clone().requires_grad_(False)
    y = torch.tensor([int(y_true)], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    z1, u1, a1, logits = _eval_forward_capture(model, x)
    loss = criterion(logits, y)

    # Upstream gradient ∂L/∂a1[0, i] via autograd (retain graph so we can
    # also call loss.backward() afterwards for the autograd reference).
    (grad_a1,) = torch.autograd.grad(loss, a1, retain_graph=True)
    upstream = float(grad_a1[0, neuron_idx].item())

    # Local factors (read once; no autograd).
    gamma_i = float(model.bns[0].weight[neuron_idx].item())
    var_i = float(model.bns[0].running_var[neuron_idx].item())
    eps_bn = float(model.bns[0].eps)
    bn_scale = gamma_i / math.sqrt(var_i + eps_bn)
    u1_i = float(u1[0, neuron_idx].item())
    gate = 1.0 if u1_i > 0.0 else 0.0
    x_j = float(x[0, input_idx].item())

    manual = upstream * gate * bn_scale * x_j

    # Autograd reference: full backward.
    model.zero_grad(set_to_none=True)
    loss.backward()
    autograd_val = float(
        model.linears[0].weight.grad[neuron_idx, input_idx].item()
    )

    abs_err = abs(manual - autograd_val)
    denom = max(abs(autograd_val), 1e-12)
    rel_err = abs_err / denom

    predicted_label = int(logits.argmax(dim=-1).item())

    if was_training:
        model.train()

    return {
        "neuron_idx": int(neuron_idx),
        "input_idx": int(input_idx),
        "label_id": int(y_true),
        "predicted_label_id": predicted_label,
        "manual_grad": float(manual),
        "autograd_grad": float(autograd_val),
        "abs_error": float(abs_err),
        "rel_error": float(rel_err),
        "gate_active": bool(gate > 0.0),
        "bn_scale": float(bn_scale),
        "upstream_grad_a1_i": float(upstream),
        "x_j": float(x_j),
        "passed": bool(abs_err < CHAIN_RULE_TOL_DEFAULT),
    }


def verify_chain_rule_batch(
    model: GestureMLP,
    X_std: torch.Tensor,
    y: torch.Tensor,
    sample_indices: Sequence[int],
    neuron_idx: int,
    input_idx: int,
    label_map: dict[int, str] | None = None,
    tol: float = CHAIN_RULE_TOL_DEFAULT,
    required_passes: int = CHAIN_RULE_REQUIRED_PASSES,
) -> pd.DataFrame:
    """Run chain-rule verification across ``sample_indices``; return DataFrame.

    Raises ``AssertionError`` if fewer than ``required_passes`` rows pass.
    """
    rows: list[dict] = []
    for idx in sample_indices:
        x_i = X_std[idx : idx + 1]
        y_i = int(y[idx].item()) if torch.is_tensor(y) else int(y[idx])
        row = verify_chain_rule_single_sample(
            model=model,
            x_std=x_i,
            y_true=y_i,
            neuron_idx=neuron_idx,
            input_idx=input_idx,
        )
        row["sample_index"] = int(idx)
        row["label_name"] = (
            label_map.get(y_i, str(y_i)) if label_map is not None else str(y_i)
        )
        row["passed"] = bool(row["abs_error"] < tol)
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "sample_index", "label_id", "label_name", "predicted_label_id",
        "neuron_idx", "input_idx",
        "manual_grad", "autograd_grad", "abs_error", "rel_error",
        "gate_active", "bn_scale", "upstream_grad_a1_i", "x_j", "passed",
    ])

    passed = int(df["passed"].sum())
    if passed < required_passes:
        worst = df.nlargest(3, "abs_error")[
            ["sample_index", "abs_error", "rel_error", "gate_active"]
        ].to_string(index=False)
        raise AssertionError(
            f"chain-rule verification only {passed}/{len(df)} passes "
            f"(required {required_passes}, tol {tol}). Worst rows:\n{worst}"
        )
    return df
