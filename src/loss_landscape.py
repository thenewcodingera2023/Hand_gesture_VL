"""Filter-normalised random-direction loss landscape around theta_best.

Implements the Li et al. (2018) construction:

    L(theta) = L(theta_best + a * d1 + b * d2)

where d1, d2 are two random parameter-space directions whose
``nn.Linear`` rows are rescaled to match the L2 norms of the corresponding
trained-weight rows ("filter normalisation"). Loss is evaluated on a real
held-out subset of ``data/splits/val.npz`` standardised by the Stage 4
``StandardScaler`` embedded in ``runs/mlp_best.pt``.

The module guarantees state-restoration safety: parameters are mutated
in-place inside ``torch.no_grad()`` and restored in a ``try/finally`` block;
SHA-256 of ``state_dict`` bytes is asserted equal before/after.

Authoritative spec:
    - C:/Users/Harry T/.claude/plans/relevant-planning-task-files-c-users-har-ethereal-pie.md

Public API:
    - generate_filter_normalized_directions
    - select_eval_subset
    - evaluate_loss_landscape
    - compute_descent_path_2d
    - plot_landscape_contour
    - plot_landscape_surface
    - save_landscape_npz
"""

from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.mlp import GestureMLP
from src.train import DEFAULT_SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE = 51
DEFAULT_RADIUS = 1.0
DEFAULT_EVAL_SUBSAMPLE = 4000
_EPS = 1e-12
_LARGE_RADIUS_WARN = 1.5


# ---------------------------------------------------------------------------
# Direction generation
# ---------------------------------------------------------------------------


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """SHA-256 of the concatenated raw bytes of every state-dict tensor.

    Deterministic across runs of the same checkpoint on the same architecture
    on the same hardware. Used as an audit trail and as a state-restoration
    check (pre/post grid sweep).
    """
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        # Bring to CPU + contiguous so .numpy().tobytes() is well-defined for
        # all dtypes. detach() avoids the no-grad requirement.
        arr = t.detach().to("cpu").contiguous()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def generate_filter_normalized_directions(
    model: GestureMLP,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``(d1, d2)`` filter-normalised random parameter-space directions.

    For each ``nn.Linear.weight`` of shape ``(out, in)``: each output-neuron
    row of the random direction is rescaled so its L2 norm equals the L2
    norm of the corresponding trained-weight row. For ``nn.Linear.bias``
    vectors: rescaled by the trained-bias L2 norm. BatchNorm affine
    parameters (``bns.*.weight`` / ``bns.*.bias``) are held fixed — their
    direction tensors are exactly zero.

    Parameters
    ----------
    model
        Loaded ``GestureMLP``; its trained weights set the per-row scales.
    seed
        Seeds a private ``torch.Generator`` so direction generation is
        reproducible and independent of any global RNG state.

    Returns
    -------
    (d1, d2)
        Two dicts mapping ``model.named_parameters()`` keys to tensors of
        identical shape on the model's device.
    """
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    d1: dict[str, torch.Tensor] = {}
    d2: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        for d_dict in (d1, d2):
            if name.startswith("bns."):
                d_dict[name] = torch.zeros_like(param, device=device)
                continue

            # Generate on CPU with explicit generator for determinism, then
            # move to model device.
            d_cpu = torch.empty(param.shape, dtype=param.dtype)
            d_cpu.normal_(generator=gen)
            d = d_cpu.to(device)

            if param.dim() == 2:
                # Per-row filter normalisation: ||d[i, :]||_2 := ||w[i, :]||_2.
                with torch.no_grad():
                    row_norm_d = d.norm(dim=1, keepdim=True) + _EPS
                    row_norm_w = param.detach().norm(dim=1, keepdim=True)
                    d = d * (row_norm_w / row_norm_d)
            else:
                # 1-D (biases): scalar normalisation.
                with torch.no_grad():
                    n_d = float(d.norm().item()) + _EPS
                    n_w = float(param.detach().norm().item())
                    d = d * (n_w / n_d)

            d_dict[name] = d.detach()

    return d1, d2


def _per_linear_row_norm_ratios(
    model: GestureMLP, d: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alive_row_ratios, dead_row_counts)`` per ``linears[k].weight``.

    ``alive_row_ratios[k]`` is the mean of ``||d[i,:]|| / ||W[i,:]||`` over the
    rows of layer ``k`` whose trained-weight norm is non-zero. Filter
    normalisation guarantees this is exactly 1.0 (within float precision).

    ``dead_row_counts[k]`` is the number of rows of ``linears[k].weight``
    whose L2 norm is below ``_EPS``. These rows are dead post-training
    (Kaiming init survived to zero) and contribute nothing to the landscape;
    surfacing the count separately keeps the sanity check honest.
    """
    ratios: list[float] = []
    dead_counts: list[int] = []
    for k in range(4):
        name = f"linears.{k}.weight"
        if name not in d:
            continue
        W = dict(model.named_parameters())[name].detach()
        D = d[name].detach()
        with torch.no_grad():
            w_row = W.norm(dim=1)
            d_row = D.norm(dim=1)
            alive = w_row > _EPS
            dead_counts.append(int((~alive).sum().item()))
            if alive.any():
                ratios.append(float(
                    (d_row[alive] / w_row[alive]).mean().item()
                ))
            else:
                ratios.append(float("nan"))
    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(dead_counts, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Held-out subset selection
# ---------------------------------------------------------------------------


def select_eval_subset(
    npz_path: Path,
    scaler_mean: np.ndarray | list,
    scaler_scale: np.ndarray | list,
    n_samples: int = DEFAULT_EVAL_SUBSAMPLE,
    natural_only: bool = False,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``npz_path``, optionally drop synthetic/augmented rows, stratified
    subsample to ``n_samples`` rows, apply the Stage 4 scaler.

    Returns ``(X_std, y)`` as ``float32`` / ``int64`` CPU tensors.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"split file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as z:
        if "X" not in z.files or "y" not in z.files:
            raise KeyError(f"{npz_path} missing required arrays 'X' and 'y'")
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"]).astype(np.int64, copy=False)
        is_synth = (
            np.asarray(z["is_synthetic"], dtype=bool)
            if "is_synthetic" in z.files else np.zeros(X.shape[0], dtype=bool)
        )
        is_aug = (
            np.asarray(z["is_augmented"], dtype=bool)
            if "is_augmented" in z.files else np.zeros(X.shape[0], dtype=bool)
        )

    if natural_only:
        keep = ~(is_synth | is_aug)
        X = X[keep]
        y = y[keep]

    if X.shape[0] == 0:
        raise ValueError(f"no rows left in {npz_path} after natural_only filter")
    if n_samples > X.shape[0]:
        raise ValueError(
            f"n_samples={n_samples} > available rows {X.shape[0]} in {npz_path}"
        )

    if n_samples < X.shape[0]:
        # StratifiedShuffleSplit needs >= 2 samples per class. Per-class sizes
        # in val are all comfortably above this; fall back to plain shuffle if
        # any class has < 2 rows in the post-filter pool.
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=n_samples, random_state=int(seed)
            )
            idx, _ = next(splitter.split(X, y))
        except ValueError:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    mean_arr = np.asarray(scaler_mean, dtype=np.float32)
    scale_arr = np.asarray(scaler_scale, dtype=np.float32)
    if mean_arr.shape != (X.shape[1],) or scale_arr.shape != (X.shape[1],):
        raise ValueError(
            f"scaler shapes {mean_arr.shape}/{scale_arr.shape} != ({X.shape[1]},)"
        )
    X_std = (X - mean_arr) / scale_arr

    return torch.from_numpy(X_std.astype(np.float32)), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def evaluate_loss_landscape(
    model: GestureMLP,
    X_std: torch.Tensor,
    y: torch.Tensor,
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    grid_size: int = DEFAULT_GRID_SIZE,
    radius: float = DEFAULT_RADIUS,
    device: torch.device | None = None,
    eval_split: str = "val",
    seed: int | None = None,
) -> dict:
    """Compute ``L(theta_best + a*d1 + b*d2)`` on a symmetric grid.

    Mutates ``model.named_parameters()`` in-place inside ``torch.no_grad()``
    and restores them in ``try/finally``. Verifies SHA-256 of ``state_dict``
    bytes pre/post — raises ``RuntimeError`` if they differ.

    Notes
    -----
    Uses ``model.eval()`` so BatchNorm uses running statistics and Dropout
    is identity. The returned grid is therefore deterministic for a fixed
    ``(model, X_std, y, d1, d2, grid_size, radius)``.
    """
    if grid_size < 3:
        raise ValueError(f"grid_size must be >= 3, got {grid_size}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if X_std.shape[0] != y.shape[0]:
        raise ValueError(
            f"X_std rows {X_std.shape[0]} != y rows {y.shape[0]}"
        )
    if X_std.ndim != 2:
        raise ValueError(f"X_std must be 2-D, got shape {tuple(X_std.shape)}")

    if device is None:
        device = next(model.parameters()).device

    if radius > _LARGE_RADIUS_WARN:
        warnings.warn(
            f"radius={radius} > {_LARGE_RADIUS_WARN}: the perturbation now "
            f"exceeds the per-row scale of the trained weights, so the surface "
            f"extrapolates beyond filter-normalised units.",
            RuntimeWarning,
            stacklevel=2,
        )

    X = X_std.to(device)
    yt = y.to(device).long()

    a_axis = np.linspace(-radius, radius, grid_size, dtype=np.float64)
    b_axis = np.linspace(-radius, radius, grid_size, dtype=np.float64)
    loss_grid = np.empty((grid_size, grid_size), dtype=np.float32)

    # Snapshot trained weights for safe restoration.
    named_params = dict(model.named_parameters())
    theta_orig = {name: p.detach().clone() for name, p in named_params.items()}

    # Validate direction dicts cover every trainable parameter.
    missing_d1 = set(named_params) - set(d1)
    missing_d2 = set(named_params) - set(d2)
    if missing_d1 or missing_d2:
        raise ValueError(
            f"direction dicts missing keys: d1={missing_d1}, d2={missing_d2}"
        )

    hash_pre = _state_dict_sha256(model)
    was_training = model.training
    model.eval()

    nan_or_inf = False
    try:
        with torch.no_grad():
            anchor_logits = model(X)
            anchor_loss = float(F.cross_entropy(anchor_logits, yt).item())

            for ia, a in enumerate(a_axis):
                for ib, b in enumerate(b_axis):
                    for name, p in named_params.items():
                        # In-place: p <- theta_orig + a*d1 + b*d2.
                        p.copy_(theta_orig[name])
                        if a != 0.0:
                            p.add_(d1[name], alpha=float(a))
                        if b != 0.0:
                            p.add_(d2[name], alpha=float(b))
                    logits = model(X)
                    loss_val = float(F.cross_entropy(logits, yt).item())
                    if not np.isfinite(loss_val):
                        nan_or_inf = True
                    loss_grid[ia, ib] = loss_val
    finally:
        with torch.no_grad():
            for name, p in named_params.items():
                p.copy_(theta_orig[name])
        if was_training:
            model.train()

    hash_post = _state_dict_sha256(model)
    if hash_pre != hash_post:
        raise RuntimeError(
            "state_dict SHA-256 changed across landscape evaluation — "
            "training-weight contamination guard FAILED."
        )

    # Cross-check: the central cell should equal anchor_loss when grid_size
    # is odd (a_axis, b_axis pass through 0).
    if grid_size % 2 == 1:
        mid = grid_size // 2
        if abs(float(loss_grid[mid, mid]) - anchor_loss) > 1e-4:
            warnings.warn(
                f"central grid cell {float(loss_grid[mid, mid])} differs from "
                f"anchor loss {anchor_loss} by more than 1e-4 — "
                f"perturbation math may be off.",
                RuntimeWarning,
                stacklevel=2,
            )

    if nan_or_inf:
        warnings.warn(
            "loss_grid contains NaN/inf values — numerical blow-up at the "
            "current radius. Consider reducing radius or grid extent.",
            RuntimeWarning,
            stacklevel=2,
        )

    direction_norm_ratios, dead_rows_per_linear = _per_linear_row_norm_ratios(
        model, d1
    )

    return {
        "a_axis": a_axis,
        "b_axis": b_axis,
        "loss_grid": loss_grid,
        "anchor_loss": float(anchor_loss),
        "radius": float(radius),
        "grid_size": int(grid_size),
        "n_eval": int(X.shape[0]),
        "eval_split": str(eval_split),
        "seed": int(seed) if seed is not None else -1,
        "state_dict_sha256": hash_pre,
        "direction_norm_ratios": direction_norm_ratios,
        "dead_rows_per_linear": dead_rows_per_linear,
        "note": (
            "Random filter-normalised directions (Li et al. 2018); BatchNorm "
            "affine parameters held fixed; loss evaluated on a stratified "
            "subsample of the val split standardised by the Stage 4 scaler."
        ),
    }


# ---------------------------------------------------------------------------
# 2-D descent path on the (alpha, beta) restriction of L
# ---------------------------------------------------------------------------


def _bilinear(a_axis: np.ndarray, b_axis: np.ndarray, Z: np.ndarray,
              a: float, b: float) -> float:
    """Bilinear interpolation of ``Z(a_axis, b_axis)`` at the point ``(a, b)``.

    Clamps to the grid boundary; the descent step itself is responsible for
    keeping the path inside the domain.
    """
    da = float(a_axis[1] - a_axis[0])
    db = float(b_axis[1] - b_axis[0])
    fa = (a - float(a_axis[0])) / da
    fb = (b - float(b_axis[0])) / db
    n_a = Z.shape[0] - 1
    n_b = Z.shape[1] - 1
    ia = int(min(max(int(np.floor(fa)), 0), n_a - 1))
    ib = int(min(max(int(np.floor(fb)), 0), n_b - 1))
    ta = float(min(max(fa - ia, 0.0), 1.0))
    tb = float(min(max(fb - ib, 0.0), 1.0))
    z00 = float(Z[ia, ib])
    z10 = float(Z[ia + 1, ib])
    z01 = float(Z[ia, ib + 1])
    z11 = float(Z[ia + 1, ib + 1])
    return (
        z00 * (1 - ta) * (1 - tb)
        + z10 * ta * (1 - tb)
        + z01 * (1 - ta) * tb
        + z11 * ta * tb
    )


def _grad_2d(a_axis: np.ndarray, b_axis: np.ndarray, Z: np.ndarray,
             a: float, b: float, h_frac: float = 0.5) -> tuple[float, float]:
    """Central-difference gradient of ``L(a, b)`` via bilinear interpolation.

    ``h_frac`` is the step size as a fraction of one grid spacing; 0.5 gives a
    half-cell step on each side. Boundary points fall back to one-sided
    differences naturally because :func:`_bilinear` clamps.
    """
    da = float(a_axis[1] - a_axis[0]) * h_frac
    db = float(b_axis[1] - b_axis[0]) * h_frac
    dL_da = (
        _bilinear(a_axis, b_axis, Z, a + da, b)
        - _bilinear(a_axis, b_axis, Z, a - da, b)
    ) / (2 * da)
    dL_db = (
        _bilinear(a_axis, b_axis, Z, a, b + db)
        - _bilinear(a_axis, b_axis, Z, a, b - db)
    ) / (2 * db)
    return dL_da, dL_db


def compute_descent_path_2d(
    landscape_payload: dict,
    start: tuple[float, float] = (-0.8, 0.8),
    learning_rate: float = 0.05,
    n_steps: int = 60,
    tolerance: float = 1e-5,
) -> dict:
    """Gradient descent on the precomputed 2-D restriction
    ``L_{a,b}(a, b) = L(theta_best + a*d_1 + b*d_2)``.

    This is **not** the original Stage 4 trajectory through 107k-d parameter
    space — that history is unrecoverable from a single checkpoint. It is a
    real gradient descent on the real (but 2-D-restricted) loss surface
    already evaluated by :func:`evaluate_loss_landscape`, started from
    ``start``. Pedagogically: it shows what gradient descent into a basin
    looks like on this slice.

    Parameters
    ----------
    landscape_payload
        Output of :func:`evaluate_loss_landscape`.
    start
        ``(alpha_0, beta_0)`` initial point. Must lie within the grid.
    learning_rate
        Step size in (a, b) units per iteration.
    n_steps
        Maximum number of gradient steps.
    tolerance
        Stop early if the loss change between consecutive steps drops below
        this threshold.

    Returns
    -------
    dict with keys:
        ``path`` : ``np.ndarray`` of shape ``(T, 3)``, columns ``(a, b, loss)``.
        ``start`` : the start tuple.
        ``learning_rate`` : the lr used.
        ``n_steps_taken`` : actual number of steps recorded (T-1).
        ``converged`` : ``True`` if the loop hit ``tolerance`` before ``n_steps``.
    """
    a_axis = np.asarray(landscape_payload["a_axis"], dtype=np.float64)
    b_axis = np.asarray(landscape_payload["b_axis"], dtype=np.float64)
    Z = np.asarray(landscape_payload["loss_grid"], dtype=np.float64)

    if Z.shape != (a_axis.size, b_axis.size):
        raise ValueError(
            f"loss_grid shape {Z.shape} != (len(a_axis), len(b_axis))="
            f"({a_axis.size}, {b_axis.size})"
        )
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    a0, b0 = float(start[0]), float(start[1])
    a_lo, a_hi = float(a_axis[0]), float(a_axis[-1])
    b_lo, b_hi = float(b_axis[0]), float(b_axis[-1])
    if not (a_lo <= a0 <= a_hi and b_lo <= b0 <= b_hi):
        raise ValueError(
            f"start={start} lies outside the grid "
            f"[{a_lo}, {a_hi}] x [{b_lo}, {b_hi}]"
        )

    path: list[tuple[float, float, float]] = [
        (a0, b0, _bilinear(a_axis, b_axis, Z, a0, b0))
    ]
    converged = False
    for _ in range(n_steps):
        a_t, b_t, l_t = path[-1]
        ga, gb = _grad_2d(a_axis, b_axis, Z, a_t, b_t)
        a_next = a_t - learning_rate * ga
        b_next = b_t - learning_rate * gb
        # Clamp inside the grid so bilinear interp stays valid.
        a_next = float(min(max(a_next, a_lo), a_hi))
        b_next = float(min(max(b_next, b_lo), b_hi))
        l_next = _bilinear(a_axis, b_axis, Z, a_next, b_next)
        path.append((a_next, b_next, l_next))
        if abs(l_next - l_t) < tolerance:
            converged = True
            break

    arr = np.asarray(path, dtype=np.float64)
    return {
        "path": arr,
        "start": (a0, b0),
        "learning_rate": float(learning_rate),
        "n_steps_taken": int(arr.shape[0] - 1),
        "converged": bool(converged),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_landscape_contour(
    payload: dict,
    output_path: Path | None = None,
    title: str | None = None,
    descent_path: dict | None = None,
) -> Figure:
    """Render the landscape as a filled contour map with overlaid level curves.

    If ``descent_path`` is provided (output of :func:`compute_descent_path_2d`),
    overlays the trajectory as a red marker-line with distinct start (black)
    and end (lime) markers.
    """
    a = payload["a_axis"]
    b = payload["b_axis"]
    Z = payload["loss_grid"]
    n_eval = int(payload.get("n_eval", -1))
    split = str(payload.get("eval_split", "val"))
    anchor = float(payload["anchor_loss"])

    A, B = np.meshgrid(a, b, indexing="ij")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cf = ax.contourf(A, B, Z, levels=30, cmap="viridis")
    cs = ax.contour(A, B, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    ax.plot(0.0, 0.0, "r*", markersize=14, label=f"theta_best (loss={anchor:.3f})")
    if descent_path is not None:
        pa = descent_path["path"]
        ax.plot(
            pa[:, 0], pa[:, 1],
            color="red", marker=".", linewidth=1.5, markersize=4,
            label="descent path", zorder=4,
        )
        ax.scatter([pa[0, 0]], [pa[0, 1]], color="black",
                   s=80, zorder=5, label="start")
        ax.scatter([pa[-1, 0]], [pa[-1, 1]], color="lime",
                   s=80, edgecolor="black", linewidths=1, zorder=5, label="end")
    ax.set_xlabel(r"$\alpha$  (filter-normalised $d_1$)")
    ax.set_ylabel(r"$\beta$  (filter-normalised $d_2$)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(cf, ax=ax, label=f"cross-entropy loss ({split}, n={n_eval})")
    ax.set_title(
        title
        or "Loss landscape — filter-normalised random directions around "
        + r"$\theta_{best}$"
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_landscape_surface(
    payload: dict,
    output_path: Path | None = None,
    title: str | None = None,
    elev: float = 30.0,
    azim: float = -60.0,
    descent_path: dict | None = None,
) -> Figure:
    """Render the landscape as a 3D surface ``(a, b, loss)``.

    If ``descent_path`` is provided, the trajectory is drawn above the surface
    with start/end markers. Z values are lifted by a small offset so the line
    is visible against the surface.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    a = payload["a_axis"]
    b = payload["b_axis"]
    Z = payload["loss_grid"]
    n_eval = int(payload.get("n_eval", -1))
    split = str(payload.get("eval_split", "val"))
    anchor = float(payload["anchor_loss"])

    A, B = np.meshgrid(a, b, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.7,
                            linewidth=0, antialiased=True)
    ax.scatter([0.0], [0.0], [anchor], color="red", s=60,
               label=f"theta_best (loss={anchor:.3f})", depthshade=False)
    if descent_path is not None:
        pa = descent_path["path"]
        z_offset = 0.02 * (float(Z.max()) - float(Z.min()))
        ax.plot(
            pa[:, 0], pa[:, 1], pa[:, 2] + z_offset,
            color="red", marker=".", linewidth=2.0, markersize=4,
            label="descent path", zorder=4,
        )
        ax.scatter([pa[0, 0]], [pa[0, 1]], [pa[0, 2] + z_offset],
                   color="black", s=70, label="start", depthshade=False)
        ax.scatter([pa[-1, 0]], [pa[-1, 1]], [pa[-1, 2] + z_offset],
                   color="lime", s=70, edgecolor="black",
                   linewidths=1, label="end", depthshade=False)
    ax.set_xlabel(r"$\alpha$  (FN $d_1$)")
    ax.set_ylabel(r"$\beta$  (FN $d_2$)")
    ax.set_zlabel(f"CE loss ({split}, n={n_eval})")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="cross-entropy loss")
    ax.set_title(
        title
        or "Loss landscape (3D) — filter-normalised random directions"
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def save_landscape_npz(payload: dict, output_path: Path) -> None:
    """Persist the landscape payload as a single ``.npz`` for re-plotting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        a_axis=np.asarray(payload["a_axis"], dtype=np.float64),
        b_axis=np.asarray(payload["b_axis"], dtype=np.float64),
        loss_grid=np.asarray(payload["loss_grid"], dtype=np.float32),
        anchor_loss=np.float64(payload["anchor_loss"]),
        radius=np.float64(payload["radius"]),
        grid_size=np.int64(payload["grid_size"]),
        n_eval=np.int64(payload["n_eval"]),
        eval_split=np.array(payload["eval_split"]),
        seed=np.int64(payload["seed"]),
        state_dict_sha256=np.array(payload["state_dict_sha256"]),
        direction_norm_ratios=np.asarray(
            payload["direction_norm_ratios"], dtype=np.float64
        ),
        dead_rows_per_linear=np.asarray(
            payload.get("dead_rows_per_linear", np.zeros(4, dtype=np.int64)),
            dtype=np.int64,
        ),
        note=np.array(payload["note"]),
    )


def load_landscape_npz(npz_path: Path) -> dict:
    """Inverse of :func:`save_landscape_npz` — return a payload dict."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        payload = {
            "a_axis": z["a_axis"].copy(),
            "b_axis": z["b_axis"].copy(),
            "loss_grid": z["loss_grid"].copy(),
            "anchor_loss": float(z["anchor_loss"]),
            "radius": float(z["radius"]),
            "grid_size": int(z["grid_size"]),
            "n_eval": int(z["n_eval"]),
            "eval_split": str(z["eval_split"]),
            "seed": int(z["seed"]),
            "state_dict_sha256": str(z["state_dict_sha256"]),
            "direction_norm_ratios": z["direction_norm_ratios"].copy(),
            "dead_rows_per_linear": (
                z["dead_rows_per_linear"].copy()
                if "dead_rows_per_linear" in z.files
                else np.zeros(4, dtype=np.int64)
            ),
            "note": str(z["note"]),
        }
    return payload
