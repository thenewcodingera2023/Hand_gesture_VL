"""User-aware train/val/test splitting and augmentation.

See `tasks/gesture_recognition_plan_v2.md` §2.1 and the Stage 2 plan at
`C:/Users/Harry T/.claude/plans/i-am-building-the-glimmering-grove.md`.

Inputs (from Stage 1, under `data/processed/`):
    single_hand_assembled.npz   (525,643, 279) real single-hand rows + user_id
    single_hand_features.npz    (525,643, 138) per-hand features, same row order
    synthetic_two_hand.npz      (6,500, 279)   counts 6-18, with pool indices

Output (under `data/splits/`):
    train.npz / val.npz / test.npz   X, y, seed + metadata arrays
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.feature_assembler import (
    INTER_DIST_IDX,
    LEFT_FEAT_SLICE,
    LEFT_PRESENT_IDX,
    RIGHT_FEAT_SLICE,
    RIGHT_PRESENT_IDX,
)
from src.preprocessor import PER_HAND_DIM, TWO_HAND_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 20260514
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
OVERSAMPLE_TARGET = 5000
OVERSAMPLE_CLASS_IDS = tuple(range(15, 28))  # count_6..count_18 (13 classes)
AUG_SIGMA_DEFAULT = 0.01

# Per-hand feature layout (see src/preprocessor.py):
#   [0:63]   normalised landmarks (21 x 3, wrist at origin)
#   [63:123] bone vectors (20 x 3)
#   [123:128] extension ratios in [0, 1]
#   [128:138] pairwise fingertip distances (>= 0)
RIGHT_LANDMARK_BONE = slice(0, 123)
RIGHT_EXTENSION = slice(123, 128)
RIGHT_PAIRWISE = slice(128, 138)
LEFT_LANDMARK_BONE = slice(138, 261)
LEFT_EXTENSION = slice(261, 266)
LEFT_PAIRWISE = slice(266, 276)

SPLITS_DIR_DEFAULT = Path("data/splits")
SINGLES_PATH_DEFAULT = Path("data/processed/single_hand_assembled.npz")
FEATURES_PATH_DEFAULT = Path("data/processed/single_hand_features.npz")
SYNTH_PATH_DEFAULT = Path("data/processed/synthetic_two_hand.npz")
LABELS_JSON_DEFAULT = Path("data/labels.json")

SYNTHETIC_USER_SENTINEL = "__synthetic__"

SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _ensure_file(path: Path, stage_hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"required Stage-1 artifact not found: {path}. "
            f"Run Stage 1 first ({stage_hint})."
        )


def load_singles(path: Path = SINGLES_PATH_DEFAULT) -> dict:
    """Load `single_hand_assembled.npz` and validate shape/finiteness."""
    _ensure_file(path, "python -m src.synthetic_builder build-singles")
    z = np.load(path, allow_pickle=True)
    out = {
        "X": z["X"].astype(np.float32, copy=False),
        "y": z["y"].astype(np.int32, copy=False),
        "project_label": z["project_label"].astype(object),
        "user_id": z["user_id"].astype(object),
        "hagrid_split": z["hagrid_split"].astype(object),
        "source": z["source"].astype(object),
        "right_label": z["right_label"].astype(object),
        "left_label": z["left_label"].astype(object),
    }
    assert out["X"].ndim == 2 and out["X"].shape[1] == TWO_HAND_DIM, (
        f"singles X shape {out['X'].shape} != (_, {TWO_HAND_DIM})"
    )
    assert out["X"].shape[0] == out["y"].shape[0]
    assert np.isfinite(out["X"]).all(), "NaN/Inf in singles X"
    return out


def load_per_hand_features(path: Path = FEATURES_PATH_DEFAULT) -> dict:
    """Load `single_hand_features.npz`. Row order matches single_hand_assembled."""
    _ensure_file(path, "python -m src.synthetic_builder build-singles")
    z = np.load(path, allow_pickle=True)
    return {
        "X": z["X"].astype(np.float32, copy=False),
        "y": z["y"].astype(np.int32, copy=False),
        "project_label": z["project_label"].astype(object),
        "user_id": z["user_id"].astype(object),
        "hagrid_split": z["hagrid_split"].astype(object),
        "source": z["source"].astype(object),
    }


def load_synthetic(path: Path = SYNTH_PATH_DEFAULT) -> dict:
    """Load `synthetic_two_hand.npz` and validate shape/finiteness."""
    _ensure_file(path, "python -m src.synthetic_builder build-counts")
    z = np.load(path, allow_pickle=True)
    out = {
        "X": z["X"].astype(np.float32, copy=False),
        "y": z["y"].astype(np.int32, copy=False),
        "right_label": z["right_label"].astype(object),
        "left_label": z["left_label"].astype(object),
        "right_src_idx": z["right_src_idx"].astype(np.int64, copy=False),
        "left_src_idx": z["left_src_idx"].astype(np.int64, copy=False),
        "inter_dist": z["inter_dist"].astype(np.float32, copy=False),
        "source": z["source"].astype(object),
        "project_label": z["project_label"].astype(object),
    }
    assert out["X"].ndim == 2 and out["X"].shape[1] == TWO_HAND_DIM
    assert out["X"].shape[0] == out["y"].shape[0]
    assert np.isfinite(out["X"]).all(), "NaN/Inf in synthetic X"
    return out


def load_labels(path: Path = LABELS_JSON_DEFAULT) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Synthetic -> source user traceback
# ---------------------------------------------------------------------------

def build_pool_to_original_index(
    project_label: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return {label: indices_in_singles_where_project_label==label}.

    Mirrors how `_pools_from_features` in src/synthetic_builder.py constructs
    pools (iterating singles in order and grouping by project_label). The
    pool index `synth["right_src_idx"][k]` therefore maps to
    `pool_to_orig[right_label][k]` in singles.
    """
    pool_to_orig: dict[str, np.ndarray] = {}
    labels_str = project_label.astype(str)
    for lbl in np.unique(labels_str):
        pool_to_orig[str(lbl)] = np.flatnonzero(labels_str == lbl).astype(np.int64, copy=False)
    return pool_to_orig


def trace_synthetic_user_pairs(
    synth: dict,
    singles_user_id: np.ndarray,
    singles_project_label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover (right_user_id, left_user_id) per synthetic row via pool lookup."""
    pool_to_orig = build_pool_to_original_index(singles_project_label)
    n = synth["X"].shape[0]
    right_user = np.empty(n, dtype=object)
    left_user = np.empty(n, dtype=object)
    user_id_arr = singles_user_id.astype(object)

    right_lbl_arr = synth["right_label"].astype(str)
    left_lbl_arr = synth["left_label"].astype(str)
    right_pool_idx = synth["right_src_idx"]
    left_pool_idx = synth["left_src_idx"]

    # Process per (right_label, left_label) to minimise per-row Python overhead.
    for lbl in np.unique(right_lbl_arr):
        sel = right_lbl_arr == lbl
        orig = pool_to_orig[lbl][right_pool_idx[sel]]
        right_user[sel] = user_id_arr[orig]
    for lbl in np.unique(left_lbl_arr):
        sel = left_lbl_arr == lbl
        orig = pool_to_orig[lbl][left_pool_idx[sel]]
        left_user[sel] = user_id_arr[orig]
    return right_user, left_user


# ---------------------------------------------------------------------------
# User-aware split
# ---------------------------------------------------------------------------

def split_users(
    user_ids: np.ndarray,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> tuple[set[str], set[str], set[str]]:
    """Deterministic 80/10/10 partition of unique user_ids by seed."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0; got {fractions}")
    uids = sorted({str(u) for u in user_ids})
    for u in uids:
        if not u:
            raise AssertionError("encountered empty user_id; refusing to split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uids))
    n = len(uids)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train_set = {uids[i] for i in perm[:n_train]}
    val_set = {uids[i] for i in perm[n_train : n_train + n_val]}
    test_set = {uids[i] for i in perm[n_train + n_val :]}
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set) + len(val_set) + len(test_set) == n
    return train_set, val_set, test_set


def assign_real_samples(
    user_id_arr: np.ndarray,
    train_users: set[str],
    val_users: set[str],
    test_users: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask, test_mask) over real single-hand rows."""
    uid = user_id_arr.astype(str)
    train_mask = np.fromiter((u in train_users for u in uid), dtype=bool, count=len(uid))
    val_mask = np.fromiter((u in val_users for u in uid), dtype=bool, count=len(uid))
    test_mask = np.fromiter((u in test_users for u in uid), dtype=bool, count=len(uid))
    coverage = train_mask.astype(np.int32) + val_mask.astype(np.int32) + test_mask.astype(np.int32)
    assert np.all(coverage == 1), (
        f"real sample row coverage broken: min={coverage.min()} max={coverage.max()}"
    )
    return train_mask, val_mask, test_mask


def assign_synthetic_samples(
    right_user: np.ndarray,
    left_user: np.ndarray,
    train_users: set[str],
    val_users: set[str],
    test_users: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place each synthetic row in the split where BOTH source users live."""
    n = len(right_user)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    dropped = np.zeros(n, dtype=bool)
    r = right_user.astype(str)
    l = left_user.astype(str)
    for i in range(n):
        ru, lu = r[i], l[i]
        if ru in train_users and lu in train_users:
            train_mask[i] = True
        elif ru in val_users and lu in val_users:
            val_mask[i] = True
        elif ru in test_users and lu in test_users:
            test_mask[i] = True
        else:
            dropped[i] = True
    assert (train_mask.astype(np.int32) + val_mask.astype(np.int32)
            + test_mask.astype(np.int32) + dropped.astype(np.int32)).max() == 1
    return train_mask, val_mask, test_mask, dropped


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_features(
    X: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Light additive Gaussian noise on landmark/bone/distance slices.

    Extension ratios, presence flags, and inter-hand distance are left untouched.
    Pairwise distances are clipped to >= 0 after noise. Output is a new array.
    """
    if X.ndim != 2 or X.shape[1] != TWO_HAND_DIM:
        raise ValueError(f"augment_features expects (M, {TWO_HAND_DIM}); got {X.shape}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    out = X.astype(np.float32, copy=True)
    if sigma == 0.0:
        return out

    m = X.shape[0]
    right_present = X[:, RIGHT_PRESENT_IDX] > 0.5
    left_present = X[:, LEFT_PRESENT_IDX] > 0.5

    if right_present.any():
        idx = np.flatnonzero(right_present)
        out[idx, RIGHT_LANDMARK_BONE] += rng.normal(
            0.0, sigma, size=(idx.size, RIGHT_LANDMARK_BONE.stop - RIGHT_LANDMARK_BONE.start)
        ).astype(np.float32)
        rp_start, rp_stop = RIGHT_PAIRWISE.start, RIGHT_PAIRWISE.stop
        pw = out[idx, rp_start:rp_stop] + rng.normal(
            0.0, sigma, size=(idx.size, rp_stop - rp_start)
        ).astype(np.float32)
        out[idx, rp_start:rp_stop] = np.maximum(pw, 0.0)

    if left_present.any():
        idx = np.flatnonzero(left_present)
        out[idx, LEFT_LANDMARK_BONE] += rng.normal(
            0.0, sigma, size=(idx.size, LEFT_LANDMARK_BONE.stop - LEFT_LANDMARK_BONE.start)
        ).astype(np.float32)
        lp_start, lp_stop = LEFT_PAIRWISE.start, LEFT_PAIRWISE.stop
        pw = out[idx, lp_start:lp_stop] + rng.normal(
            0.0, sigma, size=(idx.size, lp_stop - lp_start)
        ).astype(np.float32)
        out[idx, lp_start:lp_stop] = np.maximum(pw, 0.0)

    if not np.isfinite(out).all():
        raise RuntimeError("augment_features produced non-finite values")
    return out


def oversample_train_counts(
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, np.ndarray],
    target_per_class: int,
    class_ids: tuple[int, ...],
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Duplicate (with noise) rows of each class in `class_ids` up to target.

    Adds `is_augmented` bool metadata if not already present.
    Returns the concatenated arrays.
    """
    if "is_augmented" not in metadata:
        metadata["is_augmented"] = np.zeros(X.shape[0], dtype=bool)

    X_parts = [X]
    y_parts = [y]
    meta_parts: dict[str, list[np.ndarray]] = {k: [v] for k, v in metadata.items()}

    for class_id in class_ids:
        src_idx = np.flatnonzero(y == class_id)
        if src_idx.size == 0:
            raise RuntimeError(
                f"class {class_id} has zero training rows; "
                f"cannot oversample. Likely a synthetic split-coverage bug."
            )
        n_needed = max(0, target_per_class - src_idx.size)
        if n_needed == 0:
            continue
        dup_pick = rng.integers(0, src_idx.size, size=n_needed).astype(np.int64, copy=False)
        chosen = src_idx[dup_pick]
        X_dup = augment_features(X[chosen], sigma=sigma, rng=rng)
        y_dup = np.full(n_needed, class_id, dtype=y.dtype)
        X_parts.append(X_dup)
        y_parts.append(y_dup)
        for k, v in metadata.items():
            if k == "is_augmented":
                meta_parts[k].append(np.ones(n_needed, dtype=bool))
            else:
                meta_parts[k].append(v[chosen])

    X_out = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y_out = np.concatenate(y_parts, axis=0).astype(np.int32, copy=False)
    metadata_out = {k: np.concatenate(parts, axis=0) for k, parts in meta_parts.items()}

    assert X_out.shape[1] == TWO_HAND_DIM
    assert np.isfinite(X_out).all()
    assert X_out.shape[0] == y_out.shape[0]
    for k, v in metadata_out.items():
        assert v.shape[0] == X_out.shape[0], f"metadata '{k}' length mismatch"
    return X_out, y_out, metadata_out


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_split(
    out_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, np.ndarray],
    seed: int,
) -> None:
    assert X.shape[1] == TWO_HAND_DIM
    assert X.dtype == np.float32
    assert y.dtype == np.int32
    assert X.shape[0] == y.shape[0]
    assert np.isfinite(X).all(), f"NaN/Inf in {out_path}"
    for k, v in metadata.items():
        assert v.shape[0] == X.shape[0], f"metadata '{k}' length mismatch in {out_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        seed=np.int64(seed),
        **metadata,
    )
    print(f"==> Wrote {X.shape[0]:>7d} rows to {out_path}")


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def _build_split_dataset(
    name: str,
    real_mask: np.ndarray,
    syn_mask: np.ndarray,
    singles: dict,
    synth: dict,
    syn_right_user: np.ndarray,
    syn_left_user: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Concatenate real + synthetic rows for a split with aligned metadata."""
    n_real = int(real_mask.sum())
    n_syn = int(syn_mask.sum())
    n_total = n_real + n_syn

    X = np.empty((n_total, TWO_HAND_DIM), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int32)
    X[:n_real] = singles["X"][real_mask]
    X[n_real:] = synth["X"][syn_mask]
    y[:n_real] = singles["y"][real_mask]
    y[n_real:] = synth["y"][syn_mask]

    metadata = {
        "user_id": np.empty(n_total, dtype=object),
        "source": np.empty(n_total, dtype=object),
        "project_label": np.empty(n_total, dtype=object),
        "right_label": np.empty(n_total, dtype=object),
        "left_label": np.empty(n_total, dtype=object),
        "hagrid_split": np.empty(n_total, dtype=object),
        "is_synthetic": np.zeros(n_total, dtype=bool),
        "is_augmented": np.zeros(n_total, dtype=bool),
        "synth_right_user": np.empty(n_total, dtype=object),
        "synth_left_user": np.empty(n_total, dtype=object),
    }

    metadata["user_id"][:n_real] = singles["user_id"][real_mask]
    metadata["source"][:n_real] = singles["source"][real_mask]
    metadata["project_label"][:n_real] = singles["project_label"][real_mask]
    metadata["right_label"][:n_real] = singles["right_label"][real_mask]
    metadata["left_label"][:n_real] = singles["left_label"][real_mask]
    metadata["hagrid_split"][:n_real] = singles["hagrid_split"][real_mask]
    metadata["is_synthetic"][:n_real] = False
    metadata["synth_right_user"][:n_real] = ""
    metadata["synth_left_user"][:n_real] = ""

    metadata["user_id"][n_real:] = SYNTHETIC_USER_SENTINEL
    metadata["source"][n_real:] = synth["source"][syn_mask]
    metadata["project_label"][n_real:] = synth["project_label"][syn_mask]
    metadata["right_label"][n_real:] = synth["right_label"][syn_mask]
    metadata["left_label"][n_real:] = synth["left_label"][syn_mask]
    metadata["hagrid_split"][n_real:] = ""
    metadata["is_synthetic"][n_real:] = True
    metadata["synth_right_user"][n_real:] = syn_right_user[syn_mask]
    metadata["synth_left_user"][n_real:] = syn_left_user[syn_mask]

    return X, y, metadata


def build_splits(
    singles_path: Path = SINGLES_PATH_DEFAULT,
    synth_path: Path = SYNTH_PATH_DEFAULT,
    out_dir: Path = SPLITS_DIR_DEFAULT,
    seed: int = DEFAULT_SEED,
    oversample_target: int = OVERSAMPLE_TARGET,
    oversample_class_ids: tuple[int, ...] = OVERSAMPLE_CLASS_IDS,
    aug_sigma: float = AUG_SIGMA_DEFAULT,
) -> dict:
    """End-to-end Stage 2 builder. Writes train/val/test.npz to `out_dir`."""
    print(f"[stage2] seed={seed}, target_per_count={oversample_target}, sigma={aug_sigma}")
    singles = load_singles(singles_path)
    synth = load_synthetic(synth_path)

    print(f"[stage2] tracing synthetic source user pairs...")
    syn_right_user, syn_left_user = trace_synthetic_user_pairs(
        synth, singles["user_id"], singles["project_label"]
    )

    print(f"[stage2] splitting users 80/10/10...")
    train_u, val_u, test_u = split_users(singles["user_id"], seed=seed)
    print(f"  users: train={len(train_u)}, val={len(val_u)}, test={len(test_u)}")

    real_train, real_val, real_test = assign_real_samples(
        singles["user_id"], train_u, val_u, test_u
    )
    print(f"  real rows: train={int(real_train.sum())}, "
          f"val={int(real_val.sum())}, test={int(real_test.sum())}")

    syn_train, syn_val, syn_test, syn_drop = assign_synthetic_samples(
        syn_right_user, syn_left_user, train_u, val_u, test_u
    )
    print(f"  synthetic rows: train={int(syn_train.sum())}, "
          f"val={int(syn_val.sum())}, test={int(syn_test.sum())}, "
          f"dropped={int(syn_drop.sum())}")

    # Leakage assert on real users.
    real_uid = singles["user_id"].astype(str)
    s_train = set(real_uid[real_train].tolist())
    s_val = set(real_uid[real_val].tolist())
    s_test = set(real_uid[real_test].tolist())
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test)

    # Build each split and write.
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)  # distinct stream for augmentation

    summary: dict = {
        "seed": seed,
        "users": {"train": len(train_u), "val": len(val_u), "test": len(test_u)},
        "real_rows": {
            "train": int(real_train.sum()),
            "val": int(real_val.sum()),
            "test": int(real_test.sum()),
        },
        "synthetic_rows_assigned": {
            "train": int(syn_train.sum()),
            "val": int(syn_val.sum()),
            "test": int(syn_test.sum()),
        },
        "dropped_synthetic_count": int(syn_drop.sum()),
        "final_rows": {},
        "class_counts": {},
    }

    masks = {"train": (real_train, syn_train),
             "val": (real_val, syn_val),
             "test": (real_test, syn_test)}

    for name in SPLIT_NAMES:
        real_mask, syn_mask = masks[name]
        X, y, meta = _build_split_dataset(
            name, real_mask, syn_mask, singles, synth, syn_right_user, syn_left_user
        )

        if name == "train":
            print(f"[stage2] oversampling count_6..18 to {oversample_target} each (sigma={aug_sigma})...")
            X, y, meta = oversample_train_counts(
                X, y, meta,
                target_per_class=oversample_target,
                class_ids=oversample_class_ids,
                sigma=aug_sigma,
                rng=rng,
            )

        # Per-class counts for summary.
        unique, counts = np.unique(y, return_counts=True)
        summary["class_counts"][name] = {int(u): int(c) for u, c in zip(unique, counts)}
        summary["final_rows"][name] = int(X.shape[0])

        out_path = out_dir / f"{name}.npz"
        serialize_split(out_path, X, y, meta, seed=seed)

    # Final cross-split assertions on the saved files.
    _post_build_assertions(out_dir, summary)
    return summary


def _post_build_assertions(out_dir: Path, summary: dict) -> None:
    """Reload saved splits and re-check the gate invariants."""
    label_to_id = load_labels()
    valid_ids = set(label_to_id.values())

    user_sets: dict[str, set[str]] = {}
    syn_users_by_split: dict[str, set[str]] = {}
    classes_by_split: dict[str, set[int]] = {}

    for name in SPLIT_NAMES:
        z = np.load(out_dir / f"{name}.npz", allow_pickle=True)
        X, y = z["X"], z["y"]
        assert X.shape[1] == TWO_HAND_DIM
        assert X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()
        assert set(int(v) for v in np.unique(y)).issubset(valid_ids)

        uid = z["user_id"].astype(str)
        is_syn = z["is_synthetic"].astype(bool)
        real_uid = set(uid[~is_syn].tolist())
        user_sets[name] = real_uid

        syn_set = set()
        if is_syn.any():
            syn_set.update(z["synth_right_user"][is_syn].astype(str).tolist())
            syn_set.update(z["synth_left_user"][is_syn].astype(str).tolist())
        syn_users_by_split[name] = syn_set
        classes_by_split[name] = set(int(v) for v in np.unique(y))

    # Pairwise real-user disjointness.
    assert user_sets["train"].isdisjoint(user_sets["val"])
    assert user_sets["train"].isdisjoint(user_sets["test"])
    assert user_sets["val"].isdisjoint(user_sets["test"])

    # Synthetic source users contained in their split's real user set.
    for name in SPLIT_NAMES:
        if syn_users_by_split[name]:
            assert syn_users_by_split[name].issubset(user_sets[name]), (
                f"synthetic source users in '{name}' leak into another split"
            )

    # Every class present in each split (where feasible). Warn if not.
    for name in SPLIT_NAMES:
        missing = sorted(set(valid_ids) - classes_by_split[name])
        if missing:
            print(f"WARNING: split '{name}' missing class IDs: {missing}")

    # Train oversampling target met for count classes.
    z_train = np.load(out_dir / "train.npz", allow_pickle=True)
    y_train = z_train["y"]
    for cid in OVERSAMPLE_CLASS_IDS:
        n_cid = int(np.sum(y_train == cid))
        if n_cid < OVERSAMPLE_TARGET:
            print(f"WARNING: train class {cid} has {n_cid} rows (< {OVERSAMPLE_TARGET})")

    # val/test must not contain augmented rows.
    for name in ("val", "test"):
        z = np.load(out_dir / f"{name}.npz", allow_pickle=True)
        assert not bool(z["is_augmented"].any()), f"{name} contains augmented rows"


# ---------------------------------------------------------------------------
# Verify / CLI
# ---------------------------------------------------------------------------

def verify_splits(splits_dir: Path = SPLITS_DIR_DEFAULT) -> None:
    summary: dict = {}
    _post_build_assertions(splits_dir, summary)
    print("VERIFY: OK")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: user-aware splits + oversampling for hand-gesture dataset."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build train/val/test.npz from Stage 1 outputs.")
    p_build.add_argument("--singles", type=Path, default=SINGLES_PATH_DEFAULT)
    p_build.add_argument("--synthetic", type=Path, default=SYNTH_PATH_DEFAULT)
    p_build.add_argument("--out-dir", type=Path, default=SPLITS_DIR_DEFAULT)
    p_build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_build.add_argument("--oversample-target", type=int, default=OVERSAMPLE_TARGET)
    p_build.add_argument("--aug-sigma", type=float, default=AUG_SIGMA_DEFAULT)

    p_verify = sub.add_parser("verify", help="Re-check invariants of existing splits.")
    p_verify.add_argument("--splits-dir", type=Path, default=SPLITS_DIR_DEFAULT)

    args = parser.parse_args(argv)

    if args.cmd == "build":
        summary = build_splits(
            singles_path=args.singles,
            synth_path=args.synthetic,
            out_dir=args.out_dir,
            seed=args.seed,
            oversample_target=args.oversample_target,
            aug_sigma=args.aug_sigma,
        )
        print("---- summary ----")
        print(json.dumps(summary, indent=2, default=str))
    elif args.cmd == "verify":
        verify_splits(splits_dir=args.splits_dir)
    else:
        parser.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
