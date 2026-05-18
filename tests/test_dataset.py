"""Unit tests for src/dataset.py — no user_id leakage, every class in every split,
oversampling only on train, deterministic seed, augmentation invariants.

Heavy tests are marked @pytest.mark.slow. Cheap tests run against the existing
data/splits/ if present; otherwise the splits are built once into a tmp dir
(session-scoped fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import dataset as ds
from src.feature_assembler import (
    INTER_DIST_IDX,
    LEFT_PRESENT_IDX,
    RIGHT_PRESENT_IDX,
)
from src.preprocessor import TWO_HAND_DIM


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SPLITS_DIR = REPO_ROOT / "data" / "splits"
LABELS_PATH = REPO_ROOT / "data" / "labels.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def labels() -> dict[str, int]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def splits_dir(tmp_path_factory) -> Path:
    """Return a directory containing train/val/test.npz.

    Prefers `data/splits/` if it exists with all three files; otherwise builds
    them once into a tmp directory for the session. When neither the splits
    nor the Stage 1 input artifact exist (e.g. clean CI checkout), the tests
    requesting this fixture are skipped rather than erroring.
    """
    canonical = PRODUCTION_SPLITS_DIR
    if all((canonical / f"{n}.npz").is_file() for n in ds.SPLIT_NAMES):
        return canonical
    singles_path = REPO_ROOT / "data" / "processed" / "single_hand_assembled.npz"
    if not singles_path.is_file():
        pytest.skip(
            f"requires data/splits/ or {singles_path.relative_to(REPO_ROOT).as_posix()}; "
            "run Stage 1 first (python -m src.synthetic_builder build-singles)"
        )
    out = tmp_path_factory.mktemp("splits")
    ds.build_splits(out_dir=out, seed=ds.DEFAULT_SEED)
    return out


@pytest.fixture(scope="session")
def split_arrays(splits_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Map split name -> dict of arrays loaded from disk."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in ds.SPLIT_NAMES:
        z = np.load(splits_dir / f"{name}.npz", allow_pickle=True)
        out[name] = {k: z[k] for k in z.files}
    return out


# ---------------------------------------------------------------------------
# Cheap structural / leakage tests
# ---------------------------------------------------------------------------

def test_x_and_y_lengths_match(split_arrays):
    for name, arrs in split_arrays.items():
        assert len(arrs["X"]) == len(arrs["y"]), f"{name}: X/y length mismatch"


def test_feature_dimension_279(split_arrays):
    for name, arrs in split_arrays.items():
        assert arrs["X"].shape[1] == TWO_HAND_DIM, f"{name}: feature dim != {TWO_HAND_DIM}"


def test_dtypes(split_arrays):
    for name, arrs in split_arrays.items():
        assert arrs["X"].dtype == np.float32, f"{name}: X dtype != float32"
        assert arrs["y"].dtype == np.int32, f"{name}: y dtype != int32"


def test_finite_features(split_arrays):
    for name, arrs in split_arrays.items():
        assert np.isfinite(arrs["X"]).all(), f"{name}: NaN/Inf in X"


def test_label_ids_valid(split_arrays, labels):
    valid = set(labels.values())
    from src.models.mlp import NUM_CLASSES
    assert valid == set(range(NUM_CLASSES))
    for name, arrs in split_arrays.items():
        present = set(int(v) for v in np.unique(arrs["y"]))
        assert present.issubset(valid), f"{name}: invalid label IDs {present - valid}"


def test_every_class_in_every_split(split_arrays, labels):
    valid = set(labels.values())
    for name, arrs in split_arrays.items():
        present = set(int(v) for v in np.unique(arrs["y"]))
        missing = valid - present
        assert not missing, f"{name}: missing class IDs {sorted(missing)}"


def test_metadata_array_lengths_match_x(split_arrays):
    for name, arrs in split_arrays.items():
        n = arrs["X"].shape[0]
        for k, v in arrs.items():
            if k == "seed":
                continue
            assert v.shape[0] == n, f"{name}: metadata '{k}' length {v.shape[0]} != {n}"


def test_no_user_id_overlap_between_splits(split_arrays):
    user_sets = {}
    for name, arrs in split_arrays.items():
        uid = arrs["user_id"].astype(str)
        is_syn = arrs["is_synthetic"].astype(bool)
        user_sets[name] = set(uid[~is_syn].tolist())
    assert user_sets["train"].isdisjoint(user_sets["val"]), "train/val user overlap"
    assert user_sets["train"].isdisjoint(user_sets["test"]), "train/test user overlap"
    assert user_sets["val"].isdisjoint(user_sets["test"]), "val/test user overlap"


def test_synthetic_placement_no_leakage(split_arrays):
    real_users: dict[str, set[str]] = {}
    for name, arrs in split_arrays.items():
        uid = arrs["user_id"].astype(str)
        is_syn = arrs["is_synthetic"].astype(bool)
        real_users[name] = set(uid[~is_syn].tolist())

    for name, arrs in split_arrays.items():
        is_syn = arrs["is_synthetic"].astype(bool)
        if not is_syn.any():
            continue
        right = arrs["synth_right_user"][is_syn].astype(str)
        left = arrs["synth_left_user"][is_syn].astype(str)
        right_set = set(right.tolist())
        left_set = set(left.tolist())
        assert right_set.issubset(real_users[name]), (
            f"{name}: synth_right_user leaks outside split"
        )
        assert left_set.issubset(real_users[name]), (
            f"{name}: synth_left_user leaks outside split"
        )
        for other in ds.SPLIT_NAMES:
            if other == name:
                continue
            assert right_set.isdisjoint(real_users[other]), (
                f"{name}: synth_right_user overlaps real users of {other}"
            )
            assert left_set.isdisjoint(real_users[other]), (
                f"{name}: synth_left_user overlaps real users of {other}"
            )


def test_oversampling_only_train(split_arrays):
    assert bool(split_arrays["train"]["is_augmented"].any()), "train missing augmented rows"
    assert not bool(split_arrays["val"]["is_augmented"].any()), "val has augmented rows"
    assert not bool(split_arrays["test"]["is_augmented"].any()), "test has augmented rows"


def test_train_count_classes_reach_target(split_arrays):
    y_train = split_arrays["train"]["y"]
    for cid in ds.OVERSAMPLE_CLASS_IDS:
        n_cid = int(np.sum(y_train == cid))
        assert n_cid >= ds.OVERSAMPLE_TARGET, (
            f"train class {cid} has {n_cid} rows (< {ds.OVERSAMPLE_TARGET})"
        )


def test_split_proportions_approximate_80_10_10(split_arrays):
    """Real-row proportions should be within +/-5% of (0.80, 0.10, 0.10)."""
    real_counts = {}
    for name, arrs in split_arrays.items():
        is_syn = arrs["is_synthetic"].astype(bool)
        is_aug = arrs["is_augmented"].astype(bool)
        real_counts[name] = int(np.sum(~is_syn & ~is_aug))
    total = sum(real_counts.values())
    train_frac = real_counts["train"] / total
    val_frac = real_counts["val"] / total
    test_frac = real_counts["test"] / total
    assert 0.75 < train_frac < 0.85, f"train frac {train_frac:.3f} not ~0.80"
    assert 0.05 < val_frac < 0.15, f"val frac {val_frac:.3f} not ~0.10"
    assert 0.05 < test_frac < 0.15, f"test frac {test_frac:.3f} not ~0.10"


def test_synthetic_rows_have_left_present_one(split_arrays):
    for name, arrs in split_arrays.items():
        is_syn = arrs["is_synthetic"].astype(bool)
        if not is_syn.any():
            continue
        assert np.all(arrs["X"][is_syn, LEFT_PRESENT_IDX] == 1.0), (
            f"{name}: synthetic rows must have left_present=1.0"
        )
        assert np.all(arrs["X"][is_syn, RIGHT_PRESENT_IDX] == 1.0), (
            f"{name}: synthetic rows must have right_present=1.0"
        )


def test_real_rows_have_left_present_zero(split_arrays):
    """Stage 1 produces only single-hand HaGRID rows for real samples."""
    for name, arrs in split_arrays.items():
        is_syn = arrs["is_synthetic"].astype(bool)
        is_aug = arrs["is_augmented"].astype(bool)
        real_mask = ~is_syn & ~is_aug
        if not real_mask.any():
            continue
        assert np.all(arrs["X"][real_mask, LEFT_PRESENT_IDX] == 0.0), (
            f"{name}: real rows must have left_present=0.0"
        )
        assert np.all(arrs["X"][real_mask, RIGHT_PRESENT_IDX] == 1.0), (
            f"{name}: real rows must have right_present=1.0"
        )


def test_augmented_rows_preserve_presence_and_inter_dist(split_arrays):
    arrs = split_arrays["train"]
    aug_mask = arrs["is_augmented"].astype(bool)
    if not aug_mask.any():
        pytest.skip("no augmented rows in train")
    Xa = arrs["X"][aug_mask]
    # Augmented rows are duplicated from synthetic count_6..18 rows, which all
    # have right_present=left_present=1.0. Presence flags must not be touched.
    assert np.all(Xa[:, RIGHT_PRESENT_IDX] == 1.0)
    assert np.all(Xa[:, LEFT_PRESENT_IDX] == 1.0)
    # inter_dist must be unchanged by augment_features; just check it's finite
    # and non-negative (it is sampled from a non-negative empirical/uniform prior).
    assert np.isfinite(Xa[:, INTER_DIST_IDX]).all()
    assert np.all(Xa[:, INTER_DIST_IDX] >= 0.0)


def test_augmented_rows_finite_and_shape(split_arrays):
    arrs = split_arrays["train"]
    aug_mask = arrs["is_augmented"].astype(bool)
    if not aug_mask.any():
        pytest.skip("no augmented rows in train")
    Xa = arrs["X"][aug_mask]
    assert Xa.shape[1] == TWO_HAND_DIM
    assert np.isfinite(Xa).all()


# ---------------------------------------------------------------------------
# Unit tests for individual functions
# ---------------------------------------------------------------------------

def test_split_users_deterministic():
    uids = np.array([f"u{i:03d}" for i in range(100)], dtype=object)
    a = ds.split_users(uids, seed=ds.DEFAULT_SEED)
    b = ds.split_users(uids, seed=ds.DEFAULT_SEED)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_split_users_different_seed_changes_assignment():
    uids = np.array([f"u{i:03d}" for i in range(100)], dtype=object)
    a = ds.split_users(uids, seed=ds.DEFAULT_SEED)
    b = ds.split_users(uids, seed=ds.DEFAULT_SEED + 1)
    # train sets very unlikely to be identical for 100 users
    assert a[0] != b[0]


def test_split_users_fractions():
    uids = np.array([f"u{i:04d}" for i in range(10000)], dtype=object)
    train, val, test = ds.split_users(uids, seed=ds.DEFAULT_SEED)
    assert len(train) + len(val) + len(test) == 10000
    assert 0.78 < len(train) / 10000 < 0.82
    assert 0.08 < len(val) / 10000 < 0.12
    assert 0.08 < len(test) / 10000 < 0.12
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def test_split_users_rejects_empty_user():
    uids = np.array(["alice", "", "bob"], dtype=object)
    with pytest.raises(AssertionError):
        ds.split_users(uids)


def test_augment_features_preserves_shape_and_finiteness():
    rng = np.random.default_rng(0)
    X = np.random.RandomState(0).randn(50, TWO_HAND_DIM).astype(np.float32)
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    X[:, ds.RIGHT_PAIRWISE] = np.abs(X[:, ds.RIGHT_PAIRWISE])
    X[:, ds.LEFT_PAIRWISE] = np.abs(X[:, ds.LEFT_PAIRWISE])
    Xa = ds.augment_features(X, sigma=0.01, rng=rng)
    assert Xa.shape == X.shape
    assert Xa.dtype == np.float32
    assert np.isfinite(Xa).all()


def test_augment_features_does_not_touch_presence_or_inter_dist():
    rng = np.random.default_rng(0)
    X = np.random.RandomState(0).randn(20, TWO_HAND_DIM).astype(np.float32)
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    X[:, INTER_DIST_IDX] = 2.5
    X[:, ds.RIGHT_PAIRWISE] = np.abs(X[:, ds.RIGHT_PAIRWISE])
    X[:, ds.LEFT_PAIRWISE] = np.abs(X[:, ds.LEFT_PAIRWISE])
    Xa = ds.augment_features(X, sigma=0.05, rng=rng)
    assert np.all(Xa[:, RIGHT_PRESENT_IDX] == 1.0)
    assert np.all(Xa[:, LEFT_PRESENT_IDX] == 1.0)
    assert np.allclose(Xa[:, INTER_DIST_IDX], 2.5)


def test_augment_features_skips_absent_hand():
    rng = np.random.default_rng(0)
    X = np.zeros((10, TWO_HAND_DIM), dtype=np.float32)
    X[:, ds.RIGHT_LANDMARK_BONE] = 1.0
    X[:, RIGHT_PRESENT_IDX] = 1.0
    # left absent, all zeros
    Xa = ds.augment_features(X, sigma=0.01, rng=rng)
    # Right side perturbed:
    assert not np.allclose(Xa[:, ds.RIGHT_LANDMARK_BONE], 1.0)
    # Left side untouched (zero):
    assert np.all(Xa[:, ds.LEFT_LANDMARK_BONE] == 0.0)
    assert np.all(Xa[:, ds.LEFT_PAIRWISE] == 0.0)


def test_augment_features_pairwise_nonnegative():
    rng = np.random.default_rng(0)
    X = np.zeros((100, TWO_HAND_DIM), dtype=np.float32)
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    X[:, ds.RIGHT_PAIRWISE] = 0.001  # near zero so noise pushes some negative
    X[:, ds.LEFT_PAIRWISE] = 0.001
    Xa = ds.augment_features(X, sigma=0.1, rng=rng)
    assert np.all(Xa[:, ds.RIGHT_PAIRWISE] >= 0.0)
    assert np.all(Xa[:, ds.LEFT_PAIRWISE] >= 0.0)


def test_augment_features_zero_sigma_is_identity():
    rng = np.random.default_rng(0)
    X = np.random.RandomState(0).randn(5, TWO_HAND_DIM).astype(np.float32)
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    X[:, ds.RIGHT_PAIRWISE] = np.abs(X[:, ds.RIGHT_PAIRWISE])
    X[:, ds.LEFT_PAIRWISE] = np.abs(X[:, ds.LEFT_PAIRWISE])
    Xa = ds.augment_features(X, sigma=0.0, rng=rng)
    assert np.array_equal(Xa, X)


def test_oversample_train_counts_reaches_target():
    rng = np.random.default_rng(0)
    # 3 classes: cid=15 has 10 rows, cid=16 has 0 (should raise), cid=17 has 6000 (no-op).
    n = 10 + 6000
    X = np.zeros((n, TWO_HAND_DIM), dtype=np.float32)
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    y = np.concatenate([np.full(10, 15, dtype=np.int32),
                        np.full(6000, 17, dtype=np.int32)])
    meta = {"is_synthetic": np.ones(n, dtype=bool),
            "user_id": np.full(n, ds.SYNTHETIC_USER_SENTINEL, dtype=object)}
    Xo, yo, mo = ds.oversample_train_counts(
        X, y, meta,
        target_per_class=5000,
        class_ids=(15, 17),
        sigma=0.01,
        rng=rng,
    )
    assert int(np.sum(yo == 15)) == 5000
    assert int(np.sum(yo == 17)) == 6000  # no-op (already > target)
    assert bool(mo["is_augmented"].any())
    aug_for_15 = mo["is_augmented"] & (yo == 15)
    assert int(aug_for_15.sum()) == 5000 - 10


def test_oversample_train_counts_raises_on_missing_class():
    rng = np.random.default_rng(0)
    X = np.zeros((5, TWO_HAND_DIM), dtype=np.float32)
    y = np.full(5, 15, dtype=np.int32)
    meta = {"is_synthetic": np.ones(5, dtype=bool)}
    with pytest.raises(RuntimeError):
        ds.oversample_train_counts(
            X, y, meta,
            target_per_class=100,
            class_ids=(15, 16),
            sigma=0.01,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# Heavy / slow tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_deterministic_seed_identical_splits(tmp_path):
    """Two builds with same seed produce identical X/y/metadata."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    ds.build_splits(out_dir=a_dir, seed=ds.DEFAULT_SEED)
    ds.build_splits(out_dir=b_dir, seed=ds.DEFAULT_SEED)
    for name in ds.SPLIT_NAMES:
        za = np.load(a_dir / f"{name}.npz", allow_pickle=True)
        zb = np.load(b_dir / f"{name}.npz", allow_pickle=True)
        assert np.array_equal(za["X"], zb["X"]), f"{name}: X differs"
        assert np.array_equal(za["y"], zb["y"]), f"{name}: y differs"
        for k in ("user_id", "source", "project_label", "is_synthetic",
                  "is_augmented", "synth_right_user", "synth_left_user"):
            la = za[k].astype(str) if za[k].dtype == object else za[k]
            lb = zb[k].astype(str) if zb[k].dtype == object else zb[k]
            assert np.array_equal(la, lb), f"{name}: metadata '{k}' differs"


@pytest.mark.slow
def test_different_seed_changes_split_assignment(tmp_path):
    """Different seed -> different user partition (overwhelmingly likely)."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    ds.build_splits(out_dir=a_dir, seed=ds.DEFAULT_SEED)
    ds.build_splits(out_dir=b_dir, seed=ds.DEFAULT_SEED + 1)
    za = np.load(a_dir / "train.npz", allow_pickle=True)
    zb = np.load(b_dir / "train.npz", allow_pickle=True)
    train_users_a = set(za["user_id"][~za["is_synthetic"].astype(bool)].astype(str).tolist())
    train_users_b = set(zb["user_id"][~zb["is_synthetic"].astype(bool)].astype(str).tolist())
    assert train_users_a != train_users_b
