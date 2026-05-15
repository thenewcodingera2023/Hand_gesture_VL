"""Build synthetic two-hand count samples (counts 6-18) from single-hand pools.

Also owns the bridge from raw HaGRID records -> per-hand (138,) features and
single-hand assembled (279,) records, since both share pool construction.

See `tasks/gesture_recognition_plan_v2.md` §1.3, §2.3, §5.2 and the Stage 1 plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from src.preprocessor import (
    BONE_PAIRS,
    FINGER_BONE_GROUPS,
    FINGERTIPS,
    PALM_REF,
    PER_HAND_DIM,
    TWO_HAND_DIM,
    WRIST,
    preprocess_hand,
)

COUNT_COMPOSITION: dict[str, tuple[str, str]] = {
    "count_6":  ("open_palm", "count_1"),
    "count_7":  ("open_palm", "count_2"),
    "count_8":  ("open_palm", "count_3"),
    "count_9":  ("open_palm", "count_4"),
    "count_10": ("open_palm", "open_palm"),
    "count_11": ("count_1",   "open_palm"),
    "count_12": ("count_2",   "open_palm"),
    "count_13": ("count_3",   "open_palm"),
    "count_14": ("count_4",   "open_palm"),
    "count_15": ("count_1",   "count_1"),
    "count_16": ("count_2",   "count_1"),
    "count_17": ("count_3",   "count_1"),
    "count_18": ("count_4",   "count_1"),
}

SAMPLES_PER_COUNT = 500
DEFAULT_SEED = 20260514

DEFAULT_INTER_DIST_PRIOR_PATH = Path("data/processed/inter_hand_distance_prior.npy")
DEFAULT_INTER_DIST_FALLBACK = (1.5, 3.5)

# Minimum raw palm size (in raw landmark units) for a detection to be kept.
# Anything below this is a degenerate MediaPipe output.
_RAW_PALM_MIN = 1e-4

LABELS_JSON_DEFAULT = Path("data/labels.json")


def _load_label_to_id(path: Path = LABELS_JSON_DEFAULT) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-hand feature build
# ---------------------------------------------------------------------------

def _vectorized_per_hand_features(
    landmarks_raw: np.ndarray,
    hands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched equivalent of preprocessor.preprocess_hand over N records.

    Numerically identical to looping preprocess_hand row-by-row up to float32
    rounding. Returns (features (N,138) float32, keep_mask (N,) bool) where
    keep_mask=False marks degenerate detections (raw palm < _RAW_PALM_MIN).
    """
    lm = landmarks_raw.astype(np.float32, copy=True)
    n = lm.shape[0]
    assert lm.shape == (n, 21, 3)

    # Translate to wrist origin: lm -= lm[:, WRIST:WRIST+1]
    lm -= lm[:, WRIST:WRIST + 1, :]

    # Raw palm size (post-translate, pre-normalize) for the keep mask.
    raw_palm = np.linalg.norm(lm[:, PALM_REF, :], axis=1)
    keep = raw_palm >= _RAW_PALM_MIN

    # Scale by palm size with 1e-6 epsilon (matches preprocessor.normalize_hand).
    scale = (raw_palm + 1e-6).reshape(n, 1, 1)
    lm /= scale

    # Mirror left hands.
    is_left = (hands == "Left")
    if is_left.any():
        lm[is_left, :, 0] *= -1.0

    # Bone vectors: (N, 20, 3) -> (N, 60)
    parents = np.asarray([p for p, _ in BONE_PAIRS], dtype=np.int32)
    children = np.asarray([c for _, c in BONE_PAIRS], dtype=np.int32)
    bv = lm[:, children, :] - lm[:, parents, :]
    bone_lengths = np.linalg.norm(bv, axis=2)  # (N, 20)
    bv_flat = bv.reshape(n, -1)

    # Extension ratios per finger (5 groups of 4 bones).
    ext = np.empty((n, len(FINGER_BONE_GROUPS)), dtype=np.float32)
    for fi, group in enumerate(FINGER_BONE_GROUPS):
        chain = bone_lengths[:, list(group)].sum(axis=1)
        first_parent = BONE_PAIRS[group[0]][0]
        last_child = BONE_PAIRS[group[-1]][1]
        straight = np.linalg.norm(lm[:, last_child, :] - lm[:, first_parent, :], axis=1)
        ext[:, fi] = straight / (chain + 1e-6)
    np.clip(ext, 0.0, 1.0, out=ext)

    # Pairwise fingertip distances (10 = C(5,2)).
    tips = lm[:, FINGERTIPS, :]  # (N, 5, 3)
    n_tips = len(FINGERTIPS)
    pd = np.empty((n, n_tips * (n_tips - 1) // 2), dtype=np.float32)
    k = 0
    for i in range(n_tips):
        for j in range(i + 1, n_tips):
            pd[:, k] = np.linalg.norm(tips[:, i, :] - tips[:, j, :], axis=1)
            k += 1

    flat_lm = lm.reshape(n, -1)  # (N, 63)
    feats = np.concatenate([flat_lm, bv_flat, ext, pd], axis=1).astype(np.float32, copy=False)
    assert feats.shape == (n, PER_HAND_DIM)
    return feats, keep


def build_single_hand_features(
    raw_records_path: Path,
) -> dict:
    """Run vectorized per-hand normalisation over every raw record.

    Equivalent to row-by-row preprocessor.preprocess_hand calls but ~100x
    faster because the work is amortised across numpy broadcasts.
    """
    raw = np.load(raw_records_path, allow_pickle=True)
    landmarks_raw = raw["landmarks_raw"]
    hands = raw["hand"]
    n_raw = landmarks_raw.shape[0]
    print(f"  vectorizing per-hand features over {n_raw} raw records...")

    feats, keep = _vectorized_per_hand_features(landmarks_raw, hands)
    finite_mask = np.isfinite(feats).all(axis=1)
    keep &= finite_mask

    raw_idx = np.nonzero(keep)[0].astype(np.int32, copy=False)
    X = feats[keep].astype(np.float32, copy=False)

    project_label = raw["project_label"][raw_idx].astype(object)
    user_id = raw["user_id"][raw_idx].astype(object)
    hagrid_split = raw["hagrid_split"][raw_idx].astype(object)
    source = raw["source"][raw_idx].astype(object)

    label_to_id = _load_label_to_id()
    y = np.asarray([label_to_id[lbl] for lbl in project_label], dtype=np.int32)

    print(f"  kept {X.shape[0]} of {n_raw} records "
          f"({n_raw - X.shape[0]} dropped as degenerate / non-finite)")

    return {
        "X": X,
        "y": y,
        "project_label": project_label,
        "user_id": user_id,
        "hagrid_split": hagrid_split,
        "source": source,
        "raw_index": raw_idx,
    }


def save_single_hand_features(features: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **features)
    print(f"==> Wrote {features['X'].shape[0]} per-hand features to {out_path}")


def assemble_single_hand_dataset(features: dict) -> dict:
    """For each per-hand sample, produce a (279,) vector with the left slot
    zeroed and `right_present=1.0`. Used for control + count_1..count_5."""
    n = features["X"].shape[0]
    X = np.zeros((n, TWO_HAND_DIM), dtype=np.float32)
    X[:, RIGHT_FEAT_SLICE] = features["X"]
    X[:, RIGHT_PRESENT_IDX] = 1.0
    # Left slot stays zero, left_present=0.0, inter_dist=0.0 by construction.

    return {
        "X": X,
        "y": features["y"].copy(),
        "project_label": features["project_label"].copy(),
        "user_id": features["user_id"].copy(),
        "hagrid_split": features["hagrid_split"].copy(),
        "source": features["source"].copy(),
        "right_label": features["project_label"].copy(),
        "left_label": np.full(n, "", dtype=object),
    }


def save_single_hand_assembled(dataset: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"==> Wrote {dataset['X'].shape[0]} single-hand assembled (279,) rows to {out_path}")


# ---------------------------------------------------------------------------
# Synthetic two-hand build
# ---------------------------------------------------------------------------

def _pools_from_features(features: dict) -> dict[str, np.ndarray]:
    pools: dict[str, list[np.ndarray]] = {}
    project_label = features["project_label"]
    X = features["X"]
    for i, lbl in enumerate(project_label):
        pools.setdefault(str(lbl), []).append(X[i])
    return {k: np.stack(v, axis=0).astype(np.float32, copy=False) for k, v in pools.items()}


def load_inter_hand_distance_prior(
    path: Path = DEFAULT_INTER_DIST_PRIOR_PATH,
) -> Optional[np.ndarray]:
    if path.is_file():
        prior = np.load(path).astype(np.float32, copy=False).reshape(-1)
        if prior.size == 0:
            return None
        return prior
    return None


def sample_inter_hand_distance(
    rng: np.random.Generator,
    prior: Optional[np.ndarray],
    n: int,
) -> np.ndarray:
    if prior is not None and prior.size > 0:
        return rng.choice(prior, size=n, replace=True).astype(np.float32, copy=False)
    return rng.uniform(
        DEFAULT_INTER_DIST_FALLBACK[0],
        DEFAULT_INTER_DIST_FALLBACK[1],
        size=n,
    ).astype(np.float32, copy=False)


def build_count_class(
    count_label: str,
    pools: dict[str, np.ndarray],
    n: int,
    rng: np.random.Generator,
    prior: Optional[np.ndarray],
    label_to_id: Optional[dict[str, int]] = None,
) -> dict:
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if count_label not in COUNT_COMPOSITION:
        raise ValueError(f"unknown count_label: {count_label!r}")

    right_label, left_label = COUNT_COMPOSITION[count_label]
    if right_label not in pools or pools[right_label].shape[0] == 0:
        raise RuntimeError(f"empty pool for right-hand label {right_label!r}")
    if left_label not in pools or pools[left_label].shape[0] == 0:
        raise RuntimeError(f"empty pool for left-hand label {left_label!r}")

    right_pool = pools[right_label]
    left_pool = pools[left_label]
    right_idx = rng.integers(0, right_pool.shape[0], size=n).astype(np.int32, copy=False)
    left_idx = rng.integers(0, left_pool.shape[0], size=n).astype(np.int32, copy=False)
    inter = sample_inter_hand_distance(rng, prior, n)

    X = np.zeros((n, TWO_HAND_DIM), dtype=np.float32)
    X[:, RIGHT_FEAT_SLICE] = right_pool[right_idx]
    X[:, LEFT_FEAT_SLICE] = left_pool[left_idx]
    X[:, RIGHT_PRESENT_IDX] = 1.0
    X[:, LEFT_PRESENT_IDX] = 1.0
    X[:, INTER_DIST_IDX] = inter

    if label_to_id is None:
        label_to_id = _load_label_to_id()
    y = np.full(n, label_to_id[count_label], dtype=np.int32)

    return {
        "X": X,
        "y": y,
        "right_label": np.full(n, right_label, dtype=object),
        "left_label": np.full(n, left_label, dtype=object),
        "right_src_idx": right_idx,
        "left_src_idx": left_idx,
        "inter_dist": inter,
        "source": np.full(n, "synthetic", dtype=object),
        "project_label": np.full(n, count_label, dtype=object),
    }


def build_all_synthetic(
    pools: dict[str, np.ndarray],
    samples_per_count: int = SAMPLES_PER_COUNT,
    seed: int = DEFAULT_SEED,
    prior_path: Path = DEFAULT_INTER_DIST_PRIOR_PATH,
) -> dict:
    rng = np.random.default_rng(seed)
    prior = load_inter_hand_distance_prior(prior_path)
    if prior is None:
        print(
            f"WARNING: inter-hand distance prior file not found at {prior_path}; "
            f"falling back to uniform U{DEFAULT_INTER_DIST_FALLBACK}. "
            f"Replace with measured prior before Stage 4."
        )
    label_to_id = _load_label_to_id()

    chunks: list[dict] = []
    for count_label in COUNT_COMPOSITION:
        chunk = build_count_class(
            count_label, pools, samples_per_count, rng, prior, label_to_id=label_to_id
        )
        chunks.append(chunk)
        print(f"  built {count_label}: {chunk['X'].shape[0]} rows "
              f"({chunk['right_label'][0]} | {chunk['left_label'][0]})")

    keys = ["X", "y", "right_label", "left_label", "right_src_idx",
            "left_src_idx", "inter_dist", "source", "project_label"]
    out = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
    out["seed"] = np.int64(seed)
    return out


def save_synthetic(dataset: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"==> Wrote {dataset['X'].shape[0]} synthetic two-hand rows to {out_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_outputs(
    raw_path: Path,
    features_path: Path,
    assembled_path: Path,
    synthetic_path: Path,
    min_records: int = 200_000,
) -> None:
    raw = np.load(raw_path, allow_pickle=True)
    n_raw = raw["landmarks_raw"].shape[0]
    print(f"raw records:               {n_raw}")
    assert n_raw >= min_records, f"raw record count {n_raw} below floor {min_records}"
    assert raw["landmarks_raw"].dtype == np.float32
    assert raw["landmarks_raw"].shape[1:] == (21, 3)

    feats = np.load(features_path, allow_pickle=True)
    X1 = feats["X"]
    print(f"single-hand features:      {X1.shape}")
    assert X1.dtype == np.float32 and X1.shape[1] == PER_HAND_DIM
    assert np.isfinite(X1).all(), "NaN/Inf in single-hand features"

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(X1.shape[0], size=min(100, X1.shape[0]), replace=False)
    for i in sample_idx:
        np.testing.assert_allclose(X1[i, 0:3], 0.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(X1[i, 27:30]), 1.0, atol=1e-4)
        ext = X1[i, 123:128]
        assert np.all(ext >= 0.0) and np.all(ext <= 1.0)

    asm = np.load(assembled_path, allow_pickle=True)
    Xa = asm["X"]
    print(f"single-hand assembled:     {Xa.shape}")
    assert Xa.shape[1] == TWO_HAND_DIM
    assert np.all(Xa[:, 138:276] == 0.0), "left slot should be zero"
    assert np.all(Xa[:, RIGHT_PRESENT_IDX] == 1.0)
    assert np.all(Xa[:, LEFT_PRESENT_IDX] == 0.0)
    assert np.all(Xa[:, INTER_DIST_IDX] == 0.0)

    syn = np.load(synthetic_path, allow_pickle=True)
    Xs = syn["X"]
    print(f"synthetic two-hand:        {Xs.shape}")
    assert Xs.shape[1] == TWO_HAND_DIM
    assert np.all(Xs[:, RIGHT_PRESENT_IDX] == 1.0)
    assert np.all(Xs[:, LEFT_PRESENT_IDX] == 1.0)
    assert np.isfinite(Xs[:, INTER_DIST_IDX]).all()
    assert np.all(Xs[:, INTER_DIST_IDX] >= 0.0)
    # Variance across rows (not all identical).
    assert float(np.std(Xs[:, INTER_DIST_IDX])) > 0.0

    syn_counts = Counter(str(lbl) for lbl in syn["project_label"])
    print("synthetic per-class counts:")
    for lbl in COUNT_COMPOSITION:
        print(f"    {lbl:9s} {syn_counts[lbl]:>6d}")
        assert syn_counts[lbl] > 0, f"missing synthetic class {lbl}"

    raw_counts = Counter(str(lbl) for lbl in raw["project_label"])
    print("raw per-project-label counts:")
    for lbl, c in sorted(raw_counts.items()):
        print(f"    {lbl:14s} {c:>8d}")

    print("VALIDATE: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: per-hand features, single-hand assembly, synthetic count construction."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_singles = sub.add_parser(
        "build-singles",
        help="Build single-hand (138,) features and single-hand assembled (279,) rows.",
    )
    p_singles.add_argument("--raw", type=Path, default=Path("data/processed/hagrid_raw_records.npz"))
    p_singles.add_argument("--features-out", type=Path, default=Path("data/processed/single_hand_features.npz"))
    p_singles.add_argument("--assembled-out", type=Path, default=Path("data/processed/single_hand_assembled.npz"))

    p_counts = sub.add_parser(
        "build-counts",
        help="Build synthetic two-hand count samples for count_6..count_18.",
    )
    p_counts.add_argument("--features", type=Path, default=Path("data/processed/single_hand_features.npz"))
    p_counts.add_argument("--out", type=Path, default=Path("data/processed/synthetic_two_hand.npz"))
    p_counts.add_argument("--samples-per-count", type=int, default=SAMPLES_PER_COUNT)
    p_counts.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_counts.add_argument("--prior", type=Path, default=DEFAULT_INTER_DIST_PRIOR_PATH)

    p_validate = sub.add_parser("validate", help="Validate Stage 1 acceptance gate.")
    p_validate.add_argument("--raw", type=Path, default=Path("data/processed/hagrid_raw_records.npz"))
    p_validate.add_argument("--features", type=Path, default=Path("data/processed/single_hand_features.npz"))
    p_validate.add_argument("--assembled", type=Path, default=Path("data/processed/single_hand_assembled.npz"))
    p_validate.add_argument("--synthetic", type=Path, default=Path("data/processed/synthetic_two_hand.npz"))
    p_validate.add_argument("--min-records", type=int, default=200_000)

    args = parser.parse_args(argv)

    if args.cmd == "build-singles":
        feats = build_single_hand_features(args.raw)
        save_single_hand_features(feats, args.features_out)
        assembled = assemble_single_hand_dataset(feats)
        save_single_hand_assembled(assembled, args.assembled_out)
    elif args.cmd == "build-counts":
        feats = np.load(args.features, allow_pickle=True)
        feats_dict = {k: feats[k] for k in feats.files}
        pools = _pools_from_features(feats_dict)
        synthetic = build_all_synthetic(
            pools,
            samples_per_count=args.samples_per_count,
            seed=args.seed,
            prior_path=args.prior,
        )
        save_synthetic(synthetic, args.out)
    elif args.cmd == "validate":
        validate_outputs(
            args.raw, args.features, args.assembled, args.synthetic,
            min_records=args.min_records,
        )
    else:
        parser.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
