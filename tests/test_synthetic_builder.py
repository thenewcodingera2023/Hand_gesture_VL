"""Unit tests for src/synthetic_builder.py — output shape, presence flags, composition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.feature_assembler import (
    INTER_DIST_IDX,
    LEFT_FEAT_SLICE,
    LEFT_PRESENT_IDX,
    RIGHT_FEAT_SLICE,
    RIGHT_PRESENT_IDX,
)
from src.preprocessor import PER_HAND_DIM, TWO_HAND_DIM
from src.synthetic_builder import (
    COUNT_COMPOSITION,
    build_count_class,
    sample_inter_hand_distance,
)


LABEL_TO_ID = json.load(open(Path("data/labels.json"), "r", encoding="utf-8"))


def fake_pool(label_seed: int, n: int = 20) -> np.ndarray:
    rng = np.random.default_rng(label_seed)
    return rng.standard_normal((n, PER_HAND_DIM)).astype(np.float32)


def make_pools() -> dict[str, np.ndarray]:
    # After the 26-class collision fix, count_2 / count_5 are no longer pool
    # names — count_2 is just `peace` and count_5 is just `open_palm`. The
    # synthetic builder draws from the `peace` pool for count_7/12/16.
    return {
        lbl: fake_pool(i + 1, n=20)
        for i, lbl in enumerate(("open_palm", "count_1", "peace", "count_3", "count_4"))
    }


def test_build_count_class_shapes():
    pools = make_pools()
    out = build_count_class(
        "count_6", pools, n=64,
        rng=np.random.default_rng(0), prior=None,
        label_to_id=LABEL_TO_ID,
    )
    assert out["X"].shape == (64, TWO_HAND_DIM)
    assert out["X"].dtype == np.float32
    assert out["y"].shape == (64,)
    assert out["right_src_idx"].shape == (64,)
    assert out["left_src_idx"].shape == (64,)
    assert out["inter_dist"].shape == (64,)


def test_presence_flags_both_one():
    pools = make_pools()
    out = build_count_class(
        "count_6", pools, n=32,
        rng=np.random.default_rng(0), prior=None, label_to_id=LABEL_TO_ID,
    )
    assert np.all(out["X"][:, RIGHT_PRESENT_IDX] == 1.0)
    assert np.all(out["X"][:, LEFT_PRESENT_IDX] == 1.0)


def test_slot_donor_correspondence():
    pools = make_pools()
    out = build_count_class(
        "count_8", pools, n=20,
        rng=np.random.default_rng(0), prior=None, label_to_id=LABEL_TO_ID,
    )
    right_label = COUNT_COMPOSITION["count_8"][0]
    left_label = COUNT_COMPOSITION["count_8"][1]
    for i in range(20):
        np.testing.assert_allclose(
            out["X"][i, RIGHT_FEAT_SLICE], pools[right_label][out["right_src_idx"][i]]
        )
        np.testing.assert_allclose(
            out["X"][i, LEFT_FEAT_SLICE], pools[left_label][out["left_src_idx"][i]]
        )


def test_inter_distance_finite_and_nonzero():
    pools = make_pools()
    out = build_count_class(
        "count_15", pools, n=64,
        rng=np.random.default_rng(0), prior=None, label_to_id=LABEL_TO_ID,
    )
    inter = out["X"][:, INTER_DIST_IDX]
    assert np.isfinite(inter).all()
    assert np.all(inter >= 0.0)
    assert float(np.std(inter)) > 0.0


def test_composition_correctness_for_all_counts():
    pools = make_pools()
    rng = np.random.default_rng(0)
    for count_label, (right_label, left_label) in COUNT_COMPOSITION.items():
        out = build_count_class(
            count_label, pools, n=4, rng=rng, prior=None, label_to_id=LABEL_TO_ID,
        )
        assert out["right_label"][0] == right_label
        assert out["left_label"][0] == left_label
        assert out["X"].shape == (4, TWO_HAND_DIM)


def test_y_matches_labels_json():
    pools = make_pools()
    out = build_count_class(
        "count_6", pools, n=4,
        rng=np.random.default_rng(0), prior=None, label_to_id=LABEL_TO_ID,
    )
    assert int(out["y"][0]) == LABEL_TO_ID["count_6"]


def test_determinism_under_same_seed():
    pools = make_pools()
    out1 = build_count_class(
        "count_18", pools, n=50,
        rng=np.random.default_rng(123), prior=None, label_to_id=LABEL_TO_ID,
    )
    out2 = build_count_class(
        "count_18", pools, n=50,
        rng=np.random.default_rng(123), prior=None, label_to_id=LABEL_TO_ID,
    )
    np.testing.assert_array_equal(out1["X"], out2["X"])


def test_empty_pool_raises():
    pools = make_pools()
    pools["count_1"] = np.zeros((0, PER_HAND_DIM), dtype=np.float32)
    with pytest.raises(RuntimeError):
        build_count_class(
            "count_15", pools, n=4,
            rng=np.random.default_rng(0), prior=None, label_to_id=LABEL_TO_ID,
        )


def test_sample_inter_hand_distance_uses_prior_when_present():
    rng = np.random.default_rng(0)
    prior = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    out = sample_inter_hand_distance(rng, prior, n=10)
    assert out.shape == (10,)
    np.testing.assert_array_equal(out, np.full(10, 5.0, dtype=np.float32))


def test_sample_inter_hand_distance_falls_back_to_uniform():
    rng = np.random.default_rng(0)
    out = sample_inter_hand_distance(rng, None, n=1000)
    assert out.shape == (1000,)
    assert np.all(out >= 1.5) and np.all(out <= 3.5)
