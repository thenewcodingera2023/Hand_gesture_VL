"""Unit tests for src/preprocessor.py — shape, wrist origin, palm scale, extension ratios."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessor import (
    BONE_PAIRS,
    FINGERTIPS,
    PER_HAND_DIM,
    bone_vectors,
    extension_ratios,
    normalize_hand,
    pad_z,
    pairwise_fingertip_distances,
    preprocess_hand,
)


def make_hand(scale: float = 1.0, offset=(0.5, 0.5, 0.0), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = (rng.uniform(-1.0, 1.0, size=(21, 3)).astype(np.float32)) * scale
    base[0] = 0.0
    base[9] = np.array([0.0, scale, 0.0], dtype=np.float32)
    return base + np.asarray(offset, dtype=np.float32)


def test_wrist_at_origin_after_normalize():
    lm = make_hand()
    n = normalize_hand(lm, "Right")
    np.testing.assert_allclose(n[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_palm_scale_unit_norm():
    lm = make_hand()
    n = normalize_hand(lm, "Right")
    np.testing.assert_allclose(np.linalg.norm(n[9]), 1.0, atol=1e-5)


def test_pad_z_2d_to_3d_and_idempotent():
    z2 = np.zeros((21, 2), dtype=np.float32)
    p = pad_z(z2)
    assert p.shape == (21, 3)
    np.testing.assert_array_equal(p[:, 2], np.zeros(21, dtype=np.float32))

    z3 = np.ones((21, 3), dtype=np.float32)
    p3 = pad_z(z3)
    assert p3.shape == (21, 3)
    np.testing.assert_array_equal(p3, z3)


def test_left_hand_mirroring_flips_x_only():
    lm = make_hand(seed=1)
    n_right = normalize_hand(lm, "Right")
    n_left = normalize_hand(lm, "Left")
    np.testing.assert_allclose(n_left[:, 0], -n_right[:, 0], atol=1e-6)
    np.testing.assert_allclose(n_left[:, 1:], n_right[:, 1:], atol=1e-6)


def test_bone_vectors_shape_and_first_bone_semantics():
    n = normalize_hand(make_hand(seed=2), "Right")
    bv = bone_vectors(n)
    assert bv.shape == (60,)
    parent, child = BONE_PAIRS[0]
    np.testing.assert_allclose(bv[0:3], n[child] - n[parent], atol=1e-6)


def test_extension_ratios_in_unit_interval():
    n = normalize_hand(make_hand(seed=3), "Right")
    er = extension_ratios(n)
    assert er.shape == (5,)
    assert np.all(er >= 0.0)
    assert np.all(er <= 1.0)


def test_extension_ratio_one_for_collinear_finger():
    # Build a hand where the index finger (landmarks 0, 5, 6, 7, 8) is exactly collinear.
    lm = make_hand(seed=4)
    lm[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lm[5] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    lm[6] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    lm[7] = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    lm[8] = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    # Keep palm reference (landmark 9) non-degenerate so normalization works.
    lm[9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n = normalize_hand(lm, "Right")
    er = extension_ratios(n)
    assert abs(er[1] - 1.0) < 1e-5


def test_pairwise_fingertip_distance_shape_and_first_entry():
    n = normalize_hand(make_hand(seed=5), "Right")
    pd = pairwise_fingertip_distances(n)
    assert pd.shape == (10,)
    expected = np.linalg.norm(n[FINGERTIPS[0]] - n[FINGERTIPS[1]])
    np.testing.assert_allclose(pd[0], expected, atol=1e-6)


def test_preprocess_hand_total_shape_and_dtype():
    f = preprocess_hand(make_hand(seed=6), "Right")
    assert f.shape == (PER_HAND_DIM,)
    assert f.dtype == np.float32


def test_preprocess_hand_wrist_block_zero():
    f = preprocess_hand(make_hand(seed=7), "Right")
    np.testing.assert_allclose(f[0:3], 0.0, atol=1e-6)


def test_preprocess_hand_palm_reference_unit_norm():
    f = preprocess_hand(make_hand(seed=8), "Right")
    np.testing.assert_allclose(np.linalg.norm(f[27:30]), 1.0, atol=1e-5)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        preprocess_hand(np.zeros((20, 3), dtype=np.float32), "Right")
    with pytest.raises(ValueError):
        preprocess_hand(make_hand(seed=9), "Up")
