"""Unit tests for src/feature_assembler.py — output shape, absent-hand zeros, flags."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_assembler import (
    INTER_DIST_IDX,
    LEFT_FEAT_SLICE,
    LEFT_PRESENT_IDX,
    RIGHT_FEAT_SLICE,
    RIGHT_PRESENT_IDX,
    assemble_features,
)
from src.preprocessor import PER_HAND_DIM, TWO_HAND_DIM


def _ones() -> np.ndarray:
    return np.ones(PER_HAND_DIM, dtype=np.float32)


def test_no_hands_returns_all_zeros():
    out = assemble_features(None, None)
    assert out.shape == (TWO_HAND_DIM,)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_right_only():
    out = assemble_features(_ones(), None, inter_hand_distance=2.0)
    assert out.shape == (TWO_HAND_DIM,)
    assert out[RIGHT_FEAT_SLICE].sum() == float(PER_HAND_DIM)
    assert out[LEFT_FEAT_SLICE].sum() == 0.0
    assert out[RIGHT_PRESENT_IDX] == 1.0
    assert out[LEFT_PRESENT_IDX] == 0.0
    # Inter-hand distance is forced to 0.0 because left hand is absent.
    assert out[INTER_DIST_IDX] == 0.0


def test_left_only():
    out = assemble_features(None, _ones(), inter_hand_distance=2.0)
    assert out[RIGHT_FEAT_SLICE].sum() == 0.0
    assert out[LEFT_FEAT_SLICE].sum() == float(PER_HAND_DIM)
    assert out[RIGHT_PRESENT_IDX] == 0.0
    assert out[LEFT_PRESENT_IDX] == 1.0
    assert out[INTER_DIST_IDX] == 0.0


def test_two_hands_carries_inter_distance():
    out = assemble_features(_ones(), _ones() * 2.0, inter_hand_distance=2.5)
    assert out[RIGHT_PRESENT_IDX] == 1.0
    assert out[LEFT_PRESENT_IDX] == 1.0
    assert out[INTER_DIST_IDX] == np.float32(2.5)


def test_negative_inter_distance_clamped():
    out = assemble_features(_ones(), _ones(), inter_hand_distance=-1.0)
    assert out[INTER_DIST_IDX] == 0.0


def test_non_finite_inter_distance_clamped():
    out = assemble_features(_ones(), _ones(), inter_hand_distance=float("nan"))
    assert out[INTER_DIST_IDX] == 0.0


def test_wrong_shape_raises():
    with pytest.raises(AssertionError):
        assemble_features(np.zeros(137, dtype=np.float32), None)


def test_output_dtype_is_float32():
    out = assemble_features(_ones(), _ones(), inter_hand_distance=1.0)
    assert out.dtype == np.float32
