"""Unit tests for src/smoother.py — pure-logic smoother behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from src.smoother import Smoother


def _probs(num_classes: int, vec: list[float]) -> np.ndarray:
    assert len(vec) == num_classes
    return np.asarray(vec, dtype=np.float32)


def test_returns_none_before_window_full():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    for _ in range(6):
        s.update(_probs(3, [1.0, 0.0, 0.0]), hand_present=True)
    assert s.get() == (None, 0.0)


def test_majority_vote_returns_expected_class():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    for _ in range(7):
        s.update(_probs(3, [0.9, 0.05, 0.05]), hand_present=True)
    cls, conf = s.get()
    assert cls == 0
    assert conf == pytest.approx(0.9, abs=1e-5)


def test_threshold_suppresses_low_confidence():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    for _ in range(7):
        s.update(_probs(3, [0.6, 0.2, 0.2]), hand_present=True)
    cls, conf = s.get()
    assert cls is None
    assert conf == pytest.approx(0.6, abs=1e-5)


def test_tiebreak_picks_higher_mean_prob():
    """3 frames argmax=0 conf=0.5, 3 frames argmax=1 conf=0.9, 1 frame argmax=2 conf=0.8.

    Counts tie at 3 between classes 0 and 1. Class 1 has higher mean prob across
    the window, so the smoother must pick class 1.
    """
    s = Smoother(window=7, threshold=0.0, no_hand_clear_frames=5, num_classes=3)
    for _ in range(3):
        s.update(_probs(3, [0.5, 0.25, 0.25]), hand_present=True)
    for _ in range(3):
        s.update(_probs(3, [0.05, 0.9, 0.05]), hand_present=True)
    s.update(_probs(3, [0.1, 0.1, 0.8]), hand_present=True)
    cls, _ = s.get()
    assert cls == 1


def test_tiebreak_secondary_picks_smaller_id():
    """Construct an exact-mean tie between classes 0 and 1 by symmetry."""
    s = Smoother(window=6, threshold=0.0, no_hand_clear_frames=5, num_classes=3)
    # 3 frames argmax=0 with prob 0.7 for class 0, 0.0 elsewhere.
    # 3 frames argmax=1 with prob 0.7 for class 1, 0.0 elsewhere.
    # Both classes appear 3x with mean prob 0.35 over the window.
    for _ in range(3):
        s.update(_probs(3, [0.7, 0.15, 0.15]), hand_present=True)
    for _ in range(3):
        s.update(_probs(3, [0.15, 0.7, 0.15]), hand_present=True)
    cls, _ = s.get()
    assert cls == 0  # smaller id breaks the exact tie


def test_clears_after_5_no_hand_frames():
    s = Smoother(window=7, threshold=0.0, no_hand_clear_frames=5, num_classes=3)
    for _ in range(7):
        s.update(_probs(3, [0.9, 0.05, 0.05]), hand_present=True)
    pre_cls, pre_conf = s.get()
    assert pre_cls == 0

    for _ in range(4):
        s.update(None, hand_present=False)
    cls_during, conf_during = s.get()
    assert cls_during == 0
    assert conf_during == pytest.approx(pre_conf)

    s.update(None, hand_present=False)  # 5th no-hand → clears
    assert s.get() == (None, 0.0)
    assert len(s.window) == 0
    assert s.no_hand_streak == 0


def test_no_hand_streak_resets_on_hand_present():
    s = Smoother(window=7, threshold=0.0, no_hand_clear_frames=5, num_classes=3)
    for _ in range(7):
        s.update(_probs(3, [0.9, 0.05, 0.05]), hand_present=True)
    for _ in range(3):
        s.update(None, hand_present=False)
    s.update(_probs(3, [0.9, 0.05, 0.05]), hand_present=True)  # resets streak
    for _ in range(4):
        s.update(None, hand_present=False)
    # Still under 5 consecutive no-hand frames → smoother remains live.
    cls, _ = s.get()
    assert cls == 0


def test_update_validates_probs_shape():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    with pytest.raises(ValueError):
        s.update(np.zeros(2, dtype=np.float32), hand_present=True)


def test_update_validates_probs_finite():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    bad = np.array([np.nan, 0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        s.update(bad, hand_present=True)


def test_update_rejects_none_with_hand_present():
    s = Smoother(window=7, threshold=0.75, no_hand_clear_frames=5, num_classes=3)
    with pytest.raises(ValueError):
        s.update(None, hand_present=True)


def test_reset_clears_state():
    s = Smoother(window=7, threshold=0.0, no_hand_clear_frames=5, num_classes=3)
    for _ in range(7):
        s.update(_probs(3, [0.9, 0.05, 0.05]), hand_present=True)
    assert s.get()[0] == 0
    s.reset()
    assert s.get() == (None, 0.0)
    assert len(s.window) == 0
    assert s.no_hand_streak == 0
