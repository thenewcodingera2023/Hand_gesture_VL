"""Unit tests for src/inference.py — helpers only (no webcam, no mediapipe)."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from src.feature_assembler import (
    INTER_DIST_IDX,
    LEFT_FEAT_SLICE,
    LEFT_PRESENT_IDX,
    RIGHT_FEAT_SLICE,
    RIGHT_PRESENT_IDX,
)
from src.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LABELS,
    build_feature_vector,
    load_inference_artifacts,
    predict_probs,
)
from src.models.mlp import NUM_CLASSES
from src.preprocessor import TWO_HAND_DIM


@dataclass
class MockHand:
    landmarks_xy: np.ndarray
    handedness: str
    score: float = 0.95


def _synthetic_hand_xy(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, size=(21, 2)).astype(np.float32)
    # Force a non-zero palm size by spacing wrist (0) and middle MCP (9).
    base[0] = np.array([0.4, 0.5], dtype=np.float32)
    base[9] = np.array([0.6, 0.3], dtype=np.float32)
    return base


def test_load_artifacts_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inference_artifacts(checkpoint_path=tmp_path / "nope.pt")


def test_load_artifacts_real_checkpoint():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present at {DEFAULT_CHECKPOINT}")
    artifacts = load_inference_artifacts()
    assert artifacts.model.input_dim == 279
    assert artifacts.model.num_classes == NUM_CLASSES
    assert artifacts.scaler_mean.shape == (279,)
    assert artifacts.scaler_scale.shape == (279,)
    assert np.all(artifacts.scaler_scale > 0)
    assert set(artifacts.id_to_label.keys()) == set(range(NUM_CLASSES))

    with open(DEFAULT_LABELS, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    assert set(artifacts.id_to_label.values()) == set(name_to_id.keys())


def test_inverse_label_mapping_complete():
    with open(DEFAULT_LABELS, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    inverse = {int(v): k for k, v in name_to_id.items()}
    assert len(inverse) == NUM_CLASSES
    assert set(inverse.keys()) == set(range(NUM_CLASSES))


def test_build_feature_vector_empty_hands():
    feat, present = build_feature_vector([])
    assert feat.shape == (TWO_HAND_DIM,)
    assert feat.dtype == np.float32
    assert np.all(feat == 0.0)
    assert present is False


def test_build_feature_vector_one_hand_right():
    h = MockHand(landmarks_xy=_synthetic_hand_xy(0), handedness="Right")
    feat, present = build_feature_vector([h])
    assert feat.shape == (TWO_HAND_DIM,)
    assert present is True
    assert feat[RIGHT_PRESENT_IDX] == 1.0
    assert feat[LEFT_PRESENT_IDX] == 0.0
    assert feat[INTER_DIST_IDX] == 0.0
    # Right slot must be non-trivial (preprocessor ran).
    assert np.linalg.norm(feat[RIGHT_FEAT_SLICE]) > 0.0
    assert np.all(feat[LEFT_FEAT_SLICE] == 0.0)


def test_build_feature_vector_one_hand_left():
    h = MockHand(landmarks_xy=_synthetic_hand_xy(1), handedness="Left")
    feat, present = build_feature_vector([h])
    assert present is True
    assert feat[RIGHT_PRESENT_IDX] == 0.0
    assert feat[LEFT_PRESENT_IDX] == 1.0
    assert np.all(feat[RIGHT_FEAT_SLICE] == 0.0)
    assert np.linalg.norm(feat[LEFT_FEAT_SLICE]) > 0.0


def test_build_feature_vector_two_hands():
    right = MockHand(landmarks_xy=_synthetic_hand_xy(2), handedness="Right")
    left = MockHand(landmarks_xy=_synthetic_hand_xy(3), handedness="Left")
    feat, present = build_feature_vector([right, left])
    assert present is True
    assert feat.shape == (TWO_HAND_DIM,)
    assert feat[RIGHT_PRESENT_IDX] == 1.0
    assert feat[LEFT_PRESENT_IDX] == 1.0
    assert feat[INTER_DIST_IDX] >= 0.0
    assert np.isfinite(feat[INTER_DIST_IDX])


def test_build_feature_vector_two_same_handedness_keeps_higher_score():
    xy_lo = _synthetic_hand_xy(10)
    xy_hi = _synthetic_hand_xy(11)
    # Mark a known landmark differently so we can detect which one was kept.
    xy_hi[5] = np.array([0.123, 0.456], dtype=np.float32)

    lo = MockHand(landmarks_xy=xy_lo, handedness="Right", score=0.55)
    hi = MockHand(landmarks_xy=xy_hi, handedness="Right", score=0.92)

    feat_from_lo, _ = build_feature_vector([lo])
    feat_from_hi, _ = build_feature_vector([hi])

    # Order should not matter; higher score always wins.
    feat_pair_a, _ = build_feature_vector([lo, hi])
    feat_pair_b, _ = build_feature_vector([hi, lo])
    np.testing.assert_allclose(feat_pair_a[RIGHT_FEAT_SLICE], feat_from_hi[RIGHT_FEAT_SLICE])
    np.testing.assert_allclose(feat_pair_b[RIGHT_FEAT_SLICE], feat_from_hi[RIGHT_FEAT_SLICE])
    # And the lo-score landmark layout must NOT match the right slot.
    assert not np.allclose(feat_pair_a[RIGHT_FEAT_SLICE], feat_from_lo[RIGHT_FEAT_SLICE])


def test_build_feature_vector_rejects_bad_landmark_shape():
    bad = MockHand(landmarks_xy=np.zeros((20, 2), dtype=np.float32), handedness="Right")
    with pytest.raises(ValueError):
        build_feature_vector([bad])


def test_predict_probs_shape_and_sum():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present at {DEFAULT_CHECKPOINT}")
    artifacts = load_inference_artifacts()
    feat = np.zeros(TWO_HAND_DIM, dtype=np.float32)
    probs = predict_probs(
        artifacts.model, feat, artifacts.scaler_mean, artifacts.scaler_scale, artifacts.device
    )
    assert probs.shape == (NUM_CLASSES,)
    assert probs.dtype == np.float32
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_predict_probs_validates_input_shape():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present at {DEFAULT_CHECKPOINT}")
    artifacts = load_inference_artifacts()
    with pytest.raises(ValueError):
        predict_probs(
            artifacts.model,
            np.zeros(100, dtype=np.float32),
            artifacts.scaler_mean,
            artifacts.scaler_scale,
            artifacts.device,
        )


def test_model_forward_eval_batch_one():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present at {DEFAULT_CHECKPOINT}")
    artifacts = load_inference_artifacts()
    artifacts.model.eval()
    with torch.no_grad():
        out = artifacts.model(torch.randn(1, 279))
    assert out.shape == (1, NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_softmax_sums_to_one():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present at {DEFAULT_CHECKPOINT}")
    artifacts = load_inference_artifacts()
    with torch.no_grad():
        probs = torch.softmax(artifacts.model(torch.randn(1, 279)), dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
