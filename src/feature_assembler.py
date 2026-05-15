"""Two-hand combiner producing the (279,) input feature vector.

Layout (see `tasks/gesture_recognition_plan_v2.md` §5.2):
    [0:138]   right-hand features  (138,)
    [138:276] left-hand  features  (138,)
    [276]     right_present flag   {0.0, 1.0}
    [277]     left_present flag    {0.0, 1.0}
    [278]     inter-hand wrist distance (in right-palm-size units, 0.0 if any hand absent)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from src.preprocessor import PER_HAND_DIM, TWO_HAND_DIM, WRIST, preprocess_hand

RIGHT_FEAT_SLICE = slice(0, 138)
LEFT_FEAT_SLICE = slice(138, 276)
RIGHT_PRESENT_IDX = 276
LEFT_PRESENT_IDX = 277
INTER_DIST_IDX = 278


def assemble_features(
    right_features: Optional[np.ndarray],
    left_features: Optional[np.ndarray],
    inter_hand_distance: float = 0.0,
) -> np.ndarray:
    out = np.zeros(TWO_HAND_DIM, dtype=np.float32)

    if right_features is not None:
        rf = np.asarray(right_features, dtype=np.float32)
        assert rf.shape == (PER_HAND_DIM,), f"right_features must be ({PER_HAND_DIM},); got {rf.shape}"
        out[RIGHT_FEAT_SLICE] = rf
        out[RIGHT_PRESENT_IDX] = 1.0

    if left_features is not None:
        lf = np.asarray(left_features, dtype=np.float32)
        assert lf.shape == (PER_HAND_DIM,), f"left_features must be ({PER_HAND_DIM},); got {lf.shape}"
        out[LEFT_FEAT_SLICE] = lf
        out[LEFT_PRESENT_IDX] = 1.0

    if right_features is not None and left_features is not None:
        d = float(inter_hand_distance)
        if not math.isfinite(d) or d < 0.0:
            d = 0.0
        out[INTER_DIST_IDX] = d

    return out


def assemble_from_hands(
    detected_hands: Iterable[tuple[np.ndarray, str]],
) -> np.ndarray:
    """Convenience: take raw landmarks and handedness labels, run the
    per-hand pipeline, and assemble.

    Inter-hand distance is computed from RAW (pre-normalised) wrist coordinates
    divided by the right hand's palm size when both hands are present; 0.0
    otherwise.
    """
    right_raw: Optional[np.ndarray] = None
    left_raw: Optional[np.ndarray] = None

    for raw, hand in detected_hands:
        if hand == "Right":
            right_raw = np.asarray(raw, dtype=np.float32)
        elif hand == "Left":
            left_raw = np.asarray(raw, dtype=np.float32)

    right_feat = preprocess_hand(right_raw, "Right") if right_raw is not None else None
    left_feat = preprocess_hand(left_raw, "Left") if left_raw is not None else None

    inter = 0.0
    if right_raw is not None and left_raw is not None:
        # Normalize the right hand once to recover palm size in the same units
        # as the raw landmarks: the divisor used during normalize_hand.
        right_padded = right_raw if right_raw.shape[1] == 3 else np.hstack(
            [right_raw, np.zeros((right_raw.shape[0], 1), dtype=np.float32)]
        )
        right_translated = right_padded - right_padded[WRIST]
        palm_size = float(np.linalg.norm(right_translated[9]))
        wrist_gap = float(np.linalg.norm(right_padded[WRIST] - (
            left_raw if left_raw.shape[1] == 3 else np.hstack(
                [left_raw, np.zeros((left_raw.shape[0], 1), dtype=np.float32)]
            )
        )[WRIST]))
        if palm_size > 1e-6:
            inter = wrist_gap / palm_size

    return assemble_features(right_feat, left_feat, inter)
