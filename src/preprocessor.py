"""Per-hand normalization and feature engineering -> (138,) vector.

The pipeline mirrors `tasks/gesture_recognition_plan_v2.md` §5.1:
    pad_z -> wrist-translate -> palm-scale -> left-mirror
    -> [normalised landmarks (63), bone vectors (60),
        extension ratios (5), pairwise fingertip distances (10)]
"""

from __future__ import annotations

import numpy as np

N_LANDMARKS = 21
WRIST = 0
PALM_REF = 9
FINGERTIPS = (4, 8, 12, 16, 20)

BONE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGER_BONE_GROUPS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
)

PER_HAND_DIM = 138
TWO_HAND_DIM = 279

_VALID_HANDS = ("Left", "Right")
_EPS = 1e-6


def pad_z(landmarks: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != N_LANDMARKS:
        raise ValueError(
            f"landmarks must have shape ({N_LANDMARKS}, 2|3); got {arr.shape}"
        )
    if arr.shape[1] == 3:
        return arr.copy()
    if arr.shape[1] == 2:
        return np.hstack([arr, np.zeros((N_LANDMARKS, 1), dtype=np.float32)])
    raise ValueError(
        f"landmarks last dim must be 2 or 3; got {arr.shape[1]}"
    )


def normalize_hand(landmarks: np.ndarray, hand: str) -> np.ndarray:
    if hand not in _VALID_HANDS:
        raise ValueError(f"hand must be one of {_VALID_HANDS}; got {hand!r}")
    lm = pad_z(landmarks)
    lm = lm - lm[WRIST]
    palm = float(np.linalg.norm(lm[PALM_REF]))
    lm = lm / (palm + _EPS)
    if hand == "Left":
        lm[:, 0] *= -1.0
    return lm.astype(np.float32, copy=False)


def bone_vectors(landmarks: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (N_LANDMARKS, 3):
        raise ValueError(f"landmarks must be (21,3); got {lm.shape}")
    out = np.empty((len(BONE_PAIRS), 3), dtype=np.float32)
    for i, (parent, child) in enumerate(BONE_PAIRS):
        out[i] = lm[child] - lm[parent]
    return out.reshape(-1)


def extension_ratios(landmarks: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (N_LANDMARKS, 3):
        raise ValueError(f"landmarks must be (21,3); got {lm.shape}")
    out = np.empty(len(FINGER_BONE_GROUPS), dtype=np.float32)
    for i, group in enumerate(FINGER_BONE_GROUPS):
        chain = 0.0
        for bone_idx in group:
            parent, child = BONE_PAIRS[bone_idx]
            chain += float(np.linalg.norm(lm[child] - lm[parent]))
        first_parent = BONE_PAIRS[group[0]][0]
        last_child = BONE_PAIRS[group[-1]][1]
        straight = float(np.linalg.norm(lm[last_child] - lm[first_parent]))
        ratio = straight / (chain + _EPS)
        out[i] = ratio
    return np.clip(out, 0.0, 1.0)


def pairwise_fingertip_distances(landmarks: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape != (N_LANDMARKS, 3):
        raise ValueError(f"landmarks must be (21,3); got {lm.shape}")
    n = len(FINGERTIPS)
    out = np.empty(n * (n - 1) // 2, dtype=np.float32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = float(np.linalg.norm(lm[FINGERTIPS[i]] - lm[FINGERTIPS[j]]))
            k += 1
    return out


def preprocess_hand(landmarks: np.ndarray, hand: str) -> np.ndarray:
    lm = normalize_hand(landmarks, hand)
    flat = lm.reshape(-1)
    bv = bone_vectors(lm)
    ext = extension_ratios(lm)
    pd = pairwise_fingertip_distances(lm)
    out = np.concatenate([flat, bv, ext, pd]).astype(np.float32, copy=False)
    if out.shape != (PER_HAND_DIM,):
        raise ValueError(f"per-hand feature shape mismatch: {out.shape}")
    return out
