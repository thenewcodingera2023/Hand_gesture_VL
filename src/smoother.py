"""Sliding-window majority-vote smoother with confidence gate and no-hand clear.

Stage 5 component. Stateful smoother over per-frame softmax probabilities.
Pure NumPy / standard library — no torch, no MediaPipe, no OpenCV imports.

Behaviour contract (gesture_recognition_plan_v2.md §7.2, Stage 5 plan §3.2):

- Maintains a sliding window of the last `window` (default 7) per-frame
  softmax vectors, each of shape (num_classes,).
- `get()` returns (class_id, confidence):
    * `class_id` = majority vote over per-frame argmaxes.
      Ties broken by highest mean probability across the window;
      secondary tie broken by smallest class id (deterministic).
    * `confidence` = mean over the window of probs[picked_class].
    * If `confidence < threshold` (default 0.75), returns (None, confidence).
    * Returns (None, 0.0) before the window is full.
- No-hand handling: `update(None, hand_present=False)` increments a streak
  counter. After `no_hand_clear_frames` (default 5) consecutive no-hand
  updates, the window is cleared and the smoother goes silent.
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np


class Smoother:
    def __init__(
        self,
        window: int = 7,
        threshold: float = 0.75,
        no_hand_clear_frames: int = 5,
        num_classes: int = 28,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1]; got {threshold}")
        if no_hand_clear_frames < 1:
            raise ValueError(
                f"no_hand_clear_frames must be >= 1; got {no_hand_clear_frames}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")

        self.window_size = int(window)
        self.threshold = float(threshold)
        self.no_hand_clear_frames = int(no_hand_clear_frames)
        self.num_classes = int(num_classes)

        self.window: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.window_size
        )
        self.no_hand_streak: int = 0

    def update(
        self, probs: Optional[np.ndarray], hand_present: bool
    ) -> None:
        if not hand_present:
            self.no_hand_streak += 1
            if self.no_hand_streak >= self.no_hand_clear_frames:
                self.reset()
            return

        if probs is None:
            raise ValueError("probs required when hand_present=True")
        arr = np.asarray(probs, dtype=np.float32)
        if arr.shape != (self.num_classes,):
            raise ValueError(
                f"probs must have shape ({self.num_classes},); got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("probs contains NaN or inf")

        self.no_hand_streak = 0
        self.window.append(arr.astype(np.float32, copy=True))

    def get(self) -> tuple[Optional[int], float]:
        if len(self.window) < self.window_size:
            return (None, 0.0)

        W = np.stack(list(self.window), axis=0)
        argmaxes = W.argmax(axis=1)
        counts = np.bincount(argmaxes, minlength=self.num_classes)
        max_count = int(counts.max())
        candidates = np.flatnonzero(counts == max_count)

        if candidates.size == 1:
            picked = int(candidates[0])
        else:
            mean_probs = W[:, candidates].mean(axis=0)
            best = int(mean_probs.argmax())
            top_mean = float(mean_probs[best])
            tied = candidates[np.isclose(mean_probs, top_mean, rtol=0.0, atol=0.0)]
            picked = int(tied.min())

        conf = float(W[:, picked].mean())
        if conf < self.threshold:
            return (None, conf)
        return (picked, conf)

    def reset(self) -> None:
        self.window.clear()
        self.no_hand_streak = 0
