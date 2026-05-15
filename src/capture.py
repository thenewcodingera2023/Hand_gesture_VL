"""Webcam capture wrapping OpenCV ``VideoCapture`` and MediaPipe HandLandmarker.

Stage 5 component. Provides :class:`HandCapture` as a context manager whose
``frames()`` method yields ``(frame_bgr, hands)`` per webcam frame, where
``hands`` is a list of :class:`DetectedHand`.

Mirrors the MediaPipe Tasks invocation used by ``smoke_test_mediapipe.py`` so
the existing ``hand_landmarker.task`` asset at the repo root is consumed
directly. Uses ``RunningMode.IMAGE``: simpler than VIDEO mode, fast enough on
laptop CPU to hit the Stage 5 25-FPS gate.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HAND_LANDMARKER_TASK = REPO_ROOT / "hand_landmarker.task"


@dataclass
class DetectedHand:
    landmarks_xy: np.ndarray   # (21, 2) float32, MediaPipe-normalised image coords [0, 1]
    landmarks_xyz: np.ndarray  # (21, 3) float32, raw MediaPipe values (z kept for overlay only)
    handedness: str            # "Left" or "Right"
    score: float               # MediaPipe handedness category score


class HandCapture:
    def __init__(
        self,
        camera_index: int = 0,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,  # accepted for API symmetry; unused in IMAGE mode
        model_path: pathlib.Path = HAND_LANDMARKER_TASK,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        if not pathlib.Path(model_path).is_file():
            raise FileNotFoundError(f"MediaPipe HandLandmarker task asset not found: {model_path}")

        # On Windows, DSHOW backend avoids the slow MSMF cold-start warm-up.
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            raise RuntimeError(
                f"failed to open camera index {camera_index}; pass --camera N to pick a different one"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
        self._cap = cap

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=int(max_num_hands),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_detection_confidence),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._released = False

    def frames(self) -> Iterator[tuple[np.ndarray, list[DetectedHand]]]:
        while True:
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("camera read failed")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._detector.detect(mp_image)

            hands: list[DetectedHand] = []
            handedness_list = getattr(result, "handedness", []) or []
            landmarks_list = getattr(result, "hand_landmarks", []) or []
            for i, landmarks in enumerate(landmarks_list):
                if i >= len(handedness_list) or not handedness_list[i]:
                    continue
                cat = handedness_list[i][0]
                label = getattr(cat, "category_name", None)
                if label not in ("Left", "Right"):
                    continue
                score = float(getattr(cat, "score", 0.0))
                xyz = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
                )
                if xyz.shape != (21, 3):
                    continue
                xy = xyz[:, :2].copy()
                hands.append(
                    DetectedHand(
                        landmarks_xy=xy,
                        landmarks_xyz=xyz,
                        handedness=label,
                        score=score,
                    )
                )

            yield frame_bgr, hands

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._detector.close()
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "HandCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
