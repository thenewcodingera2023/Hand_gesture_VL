"""Stage 0 smoke test: verify MediaPipe HandLandmarker loads and processes a frame."""
import sys
import pathlib
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = str(pathlib.Path(__file__).parent / "hand_landmarker.task")


def test_mediapipe_handlandmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )

    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    # A blank frame won't have hands — that's fine for loading verification.
    # Check the result object has the expected structure.
    assert hasattr(result, "hand_landmarks"), "Result missing hand_landmarks"
    assert hasattr(result, "handedness"), "Result missing handedness"
    assert isinstance(result.hand_landmarks, list)
    assert isinstance(result.handedness, list)

    if result.hand_landmarks:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            assert len(landmarks) == 21, "Expected 21 NormalizedLandmark per hand"
            label = handedness[0].category_name
            assert label in ("Left", "Right"), f"Unexpected handedness: {label}"
            print(f"  Hand detected: {label}, landmarks[0]=(x={landmarks[0].x:.3f}, y={landmarks[0].y:.3f}, z={landmarks[0].z:.3f})")
    else:
        print("  No hands in blank frame — HandLandmarker loaded and returned valid empty result.")

    print("MediaPipe HandLandmarker smoke test passed.")


if __name__ == "__main__":
    test_mediapipe_handlandmarker()
    print("ok")
