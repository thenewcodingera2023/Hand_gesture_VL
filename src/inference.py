"""Real-time inference loop: webcam -> MediaPipe -> MLP -> overlay.

Stage 5 entry point. Loads the Stage 4 checkpoint at ``runs/mlp_best.pt``,
opens the webcam via :class:`src.capture.HandCapture`, converts each frame's
MediaPipe detections into the trained 279-d feature vector through the existing
``src.feature_assembler.assemble_from_hands`` path, standardises with the
checkpoint's persisted ``scaler_mean``/``scaler_scale``, runs ``GestureMLP`` in
``eval()`` mode under ``torch.no_grad()``, smooths predictions over a 7-frame
window with a 0.75 confidence gate, and overlays the label + confidence + FPS
on the OpenCV preview.

Two non-obvious training/inference contracts honoured here:

1. **StandardScaler**: training fits a scaler on the train split; the persisted
   ``scaler_mean`` and ``scaler_scale`` (length 279 each) must be applied to
   every live feature vector before the forward pass.
2. **z=0 stripping**: every HaGRIDv2 training row has ``landmarks_raw[:,2]==0``.
   Live MediaPipe z is dropped via ``DetectedHand.landmarks_xy`` so that
   ``preprocessor.pad_z`` re-fills z with zeros and the live distribution
   matches training.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from src.feature_assembler import assemble_from_hands
from src.models.mlp import GestureMLP
from src.preprocessor import BONE_PAIRS, TWO_HAND_DIM
from src.smoother import Smoother

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "mlp_best.pt"
DEFAULT_LABELS = REPO_ROOT / "data" / "labels.json"

EXPECTED_HIDDEN_DIMS: tuple[int, int, int] = (256, 128, 64)
EXPECTED_DROPOUTS: tuple[float, float, float] = (0.3, 0.3, 0.2)
EXPECTED_INPUT_DIM = 279
EXPECTED_NUM_CLASSES = 28

WINDOW_TITLE = "Gesture Inference"


@dataclass
class InferenceArtifacts:
    model: GestureMLP
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    id_to_label: dict[int, str]
    device: torch.device


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("warning: --device cuda requested but CUDA unavailable; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def _load_id_to_label(labels_path: pathlib.Path) -> dict[int, str]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    if not isinstance(name_to_id, dict) or len(name_to_id) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"labels file must contain {EXPECTED_NUM_CLASSES} entries; got {len(name_to_id) if isinstance(name_to_id, dict) else type(name_to_id).__name__}"
        )
    id_to_label: dict[int, str] = {}
    for name, idx in name_to_id.items():
        id_to_label[int(idx)] = str(name)
    if set(id_to_label.keys()) != set(range(EXPECTED_NUM_CLASSES)):
        raise ValueError(
            f"labels file ids must be 0..{EXPECTED_NUM_CLASSES - 1}; got {sorted(id_to_label.keys())}"
        )
    return id_to_label


def load_inference_artifacts(
    checkpoint_path: pathlib.Path = DEFAULT_CHECKPOINT,
    labels_path: pathlib.Path = DEFAULT_LABELS,
    device: str = "cpu",
) -> InferenceArtifacts:
    checkpoint_path = pathlib.Path(checkpoint_path)
    labels_path = pathlib.Path(labels_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    id_to_label = _load_id_to_label(labels_path)
    torch_device = _resolve_device(device)

    ck = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    cfg = ck.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint missing 'config' dict; keys={sorted(ck.keys())}")
    if (
        int(cfg.get("input_dim", -1)) != EXPECTED_INPUT_DIM
        or int(cfg.get("num_classes", -1)) != EXPECTED_NUM_CLASSES
        or tuple(cfg.get("hidden_dims", ())) != EXPECTED_HIDDEN_DIMS
        or tuple(cfg.get("dropouts", ())) != EXPECTED_DROPOUTS
    ):
        raise RuntimeError(
            "checkpoint config does not match GestureMLP architecture. "
            f"expected input_dim={EXPECTED_INPUT_DIM}, num_classes={EXPECTED_NUM_CLASSES}, "
            f"hidden_dims={EXPECTED_HIDDEN_DIMS}, dropouts={EXPECTED_DROPOUTS}; "
            f"got {cfg}"
        )

    ck_labels = ck.get("labels")
    if isinstance(ck_labels, dict):
        ck_labels_int = {int(k): str(v) for k, v in ck_labels.items()}
        if ck_labels_int != id_to_label:
            raise RuntimeError(
                "label mapping in checkpoint disagrees with data/labels.json; "
                "retrain or align label files before running Stage 5"
            )

    model = GestureMLP(
        input_dim=EXPECTED_INPUT_DIM,
        hidden_dims=EXPECTED_HIDDEN_DIMS,
        dropouts=EXPECTED_DROPOUTS,
        num_classes=EXPECTED_NUM_CLASSES,
    )
    try:
        model.load_state_dict(ck["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint state_dict does not match GestureMLP; ck['config']={cfg}: {exc}"
        ) from exc
    model.to(torch_device)
    model.eval()

    scaler_mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(ck["scaler_scale"], dtype=np.float32)
    if scaler_mean.shape != (EXPECTED_INPUT_DIM,):
        raise RuntimeError(f"scaler_mean must be ({EXPECTED_INPUT_DIM},); got {scaler_mean.shape}")
    if scaler_scale.shape != (EXPECTED_INPUT_DIM,):
        raise RuntimeError(f"scaler_scale must be ({EXPECTED_INPUT_DIM},); got {scaler_scale.shape}")
    if not np.all(scaler_scale > 0):
        raise RuntimeError("scaler_scale must be strictly positive")

    return InferenceArtifacts(
        model=model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        id_to_label=id_to_label,
        device=torch_device,
    )


def build_feature_vector(hands) -> tuple[np.ndarray, bool]:
    """Convert a list of DetectedHand-like objects into a (279,) float32 vector.

    Each item must expose ``landmarks_xy`` (shape (21, 2), float),
    ``handedness`` ("Left" | "Right"), and ``score`` (float).

    z is intentionally stripped: training HaGRID rows are z=0 throughout, so the
    live ``(21, 2)`` array flows into ``preprocessor.pad_z`` which fills z with
    zeros — matching the training distribution exactly.
    """
    if not hands:
        return (np.zeros(TWO_HAND_DIM, dtype=np.float32), False)

    by_hand: dict[str, tuple[float, np.ndarray]] = {}
    for h in hands:
        label = getattr(h, "handedness", None)
        if label not in ("Left", "Right"):
            continue
        xy = np.asarray(getattr(h, "landmarks_xy"), dtype=np.float32)
        if xy.shape != (21, 2):
            raise ValueError(f"landmarks_xy must have shape (21, 2); got {xy.shape}")
        score = float(getattr(h, "score", 0.0))
        existing = by_hand.get(label)
        if existing is None or score > existing[0]:
            by_hand[label] = (score, xy)

    if not by_hand:
        return (np.zeros(TWO_HAND_DIM, dtype=np.float32), False)

    pairs: list[tuple[np.ndarray, str]] = []
    if "Right" in by_hand:
        pairs.append((by_hand["Right"][1], "Right"))
    if "Left" in by_hand:
        pairs.append((by_hand["Left"][1], "Left"))

    feat = assemble_from_hands(pairs)
    if feat.shape != (TWO_HAND_DIM,):
        raise AssertionError(f"assembled feature shape mismatch: {feat.shape}")
    return (feat.astype(np.float32, copy=False), True)


def predict_probs(
    model: GestureMLP,
    x_279: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if x_279.shape != (TWO_HAND_DIM,):
        raise ValueError(f"x_279 must be shape (279,); got {x_279.shape}")
    x_std = (x_279.astype(np.float32, copy=False) - mean) / scale
    x_t = torch.from_numpy(x_std.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_t)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    return probs


def _put_text_with_outline(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    thickness: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame_bgr: np.ndarray,
    label: Optional[str],
    confidence: float,
    fps: float,
    hands,
    draw_landmarks: bool,
) -> None:
    h, w = frame_bgr.shape[:2]
    display_label = label if label is not None else "---"

    _put_text_with_outline(frame_bgr, display_label, (10, 35), 1.1, 2)
    _put_text_with_outline(frame_bgr, f"conf: {confidence:.2f}", (10, 70), 0.7, 2)
    _put_text_with_outline(frame_bgr, f"FPS: {fps:.1f}", (10, 100), 0.7, 2)
    _put_text_with_outline(frame_bgr, "q/Esc to quit", (10, h - 10), 0.5, 1)

    if not draw_landmarks:
        return

    box_color = (0, 255, 0)      # green BGR
    label_color = (0, 0, 255)    # red BGR
    label_scale = 0.9
    label_thickness = 2
    bone_color = (200, 200, 200)
    dot_color = (0, 255, 0)
    bbox_pad = 15

    for h_obj in hands:
        xy = np.asarray(getattr(h_obj, "landmarks_xy"), dtype=np.float32)
        if xy.shape != (21, 2):
            continue
        px = (xy[:, 0] * w).astype(np.int32)
        py = (xy[:, 1] * h).astype(np.int32)

        x1 = int(max(0, px.min() - bbox_pad))
        y1 = int(max(0, py.min() - bbox_pad))
        x2 = int(min(w - 1, px.max() + bbox_pad))
        y2 = int(min(h - 1, py.max() + bbox_pad))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        (tw, th), baseline = cv2.getTextSize(
            display_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness
        )
        text_x = int(max(0, min(w - tw, x1 + (x2 - x1 - tw) // 2)))
        if y1 - 8 - baseline >= th:
            text_y = y1 - 8
        else:
            text_y = min(h - 4, y1 + th + 8)
        _put_text_with_outline(
            frame_bgr, display_label, (text_x, text_y), label_scale, label_thickness, label_color
        )

        for parent, child in BONE_PAIRS:
            cv2.line(
                frame_bgr,
                (int(px[parent]), int(py[parent])),
                (int(px[child]), int(py[child])),
                bone_color,
                1,
                cv2.LINE_AA,
            )
        for i in range(21):
            cv2.circle(frame_bgr, (int(px[i]), int(py[i])), 3, dot_color, -1, cv2.LINE_AA)


def run(
    camera_index: int = 0,
    checkpoint_path: pathlib.Path = DEFAULT_CHECKPOINT,
    labels_path: pathlib.Path = DEFAULT_LABELS,
    window: int = 7,
    threshold: float = 0.75,
    no_hand_clear_frames: int = 5,
    min_detection_confidence: float = 0.5,
    device: str = "cpu",
    draw_landmarks: bool = True,
) -> None:
    from src.capture import HandCapture  # lazy import: avoids mediapipe at test time

    artifacts = load_inference_artifacts(checkpoint_path, labels_path, device)
    smoother = Smoother(
        window=window,
        threshold=threshold,
        no_hand_clear_frames=no_hand_clear_frames,
        num_classes=artifacts.model.num_classes,
    )
    fps_times: collections.deque[float] = collections.deque(maxlen=30)

    print(
        f"[inference] loaded checkpoint {checkpoint_path} on device {artifacts.device}; "
        f"num_classes={artifacts.model.num_classes}, threshold={threshold}, window={window}"
    )

    with HandCapture(
        camera_index=camera_index,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as cap:
        for frame_bgr, hands in cap.frames():
            feat, hand_present = build_feature_vector(hands)
            if hand_present:
                probs = predict_probs(
                    artifacts.model,
                    feat,
                    artifacts.scaler_mean,
                    artifacts.scaler_scale,
                    artifacts.device,
                )
                smoother.update(probs, hand_present=True)
            else:
                smoother.update(None, hand_present=False)

            class_id, conf = smoother.get()
            label = artifacts.id_to_label.get(class_id) if class_id is not None else None

            fps_times.append(time.perf_counter())
            if len(fps_times) >= 2:
                fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-6)
            else:
                fps = 0.0

            draw_overlay(frame_bgr, label, conf, fps, hands, draw_landmarks)
            cv2.imshow(WINDOW_TITLE, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 real-time gesture inference")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=DEFAULT_CHECKPOINT,
        help=f"path to MLP checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--labels",
        type=pathlib.Path,
        default=DEFAULT_LABELS,
        help=f"path to labels.json (default: {DEFAULT_LABELS})",
    )
    parser.add_argument("--window", type=int, default=7, help="smoother window size (default: 7)")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="confidence threshold (default: 0.75)"
    )
    parser.add_argument(
        "--no-hand-clear",
        type=int,
        default=5,
        help="frames of no-hand before smoother clears (default: 5)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min hand detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="torch device (default: cpu)"
    )
    parser.add_argument(
        "--no-landmarks", action="store_true", help="disable landmark drawing on the overlay"
    )
    args = parser.parse_args()

    run(
        camera_index=args.camera,
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        window=args.window,
        threshold=args.threshold,
        no_hand_clear_frames=args.no_hand_clear,
        min_detection_confidence=args.min_detection_confidence,
        device=args.device,
        draw_landmarks=not args.no_landmarks,
    )


if __name__ == "__main__":
    main()
