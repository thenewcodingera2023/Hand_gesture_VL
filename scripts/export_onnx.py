"""Export runs/mlp_best.pt -> web/public/model/gesture_mlp.onnx + scaler.json + labels.json.

Single-purpose script. Refuses to emit a stale 28-class artifact. Verifies
ONNX <-> PyTorch numerical parity on random inputs before declaring success.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import (  # noqa: E402
    DROPOUTS as EXPECTED_DROPOUTS,
    HIDDEN_DIMS as EXPECTED_HIDDEN_DIMS,
    INPUT_DIM as EXPECTED_INPUT_DIM,
    NUM_CLASSES as EXPECTED_NUM_CLASSES,
    GestureMLP,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "mlp_best.pt"
DEFAULT_LABELS = REPO_ROOT / "data" / "labels.json"
DEFAULT_OUT_DIR = REPO_ROOT / "web" / "public" / "model"
DEFAULT_MP_TASK = REPO_ROOT / "hand_landmarker.task"
DEFAULT_MP_OUT = REPO_ROOT / "web" / "public" / "mediapipe" / "hand_landmarker.task"

PARITY_BATCH = 16
PARITY_ATOL = 1e-5


def _load_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return ck


def _build_model(ck: dict) -> GestureMLP:
    model = GestureMLP(
        input_dim=EXPECTED_INPUT_DIM,
        hidden_dims=EXPECTED_HIDDEN_DIMS,
        dropouts=EXPECTED_DROPOUTS,
        num_classes=EXPECTED_NUM_CLASSES,
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def _validate_labels(labels_path: pathlib.Path) -> dict[str, int]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    if not isinstance(name_to_id, dict) or len(name_to_id) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"labels must have {EXPECTED_NUM_CLASSES} entries; got {len(name_to_id) if isinstance(name_to_id, dict) else type(name_to_id).__name__}"
        )
    ids = {int(v) for v in name_to_id.values()}
    if ids != set(range(EXPECTED_NUM_CLASSES)):
        raise ValueError(f"label ids must be dense 0..{EXPECTED_NUM_CLASSES - 1}; got {sorted(ids)}")
    if int(name_to_id.get("peace", -1)) != 8 or int(name_to_id.get("open_palm", -1)) != 9:
        raise ValueError("post-fix schema requires peace=8 and open_palm=9")
    if "count_2" in name_to_id or "count_5" in name_to_id:
        raise ValueError("post-fix schema must not contain count_2 or count_5")
    return {str(k): int(v) for k, v in name_to_id.items()}


def _export_onnx(model: GestureMLP, out_path: pathlib.Path) -> None:
    dummy = torch.zeros(1, EXPECTED_INPUT_DIM, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        opset_version=17,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )


def _verify_parity(model: GestureMLP, onnx_path: pathlib.Path) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(20260518)
    x = rng.standard_normal((PARITY_BATCH, EXPECTED_INPUT_DIM)).astype(np.float32)

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).cpu().numpy()

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"x": x})[0]

    if onnx_logits.shape != torch_logits.shape:
        raise RuntimeError(f"shape mismatch: onnx={onnx_logits.shape} torch={torch_logits.shape}")
    max_abs = float(np.max(np.abs(onnx_logits - torch_logits)))
    if max_abs > PARITY_ATOL:
        raise RuntimeError(f"ONNX<->PyTorch parity FAILED: max_abs={max_abs:.3e} > atol={PARITY_ATOL:.0e}")
    return max_abs


def _write_scaler(ck: dict, out_path: pathlib.Path) -> None:
    mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ck["scaler_scale"], dtype=np.float32)
    if mean.shape != (EXPECTED_INPUT_DIM,) or scale.shape != (EXPECTED_INPUT_DIM,):
        raise RuntimeError(
            f"scaler shape mismatch: mean={mean.shape}, scale={scale.shape}, expected ({EXPECTED_INPUT_DIM},)"
        )
    if not np.all(scale > 0):
        raise RuntimeError("scaler_scale must be strictly positive")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "scale": scale.tolist()}, f)


def _write_labels(name_to_id: dict[str, int], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(name_to_id, f, indent=2)


def _copy_mediapipe_task(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GestureMLP to ONNX for browser inference")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mediapipe-task", type=pathlib.Path, default=DEFAULT_MP_TASK)
    parser.add_argument("--mediapipe-out", type=pathlib.Path, default=DEFAULT_MP_OUT)
    args = parser.parse_args()

    ck = _load_checkpoint(args.checkpoint)
    name_to_id = _validate_labels(args.labels)
    model = _build_model(ck)

    onnx_path = args.out_dir / "gesture_mlp.onnx"
    scaler_path = args.out_dir / "scaler.json"
    labels_path = args.out_dir / "labels.json"

    _export_onnx(model, onnx_path)
    max_abs = _verify_parity(model, onnx_path)
    _write_scaler(ck, scaler_path)
    _write_labels(name_to_id, labels_path)

    mp_copied = _copy_mediapipe_task(args.mediapipe_task, args.mediapipe_out)

    onnx_size_kb = onnx_path.stat().st_size / 1024.0
    print(
        f"PASS export_onnx: num_classes={EXPECTED_NUM_CLASSES} input_dim={EXPECTED_INPUT_DIM} "
        f"max_abs_diff={max_abs:.3e} (atol={PARITY_ATOL:.0e}) "
        f"onnx={onnx_size_kb:.1f}KB scaler=ok labels=ok "
        f"mediapipe_task={'copied' if mp_copied else 'missing'} "
        f"out_dir={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
