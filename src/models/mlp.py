"""MLP classifier: R^279 -> R^26 with BatchNorm, ReLU, Dropout, Kaiming init.

Stage 4 main model. Pure ``nn.Module`` definition — no IO, no training, no
device handling. The training loop lives in ``src/train.py``.

Architecture (gesture_recognition_plan_v2.md §6.2, stage4_handoff.md §5):

    Linear(279, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
    Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
    Linear(128,  64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
    Linear( 64,  26)                                            # logits

``forward`` returns logits. ``nn.CrossEntropyLoss`` consumes them directly
(numerically stable); Stage 5 inference applies ``F.softmax`` at the boundary.

The depth (3 hidden layers) and activation choice (ReLU) are fixed by Stage 6's
chain-rule visualization plan — see stage4_handoff.md §7.6.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 279
NUM_CLASSES = 26
HIDDEN_DIMS: tuple[int, int, int] = (256, 128, 64)
DROPOUTS: tuple[float, float, float] = (0.3, 0.3, 0.2)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GestureMLP(nn.Module):
    """3-hidden-layer MLP: ``R^input_dim`` -> ``R^num_classes`` logits.

    Forward order::

        Linear(input_dim, hidden_dims[0]) -> BN -> ReLU -> Dropout(dropouts[0])
        Linear(hidden_dims[0], hidden_dims[1]) -> BN -> ReLU -> Dropout(dropouts[1])
        Linear(hidden_dims[1], hidden_dims[2]) -> BN -> ReLU -> Dropout(dropouts[2])
        Linear(hidden_dims[2], num_classes)                                  # logits

    Defaults match plan §6.2 exactly: 279 -> 256 -> 128 -> 64 -> 26 with
    dropout probabilities ``(0.3, 0.3, 0.2)``.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
        dropouts: tuple[float, ...] = DROPOUTS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError(
                f"GestureMLP requires exactly 3 hidden layers (Stage 6 chain-rule "
                f"trace depends on this); got hidden_dims={hidden_dims!r}"
            )
        if len(dropouts) != 3:
            raise ValueError(
                f"GestureMLP requires exactly 3 dropout probabilities; "
                f"got dropouts={dropouts!r}"
            )
        for p in dropouts:
            if not (0.0 <= p < 1.0):
                raise ValueError(
                    f"dropout probabilities must be in [0, 1); got {p}"
                )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropouts_p = tuple(float(p) for p in dropouts)
        self.num_classes = int(num_classes)

        dims = (self.input_dim, *self.hidden_dims, self.num_classes)
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(4)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(h) for h in self.hidden_dims]
        )
        self.drops = nn.ModuleList(
            [nn.Dropout(p) for p in self.dropouts_p]
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, num_classes)``. No softmax."""
        for i in range(3):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.drops[i](x)
        return self.linears[3](x)

    def initialize_weights(self) -> None:
        """Kaiming-uniform init on all linear weights; zero biases.

        ReLU-compatible (plan §6.2: ``init = "kaiming_uniform"``). BatchNorm
        layers keep PyTorch defaults (gamma=1, beta=0, running_mean=0,
        running_var=1).
        """
        for layer in self.linears:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    @property
    def linear_layers(self) -> list[nn.Linear]:
        """The four ``nn.Linear`` modules in forward order.

        ``linear_layers[0]`` is ``Linear(279, 256)``; ``linear_layers[3]`` is
        ``Linear(64, 26)`` (output). Used by ``src/train.py`` to compute
        per-layer Frobenius weight norms for the training log, and by
        Stage 6's manual chain-rule trace.
        """
        return list(self.linears)


# ---------------------------------------------------------------------------
# Label schema guard
# ---------------------------------------------------------------------------


def assert_labels_consistent(
    labels_json_path: Path = Path("data/labels.json"),
    expected_num_classes: int = NUM_CLASSES,
) -> dict[str, int]:
    """Validate that ``data/labels.json`` matches ``NUM_CLASSES``.

    Returns the loaded ``{name: id}`` mapping. Raises ``ValueError`` if the
    file is missing, has the wrong number of classes, or has non-dense ids.
    """
    p = Path(labels_json_path)
    if not p.is_file():
        raise ValueError(f"labels.json not found at {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"labels.json must be a non-empty object; got {type(raw).__name__}")
    name_to_id = {str(k): int(v) for k, v in raw.items()}
    if len(name_to_id) != expected_num_classes:
        raise ValueError(
            f"labels.json has {len(name_to_id)} entries but NUM_CLASSES={expected_num_classes}. "
            f"Edit data/labels.json or src/models/mlp.py::NUM_CLASSES so they agree."
        )
    ids = set(name_to_id.values())
    if ids != set(range(expected_num_classes)):
        raise ValueError(
            f"labels.json ids must be a dense range 0..{expected_num_classes - 1}; got {sorted(ids)}"
        )
    return name_to_id
