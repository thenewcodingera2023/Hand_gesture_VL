"""Guard tests for the 26-class label schema.

Locks in the structural fix from `tasks/peace_count2_collision_fix.md`:
`count_2` and `count_5` were duplicate labels for `peace` and `open_palm`
respectively. Re-introducing either as a separate id is a regression these
tests catch immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.mlp import NUM_CLASSES, GestureMLP, assert_labels_consistent

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO_ROOT / "data" / "labels.json"


def _load_labels() -> dict[str, int]:
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_labels_json_length_matches_num_classes():
    labels = _load_labels()
    assert len(labels) == NUM_CLASSES, (
        f"data/labels.json has {len(labels)} entries but "
        f"src.models.mlp.NUM_CLASSES={NUM_CLASSES}"
    )


def test_labels_json_ids_are_dense():
    labels = _load_labels()
    assert set(labels.values()) == set(range(NUM_CLASSES))


def test_count_2_and_count_5_are_dropped():
    labels = _load_labels()
    assert "count_2" not in labels, (
        "count_2 is structurally identical to peace — keep peace only"
    )
    assert "count_5" not in labels, (
        "count_5 is structurally identical to open_palm — keep open_palm only"
    )


def test_peace_and_open_palm_present():
    labels = _load_labels()
    assert "peace" in labels
    assert "open_palm" in labels


def test_gesture_mlp_output_dim_matches_num_classes():
    model = GestureMLP()
    assert model.num_classes == NUM_CLASSES
    assert model.linear_layers[-1].out_features == NUM_CLASSES


def test_assert_labels_consistent_passes():
    name_to_id = assert_labels_consistent(LABELS_PATH)
    assert len(name_to_id) == NUM_CLASSES


def test_assert_labels_consistent_rejects_wrong_count(tmp_path):
    """If labels.json is edited to a wrong length, the guard fires before any
    expensive training/eval work runs."""
    bad = tmp_path / "labels_bad.json"
    bad.write_text(
        json.dumps({"a": 0, "b": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NUM_CLASSES"):
        assert_labels_consistent(bad, expected_num_classes=NUM_CLASSES)


def test_assert_labels_consistent_rejects_non_dense_ids(tmp_path):
    bad = tmp_path / "labels_sparse.json"
    sparse = {f"c{i}": i for i in range(NUM_CLASSES)}
    sparse[f"c{NUM_CLASSES - 1}"] = NUM_CLASSES + 5  # gap
    bad.write_text(json.dumps(sparse), encoding="utf-8")
    with pytest.raises(ValueError, match="dense range"):
        assert_labels_consistent(bad, expected_num_classes=NUM_CLASSES)
