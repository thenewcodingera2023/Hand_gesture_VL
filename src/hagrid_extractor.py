"""Parse HaGRIDv2 annotation JSONs into raw landmark sample records.

Outputs `data/processed/hagrid_raw_records.npz` with parallel arrays so that
downstream stages reload in O(read) instead of re-parsing JSON. See
`tasks/gesture_recognition_plan_v2.md` §2.1 and the Stage 1 plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

HAGRID_RELEVANT = (
    "like", "dislike", "fist", "four", "ok", "one", "palm", "peace",
    "call", "mute", "rock", "stop", "three", "two_up",
)

# Source label -> tuple of project class names this sample contributes to.
# The 28-class schema previously emitted two project rows per peace/two_up/palm
# image (peace+count_2, palm+open_palm+count_5), causing the model to split its
# probability across visually-identical labels — see
# tasks/peace_count2_collision_fix.md. Each HaGRID image now emits exactly one
# project row. `peace`/`two_up` both map to `peace` (same hand shape).
LABEL_MAP: dict[str, tuple[str, ...]] = {
    "like":    ("thumbs_up",),
    "dislike": ("thumbs_down",),
    "stop":    ("stop",),
    "ok":      ("ok",),
    "call":    ("call",),
    "rock":    ("rock",),
    "mute":    ("mute",),
    "fist":    ("fist",),
    "peace":   ("peace",),
    "two_up":  ("peace",),
    "palm":    ("open_palm",),
    "one":     ("count_1",),
    "three":   ("count_3",),
    "four":    ("count_4",),
}

SPLITS = ("train", "val", "test")


def iter_annotation_file(json_path: Path, hagrid_split: str) -> Iterator[dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: failed to read {json_path}: {exc}", file=sys.stderr)
        return

    file_name = json_path.name
    for image_id, meta in data.items():
        labels = meta.get("labels", [])
        hand_landmarks = meta.get("hand_landmarks", [])
        user_id = meta.get("user_id", "")
        if not user_id:
            continue
        for i, src_label in enumerate(labels):
            if src_label == "no_gesture":
                continue
            if src_label not in LABEL_MAP:
                continue
            if i >= len(hand_landmarks):
                continue
            try:
                lm = np.asarray(hand_landmarks[i], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if lm.shape != (21, 2):
                continue
            if not np.isfinite(lm).all():
                continue
            lm3 = np.hstack([lm, np.zeros((21, 1), dtype=np.float32)])
            for project_label in LABEL_MAP[src_label]:
                yield {
                    "image_id": image_id,
                    "source_label": src_label,
                    "project_label": project_label,
                    "landmarks_raw": lm3,
                    "hand": "Right",
                    "user_id": user_id,
                    "source": "hagrid_v2",
                    "hagrid_split": hagrid_split,
                    "source_file": file_name,
                    "detection_index": i,
                }


def extract_split(annotations_root: Path, split: str) -> list[dict]:
    split_dir = annotations_root / split
    if not split_dir.is_dir():
        print(f"WARN: split directory missing: {split_dir}", file=sys.stderr)
        return []
    records: list[dict] = []
    for src_label in HAGRID_RELEVANT:
        json_path = split_dir / f"{src_label}.json"
        if not json_path.is_file():
            print(f"WARN: missing annotation file: {json_path}", file=sys.stderr)
            continue
        before = len(records)
        for rec in iter_annotation_file(json_path, split):
            records.append(rec)
        added = len(records) - before
        print(f"  [{split}] {src_label}: +{added} records (total {len(records)})")
    return records


def extract_all(annotations_root: Path) -> list[dict]:
    records: list[dict] = []
    for split in SPLITS:
        print(f"==> Extracting split: {split}")
        records.extend(extract_split(annotations_root, split))
    return records


def save_raw_records(records: list[dict], out_path: Path) -> None:
    if not records:
        raise RuntimeError("no records to save")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    landmarks_raw = np.empty((n, 21, 3), dtype=np.float32)
    project_label = np.empty(n, dtype=object)
    source_label = np.empty(n, dtype=object)
    hand = np.empty(n, dtype=object)
    user_id = np.empty(n, dtype=object)
    source = np.empty(n, dtype=object)
    hagrid_split = np.empty(n, dtype=object)
    image_id = np.empty(n, dtype=object)
    source_file = np.empty(n, dtype=object)
    detection_index = np.empty(n, dtype=np.int32)

    for i, rec in enumerate(records):
        landmarks_raw[i] = rec["landmarks_raw"]
        project_label[i] = rec["project_label"]
        source_label[i] = rec["source_label"]
        hand[i] = rec["hand"]
        user_id[i] = rec["user_id"]
        source[i] = rec["source"]
        hagrid_split[i] = rec["hagrid_split"]
        image_id[i] = rec["image_id"]
        source_file[i] = rec["source_file"]
        detection_index[i] = rec["detection_index"]

    np.savez_compressed(
        out_path,
        landmarks_raw=landmarks_raw,
        project_label=project_label,
        source_label=source_label,
        hand=hand,
        user_id=user_id,
        source=source,
        hagrid_split=hagrid_split,
        image_id=image_id,
        source_file=source_file,
        detection_index=detection_index,
    )
    print(f"==> Wrote {n} records to {out_path}")


def load_raw_records(in_path: Path) -> dict:
    arr = np.load(in_path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract HaGRIDv2 hand landmarks from annotation JSONs.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("data/hagrid_raw/annotations/annotations"),
        help="Root directory containing {train,val,test}/<class>.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/hagrid_raw_records.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=200_000,
        help="Acceptance-gate floor; raises if total record count is below this.",
    )
    args = parser.parse_args(argv)

    records = extract_all(args.annotations_root)

    counts = Counter(r["project_label"] for r in records)
    print("\n==> Per-project-label counts:")
    for label, count in sorted(counts.items()):
        print(f"    {label:14s} {count:>8d}")
    print(f"==> TOTAL: {len(records)} records")

    if len(records) < args.min_records:
        raise SystemExit(
            f"Acceptance gate failed: {len(records)} < {args.min_records}"
        )

    save_raw_records(records, args.out)


if __name__ == "__main__":
    main()
