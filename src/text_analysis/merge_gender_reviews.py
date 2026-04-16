#!/usr/bin/env python3
"""Merge manual review decisions into the automatic gender analysis output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_AUTO = Path("data/processed/gender_analysis_auto.csv")
DEFAULT_REVIEW = Path("data/annotations/gender_review.csv")
DEFAULT_OUTPUT = Path("data/processed/gender_analysis_final.csv")
DEFAULT_SUMMARY = Path("data/processed/gender_analysis_final_summary.json")

VALID_FRAMES = {
    "fem",
    "male_specific",
    "gmasc_generic",
    "epic_individual",
    "epic_group",
    "pair",
    "named_person",
    "unk",
}
VALID_T2I = {"f1", "m1", "fg", "mg", "mx", "ux"}


def map_frame_to_t2i(frame: str, is_group: bool) -> str:
    """Map fine-grained text frames to compact image-ready labels."""
    if frame == "fem":
        return "fg" if is_group else "f1"
    if frame == "male_specific":
        return "mg" if is_group else "m1"
    if frame in {"pair", "epic_group"}:
        return "mx"
    return "ux"


def load_csv_rows(path: Path) -> list[dict]:
    """Load a CSV file into a list of dict rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_override(value: str) -> str:
    """Normalize a manual override cell."""
    return value.strip()


def get_review_key(row: dict) -> tuple[str, str, str]:
    """Build the lemma-level review key shared by auto and review rows."""
    return (row.get("lemma", ""), row.get("auto_frame", ""), row.get("review_reason", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge manual review decisions into automatic gender analysis."
    )
    parser.add_argument("--auto", type=Path, default=DEFAULT_AUTO)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    auto_rows = load_csv_rows(args.auto)
    review_rows = load_csv_rows(args.review) if args.review.exists() else []
    review_index = {row["mention_id"]: row for row in review_rows if row.get("mention_id")}
    review_lemma_index = {
        get_review_key(row): row
        for row in review_rows
        if row.get("lemma") and row.get("auto_frame") and row.get("review_reason")
    }

    merged_rows: list[dict] = []
    applied_overrides = 0
    invalid_frame_rows = 0
    invalid_t2i_rows = 0

    for row in auto_rows:
        review = review_index.get(row["mention_id"], {})
        if not review:
            review = review_lemma_index.get(get_review_key(row), {})
        manual_frame = normalize_override(review.get("manual_frame", ""))
        manual_t2i = normalize_override(review.get("manual_t2i_label", ""))
        manual_note = normalize_override(review.get("manual_note", ""))

        final_frame = row["auto_frame"]
        if manual_frame:
            if manual_frame not in VALID_FRAMES:
                invalid_frame_rows += 1
            else:
                final_frame = manual_frame
                applied_overrides += 1

        is_group = row["is_group"] == "1"
        final_t2i = map_frame_to_t2i(final_frame, is_group)
        if manual_t2i:
            if manual_t2i not in VALID_T2I:
                invalid_t2i_rows += 1
            else:
                final_t2i = manual_t2i

        merged_rows.append(
            {
                **row,
                "manual_frame": manual_frame,
                "manual_t2i_label": manual_t2i,
                "manual_note": manual_note,
                "final_frame": final_frame,
                "final_t2i_label": final_t2i,
            }
        )

    fieldnames = list(merged_rows[0].keys()) if merged_rows else []
    write_csv(args.output, merged_rows, fieldnames)

    summary = {
        "total_mentions": len(merged_rows),
        "review_rows": len(review_rows),
        "applied_frame_overrides": applied_overrides,
        "invalid_frame_rows": invalid_frame_rows,
        "invalid_t2i_rows": invalid_t2i_rows,
        "final_frame_counts": dict(Counter(row["final_frame"] for row in merged_rows).most_common()),
        "final_t2i_counts": dict(Counter(row["final_t2i_label"] for row in merged_rows).most_common()),
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Auto rows:              {len(auto_rows)}")
    print(f"Review rows:            {len(review_rows)}")
    print(f"Applied frame overrides:{applied_overrides}")
    print(f"Invalid frame rows:     {invalid_frame_rows}")
    print(f"Invalid t2i rows:       {invalid_t2i_rows}")
    print(f"Final CSV:              {args.output}")
    print(f"Summary:                {args.summary}")


if __name__ == "__main__":
    main()
