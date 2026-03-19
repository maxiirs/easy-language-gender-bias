#!/usr/bin/env python3
"""Build a cleaned easy-language/standard-language paired dataset.

The script implements a conservative, reproducible pair-quality pipeline:
1. Load crawler outputs from ``data/raw``.
2. Normalize texts and compute interpretable similarity features.
3. Remove trivial cases (e.g. extremely short texts).
4. Detect outliers with robust statistics in a multi-feature space.
5. Export accepted, review, and rejected pairs plus a summary report.

The outlier detection is intentionally based on transparent robust measures
(median and MAD) instead of opaque thresholds from a single feature.
This makes the method easier to justify in a scientific workflow.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_FILES = [
    "dresden_data.json",
    "hannover_data.json",
    "hessen_data.json",
    "muenchen_data.json",
    "hamburg_data.json",
]

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
WHITESPACE_RE = re.compile(r"\s+")

GERMAN_STOPWORDS = {
    "aber",
    "alle",
    "als",
    "also",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "dieser",
    "doch",
    "dort",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "er",
    "es",
    "für",
    "hat",
    "haben",
    "hier",
    "ich",
    "im",
    "in",
    "ist",
    "man",
    "mit",
    "nach",
    "nicht",
    "noch",
    "nur",
    "oder",
    "sein",
    "sich",
    "sie",
    "sind",
    "so",
    "und",
    "von",
    "vom",
    "vor",
    "war",
    "werden",
    "wie",
    "wird",
    "wir",
    "zu",
    "zum",
    "zur",
}


@dataclass
class PairRecord:
    source: str
    simple_url: str
    normal_url: str
    simple_text: str
    normal_text: str
    simple_char_len: int
    normal_char_len: int
    simple_token_len: int
    normal_token_len: int
    length_ratio: float
    lexical_jaccard: float
    overlap_coefficient: float
    shared_content_terms: int
    log_length_ratio_distance: float
    ratio_outlier_z: float
    overlap_outlier_z: float
    jaccard_outlier_z: float
    outlier_score: float
    quality_label: str
    rejection_reason: str


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def content_terms(tokens: Iterable[str]) -> set[str]:
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in GERMAN_STOPWORDS
    }


def overlap_metrics(simple_text: str, normal_text: str) -> tuple[float, float, int]:
    simple_tokens = tokenize(simple_text)
    normal_tokens = tokenize(normal_text)

    simple_content = content_terms(simple_tokens)
    normal_content = content_terms(normal_tokens)

    if not simple_content or not normal_content:
        return 0.0, 0.0, 0

    shared = simple_content & normal_content
    union = simple_content | normal_content
    jaccard = len(shared) / len(union)
    overlap = len(shared) / min(len(simple_content), len(normal_content))
    return jaccard, overlap, len(shared)


def hard_rejection_reason(simple_tokens: int, normal_tokens: int) -> str:
    if simple_tokens < 20 or normal_tokens < 20:
        return "too_short"
    return ""


def median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def mad(values: Iterable[float], center: float) -> float:
    deviations = [abs(value - center) for value in values]
    return median(deviations) or 0.0


def modified_z_score(value: float, center: float, scale_mad: float) -> float:
    if scale_mad == 0:
        return 0.0 if value == center else 10.0
    return 0.6745 * (value - center) / scale_mad


def abs_modified_z_score(value: float, center: float, scale_mad: float) -> float:
    return abs(modified_z_score(value, center, scale_mad))


def quality_label(rejection_reason: str, outlier_score: float, severe_flags: int) -> str:
    if rejection_reason:
        return "rejected"
    if severe_flags >= 2 or outlier_score >= 4.0:
        return "rejected"
    if severe_flags == 1 or outlier_score >= 2.5:
        return "review"
    return "accepted"


def iter_raw_pairs() -> Iterable[tuple[str, dict]]:
    for file_name in RAW_FILES:
        path = RAW_DIR / file_name
        if not path.exists():
            continue
        source = file_name.replace("_data.json", "")
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            yield source, item


def build_records() -> list[PairRecord]:
    provisional_records: list[dict] = []

    for source, item in iter_raw_pairs():
        simple_text = normalize_text(item.get("simple_text") or "")
        normal_text = normalize_text(item.get("normal_text") or "")

        if not simple_text or not normal_text:
            continue

        simple_tokens = tokenize(simple_text)
        normal_tokens = tokenize(normal_text)
        ratio = len(normal_tokens) / len(simple_tokens) if simple_tokens else math.inf
        jaccard, overlap, shared_terms = overlap_metrics(simple_text, normal_text)
        provisional_records.append(
            {
                "source": source,
                "simple_url": item.get("simple_url") or "",
                "normal_url": item.get("normal_url") or "",
                "simple_text": simple_text,
                "normal_text": normal_text,
                "simple_char_len": len(simple_text),
                "normal_char_len": len(normal_text),
                "simple_token_len": len(simple_tokens),
                "normal_token_len": len(normal_tokens),
                "length_ratio": ratio,
                "lexical_jaccard": jaccard,
                "overlap_coefficient": overlap,
                "shared_content_terms": shared_terms,
                "log_length_ratio_distance": abs(math.log(ratio)) if ratio > 0 else math.inf,
                "rejection_reason": hard_rejection_reason(
                    len(simple_tokens), len(normal_tokens)
                ),
            }
        )

    # Robust source-specific outlier calibration.
    stats_by_source: dict[str, dict[str, float]] = {}
    for source in sorted({record["source"] for record in provisional_records}):
        source_records = [
            record for record in provisional_records if record["source"] == source
        ]
        ratio_values = [record["log_length_ratio_distance"] for record in source_records]
        overlap_values = [record["overlap_coefficient"] for record in source_records]
        jaccard_values = [record["lexical_jaccard"] for record in source_records]

        ratio_med = median(ratio_values) or 0.0
        overlap_med = median(overlap_values) or 0.0
        jaccard_med = median(jaccard_values) or 0.0

        stats_by_source[source] = {
            "ratio_median": ratio_med,
            "ratio_mad": mad(ratio_values, ratio_med),
            "overlap_median": overlap_med,
            "overlap_mad": mad(overlap_values, overlap_med),
            "jaccard_median": jaccard_med,
            "jaccard_mad": mad(jaccard_values, jaccard_med),
        }

    records: list[PairRecord] = []
    for record in provisional_records:
        source_stats = stats_by_source[record["source"]]
        ratio_z = abs_modified_z_score(
            record["log_length_ratio_distance"],
            source_stats["ratio_median"],
            source_stats["ratio_mad"],
        )
        overlap_z = abs_modified_z_score(
            record["overlap_coefficient"],
            source_stats["overlap_median"],
            source_stats["overlap_mad"],
        )
        jaccard_z = abs_modified_z_score(
            record["lexical_jaccard"],
            source_stats["jaccard_median"],
            source_stats["jaccard_mad"],
        )

        # Overly low semantic overlap is more suspicious than overly high overlap.
        if record["overlap_coefficient"] >= source_stats["overlap_median"]:
            overlap_z *= 0.25
        if record["lexical_jaccard"] >= source_stats["jaccard_median"]:
            jaccard_z *= 0.25

        outlier_score = (ratio_z + overlap_z + jaccard_z) / 3
        severe_flags = sum(value >= 3.5 for value in (ratio_z, overlap_z, jaccard_z))
        label = quality_label(record["rejection_reason"], outlier_score, severe_flags)

        records.append(
            PairRecord(
                source=record["source"],
                simple_url=record["simple_url"],
                normal_url=record["normal_url"],
                simple_text=record["simple_text"],
                normal_text=record["normal_text"],
                simple_char_len=record["simple_char_len"],
                normal_char_len=record["normal_char_len"],
                simple_token_len=record["simple_token_len"],
                normal_token_len=record["normal_token_len"],
                length_ratio=round(record["length_ratio"], 4),
                lexical_jaccard=round(record["lexical_jaccard"], 4),
                overlap_coefficient=round(record["overlap_coefficient"], 4),
                shared_content_terms=record["shared_content_terms"],
                log_length_ratio_distance=round(record["log_length_ratio_distance"], 4),
                ratio_outlier_z=round(ratio_z, 4),
                overlap_outlier_z=round(overlap_z, 4),
                jaccard_outlier_z=round(jaccard_z, 4),
                outlier_score=round(outlier_score, 4),
                quality_label=label,
                rejection_reason=record["rejection_reason"],
            )
        )

    return records


def write_csv(path: Path, records: Iterable[PairRecord]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(records: list[PairRecord]) -> dict:
    summary: dict[str, object] = {
        "total_scored_pairs": len(records),
        "by_label": Counter(record.quality_label for record in records),
        "by_source": {},
        "filter_spec": {
            "hard_exclusions": {
                "minimum_tokens_per_side": 20
            },
            "robust_outlier_model": {
                "features": [
                    "abs(log(length_ratio))",
                    "overlap_coefficient",
                    "lexical_jaccard"
                ],
                "estimator": "source-specific modified z-scores using median and MAD",
                "accepted_rule": "no hard exclusion and no severe outlier flag and mean outlier score < 2.5",
                "review_rule": "no hard exclusion and exactly one severe flag or mean outlier score in [2.5, 4.0)",
                "rejected_rule": "hard exclusion or at least two severe flags or mean outlier score >= 4.0",
                "severe_flag_threshold": 3.5
            },
        },
    }

    by_source: dict[str, dict[str, object]] = {}
    for source in sorted({record.source for record in records}):
        source_records = [record for record in records if record.source == source]
        by_source[source] = {
            "total_scored_pairs": len(source_records),
            "accepted": sum(record.quality_label == "accepted" for record in source_records),
            "review": sum(record.quality_label == "review" for record in source_records),
            "rejected": sum(record.quality_label == "rejected" for record in source_records),
            "median_length_ratio": median(record.length_ratio for record in source_records),
            "median_overlap_coefficient": median(
                record.overlap_coefficient for record in source_records
            ),
            "median_lexical_jaccard": median(
                record.lexical_jaccard for record in source_records
            ),
            "median_outlier_score": median(
                record.outlier_score for record in source_records
            ),
        }

    summary["by_source"] = by_source
    summary["by_label"] = dict(summary["by_label"])
    return summary
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    accepted_records = [record for record in records if record.quality_label == "accepted"]
    review_records = [record for record in records if record.quality_label == "review"]

    write_csv(PROCESSED_DIR / "paired_dataset_scored.csv", records)
    write_csv(PROCESSED_DIR / "paired_dataset_accepted.csv", accepted_records)
    write_csv(PROCESSED_DIR / "paired_dataset_review.csv", review_records)

    summary = build_summary(records)
    (PROCESSED_DIR / "paired_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
