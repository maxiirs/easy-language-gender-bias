#!/usr/bin/env python3
"""Filter parallel Easy Language dataset based on ELGEPA source.

Uses hard sanity exclusions (< 20 tokens) and robust length-ratio 
outlier detection (MAD / Z-Scores) to ensure semantic alignment 
without relying on computationally heavy semantic metrics.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Any

INPUT_FILE = Path("data/simple-german-articles.csv")
PROCESSED_DIR = Path("data/processed")
EXCLUDED_URL_TERMS = {"impressum", "datenschutz", "agb", "kontakt", "newsletter", "haftungsausschluss", "cookie", "login", "registrierung", "privacy", "index_de", "index.php"}

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]

def hard_rejection_reason(row: dict[str, str], simple_tokens: int, normal_tokens: int) -> str:
    simple_url = row.get("simple_url", "").lower()
    normal_url = row.get("normal_url", "").lower()
    if any(term in simple_url or term in normal_url for term in EXCLUDED_URL_TERMS):
        return "excluded_url_term"
    if simple_tokens < 20 or normal_tokens < 20:
        return "too_short"
    return ""

# Konstanten für die robuste Ausreißer-Erkennung (nach Iglewicz & Hoaglin, 1993)
# Der Faktor 0.6745 nähert den MAD an die Standardabweichung der Normalverteilung an.
# Dadurch werden die resultierenden Modified Z-Scores mit klassischen Z-Scores vergleichbar.
MAD_SCALING_FACTOR = 0.6745 
# Fallback-Score: Ist die Varianz des Datensatzes exakt 0, erhält jede Abweichung einen 
# künstlichen Extrem-Score (Z=10), um verlässlich durch den Threshold (3.5) zu fallen.
MAX_Z_SCORE = 10.0

def robust_z_score(val: float, values: list[float]) -> float:
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    if mad == 0:
        return 0.0 if val == med else MAX_Z_SCORE
    return MAD_SCALING_FACTOR * abs(val - med) / mad

def build_records() -> list[dict[str, Any]]:
    provisional_records = []

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            simple_text = row.get("simple_text", "").strip()
            normal_text = row.get("normal_text", "").strip()

            if not simple_text or not normal_text:
                continue

            source = row.get("source", "unknown")
            simple_tokens = tokenize(simple_text)
            normal_tokens = tokenize(normal_text)
            
            rejection = hard_rejection_reason(row, len(simple_tokens), len(normal_tokens))
            ratio = len(normal_tokens) / len(simple_tokens) if simple_tokens else math.inf
            log_ratio_dist = abs(math.log(ratio)) if ratio > 0 else math.inf

            record = dict(row)
            record.update({
                "source": source,
                "simple_char_len": len(simple_text),
                "normal_char_len": len(normal_text),
                "simple_token_len": len(simple_tokens),
                "normal_token_len": len(normal_tokens),
                "length_ratio": ratio,
                "log_length_ratio_distance": log_ratio_dist,
                "rejection_reason": rejection,
            })
            provisional_records.append(record)

    # Source-specific stats for length ratio
    stats_by_source = {}
    for source in sorted({r["source"] for r in provisional_records}):
        source_records = [r for r in provisional_records if r["source"] == source]
        ratio_values = [r["log_length_ratio_distance"] for r in source_records]
        stats_by_source[source] = ratio_values

    records = []
    for record in provisional_records:
        source_values = stats_by_source[record["source"]]
        
        # Calculate robust Z-Score on the fly if enough values are present
        if len(source_values) < 2:
            ratio_z = 0.0
        else:
            ratio_z = robust_z_score(record["log_length_ratio_distance"], source_values)

        # Evaluate quality (Z-Score >= 3.5 -> Extreme Outlier)
        if record["rejection_reason"] or ratio_z >= 3.5:
            label = "rejected"
        else:
            label = "accepted"

        record.update({
            "ratio_outlier_z": round(ratio_z, 4),
            "quality_label": label,
            "length_ratio": round(record["length_ratio"], 4),
            "log_length_ratio_distance": round(record["log_length_ratio_distance"], 4)
        })
        records.append(record)

    return records

def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    accepted_records = [r for r in records if r["quality_label"] == "accepted"]

    write_csv(PROCESSED_DIR / "paired_dataset_scored.csv", records)
    write_csv(PROCESSED_DIR / "paired_dataset_accepted.csv", accepted_records)

    summary = {
        "total_scored_pairs": len(records),
        "by_label": dict(Counter(r["quality_label"] for r in records)),
        "filter_spec": {
            "hard_exclusions": {
                "minimum_tokens_per_side": 20,
                "url_term_filtering": True
            },
            "ratio_outlier_detection": "Rejected if length_ratio modified Z-Score >= 3.5"
        }
    }
    
    summary_path = PROCESSED_DIR / "paired_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✅ Filterung abgeschlossen. {len(accepted_records)} von {len(records)} Paaren wurden als 'accepted' eingestuft.")

if __name__ == "__main__":
    main()
