#!/usr/bin/env python3
"""Extract actors from the filtered parallel Easy Language dataset using LLpro.

Uses LLpro's FLERT-based NER, neural coreference resolution, and 
character recognizer to identify human actors, resolve pronoun references,
and infer grammatical gender for each actor mention cluster.

Requires: pip install llpro (with its dependencies)
          python -m spacy download de_core_news_sm

Output: A CSV with one row per actor mention cluster per text side.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import spacy

# LLpro components are registered as spaCy pipeline factories on import
import llpro  # noqa: F401

DEFAULT_INPUT = Path("data/processed/paired_dataset_accepted.csv")
DEFAULT_OUTPUT = Path("data/processed/actor_features.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/actor_features_summary.json")

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip invisible characters."""
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def load_pipeline(model: str = "de_core_news_sm") -> spacy.Language:
    """Build a spaCy pipeline with the three LLpro components we need.
    
    GPU/CUDA is handled internally by LLpro's components via PyTorch,
    not via spaCy's CuPy-based GPU support.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] CUDA verfügbar ({torch.cuda.get_device_name(0)}). LLpro nutzt GPU automatisch.")
        else:
            print("[CPU] Keine GPU gefunden. LLpro läuft auf CPU.")
    except ImportError:
        print("[CPU] PyTorch nicht gefunden, LLpro läuft auf CPU.")

    nlp = spacy.load(model, disable=["ner"])

    # LLpro-Komponenten (kein ParZu → kein SWI-Prolog/SFST nötig)
    nlp.add_pipe("ner_flair")              # Named Entity Recognition (FLERT)
    nlp.add_pipe("coref_uhhlt")            # Neural Coreference Resolution
    nlp.add_pipe("character_recognizer")   # Character/Actor detection + Gender

    return nlp


def extract_actors_from_doc(doc) -> list[dict[str, Any]]:
    """Extract actor-level data from a processed spaCy Doc with LLpro extensions.
    
    LLpro's character_recognizer stores detected character mentions as Span objects
    in doc._.characters. We cluster them by surface form and cross-reference with
    coreference clusters (doc._.coref_clusters) to infer gender via linked pronouns.
    """
    actors = []

    if not hasattr(doc._, "characters") or not doc._.characters:
        return actors

    # 1. Group character mentions by normalized surface text
    from collections import defaultdict
    clusters: dict[str, dict[str, Any]] = {}
    for span in doc._.characters:
        label = span.text.strip()
        if not label:
            continue
        key = label.lower()
        if key not in clusters:
            clusters[key] = {
                "label": label,
                "mention_spans": [],
                "sent_indices": set(),
            }
        clusters[key]["mention_spans"].append(span)
        clusters[key]["sent_indices"].add(span.sent.start)

    # 2. Infer gender via coreference clusters (if available)
    FEMININE_PRONOUNS = {"sie", "ihr", "ihre", "ihren", "ihrem", "ihrer", "ihres"}
    MASCULINE_PRONOUNS = {"er", "ihn", "ihm", "sein", "seine", "seinen", "seinem", "seiner", "seines"}

    coref_clusters = []
    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
        coref_clusters = doc._.coref_clusters

    for key, cluster in clusters.items():
        # Try to find coref cluster that overlaps with character mentions
        gender = "unknown"
        masc_count = 0
        fem_count = 0

        for coref_cluster in coref_clusters:
            # Check if any mention in this coref cluster overlaps with our character
            has_character_overlap = False
            for coref_span in coref_cluster:
                for char_span in cluster["mention_spans"]:
                    if (coref_span.start < char_span.end and coref_span.end > char_span.start):
                        has_character_overlap = True
                        break
                if has_character_overlap:
                    break

            if has_character_overlap:
                # Count pronoun genders in this coref cluster
                for coref_span in coref_cluster:
                    span_text = coref_span.text.strip().lower()
                    if span_text in FEMININE_PRONOUNS:
                        fem_count += 1
                    elif span_text in MASCULINE_PRONOUNS:
                        masc_count += 1

        if masc_count > 0 or fem_count > 0:
            if masc_count > fem_count:
                gender = "masculine"
            elif fem_count > masc_count:
                gender = "feminine"
            else:
                gender = "mixed_or_unclear"

        # 3. Extract predication sentences
        predications = []
        seen_sents = set()
        for sent in doc.sents:
            if sent.start in cluster["sent_indices"]:
                sent_text = sent.text.strip()
                if sent_text and sent_text not in seen_sents:
                    seen_sents.add(sent_text)
                    predications.append(sent_text)

        actors.append({
            "actor_label": cluster["label"],
            "gender": gender,
            "mention_count": len(cluster["mention_spans"]),
            "mentions": " | ".join(sorted({s.text.strip() for s in cluster["mention_spans"]})),
            "predication_count": len(predications),
            "predications": " || ".join(predications),
        })

    return actors


def build_actor_rows(
    pair_id: str,
    side: str,
    row: dict[str, str],
    doc,
) -> list[dict[str, Any]]:
    """Build output rows for all actors found in a single document."""
    actors = extract_actors_from_doc(doc)
    output_rows = []

    for index, actor in enumerate(actors, start=1):
        output_rows.append({
            "pair_id": pair_id,
            "side": side,
            "source": row.get("source", ""),
            "actor_id": f"{pair_id}_{side}_{index:03d}",
            "actor_label": actor["actor_label"],
            "gender": actor["gender"],
            "mention_count": actor["mention_count"],
            "mentions": actor["mentions"],
            "predication_count": actor["predication_count"],
            "predications": actor["predications"],
        })

    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        print(f"[WARNUNG] Keine Zeilen zum Schreiben für {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract actors from parallel corpus using LLpro.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model", default="de_core_news_sm")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs to process (0 = all).")
    args = parser.parse_args()

    # 1. Load data
    print(f"\n--- Lade Daten: {args.input} ---")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[OK] {len(rows)} Paare geladen.")

    # 2. Load LLpro pipeline
    print("\n--- Lade LLpro Pipeline ---")
    nlp = load_pipeline(args.model)

    # 3. Prepare text inputs for batch processing via nlp.pipe()
    doc_inputs: list[tuple[str, str, dict[str, str]]] = []
    for index, row in enumerate(rows, start=1):
        pair_id = f"pair_{index:04d}"
        for side in ("simple", "normal"):
            text = row.get(f"{side}_text", "")
            if text and text.strip():
                doc_inputs.append((pair_id, side, row, normalize_text(text)))

    print(f"\n--- Starte Extraktion ({len(doc_inputs)} Dokumente) ---")

    # 4. Process with nlp.pipe for efficiency
    actor_rows: list[dict[str, Any]] = []
    texts = [item[3] for item in doc_inputs]
    for meta, doc in zip(doc_inputs, nlp.pipe(texts, batch_size=8)):
        pair_id, side, row, _text = meta
        try:
            actor_rows.extend(build_actor_rows(pair_id, side, row, doc))
        except Exception as e:
            print(f"[FEHLER] {pair_id}/{side}: {e}", file=sys.stderr)

    # 5. Write output
    count = write_csv(args.output, actor_rows)
    print(f"\n[OK] {count} Akteur-Zeilen geschrieben nach {args.output}")

    # 6. Write summary
    from collections import Counter
    summary = {
        "total_actor_rows": len(actor_rows),
        "pairs_processed": len(rows),
        "by_side": dict(Counter(r["side"] for r in actor_rows)),
        "by_source": dict(Counter(r["source"] for r in actor_rows)),
        "by_gender": dict(Counter(r["gender"] for r in actor_rows)),
        "pipeline": "LLpro (ner_flair + coref_uhhlt + character_recognizer)",
        "spacy_model": args.model,
    }
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()