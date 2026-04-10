#!/usr/bin/env python3
"""Annotate the filtered parallel corpus with LLpro and dump raw results.

Runs the LLpro pipeline (NER, Coreference, Character Recognition) on each
text and writes all annotations to a JSON file for downstream analysis.

Usage:
    python src/pipeline/annotate_texts.py --limit 5   # test run
    python src/pipeline/annotate_texts.py              # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import spacy
import llpro  # noqa: F401 – registers LLpro spaCy components

DEFAULT_INPUT = Path("data/processed/paired_dataset_accepted.csv")
DEFAULT_OUTPUT = Path("data/processed/annotated_texts.json")

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip invisible characters."""
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def load_pipeline(model: str = "de_core_news_sm") -> spacy.Language:
    """Build a spaCy pipeline with the three LLpro components."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] CUDA verfügbar ({torch.cuda.get_device_name(0)})")
        else:
            print("[CPU] Keine GPU gefunden. LLpro läuft auf CPU.")
    except ImportError:
        print("[CPU] PyTorch nicht gefunden.")

    nlp = spacy.load(model, disable=["ner"])
    nlp.add_pipe("ner_flair")
    nlp.add_pipe("coref_uhhlt")
    nlp.add_pipe("character_recognizer")
    return nlp


def serialize_doc(doc) -> dict:
    """Extract all LLpro annotations from a processed Doc into a plain dict."""
    result = {
        "text": doc.text,
        "token_count": len(doc),
        "sentences": [],
        "entities": [],
        "characters": [],
        "coref_clusters": [],
    }

    # Sentences
    for sent in doc.sents:
        result["sentences"].append({
            "start": sent.start,
            "end": sent.end,
            "text": sent.text.strip(),
        })

    # Named Entities (from ner_flair)
    for ent in doc.ents:
        result["entities"].append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })

    # Characters (from character_recognizer)
    if hasattr(doc._, "characters") and doc._.characters:
        for span in doc._.characters:
            # Extract morphological gender from head token of the span
            tokens_info = []
            for tok in span:
                morph_str = str(tok.morph) if tok.morph else ""
                tokens_info.append({
                    "text": tok.text,
                    "pos": tok.pos_,
                    "tag": tok.tag_,
                    "morph": morph_str,
                    "lemma": tok.lemma_,
                })

            # Try to get gender from morphological features of the span's root
            morph_gender = None
            for tok in span:
                gender = tok.morph.get("Gender")
                if gender:
                    morph_gender = gender[0] if isinstance(gender, list) else gender
                    break

            result["characters"].append({
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "sent_text": span.sent.text.strip() if span.sent else "",
                "morph_gender": morph_gender,
                "tokens": tokens_info,
            })

    # Coreference Clusters (from coref_uhhlt)
    if hasattr(doc._, "coref_clusters") and doc._.coref_clusters:
        for cluster_idx, cluster in enumerate(doc._.coref_clusters):
            mentions = []
            for span in cluster:
                # Store token-level morph data so downstream code can infer
                # pronoun-based gender from closed grammatical features.
                morph_gender = None
                morph_number = None
                tokens_info = []
                for tok in span:
                    morph_str = str(tok.morph) if tok.morph else ""
                    tokens_info.append({
                        "text": tok.text,
                        "pos": tok.pos_,
                        "tag": tok.tag_,
                        "morph": morph_str,
                        "lemma": tok.lemma_,
                    })
                    gender = tok.morph.get("Gender")
                    if gender:
                        morph_gender = gender[0] if isinstance(gender, list) else gender
                    number = tok.morph.get("Number")
                    if number:
                        morph_number = number[0] if isinstance(number, list) else number
                mentions.append({
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "morph_gender": morph_gender,
                    "morph_number": morph_number,
                    "tokens": tokens_info,
                })
            cluster_id = cluster.attrs.get("id", cluster_idx) if hasattr(cluster, "attrs") else cluster_idx
            result["coref_clusters"].append({
                "cluster_id": cluster_id,
                "mentions": mentions,
            })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate parallel corpus with LLpro.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="de_core_news_sm")
    parser.add_argument("--limit", type=int, default=0, help="Limit pairs (0 = all).")
    args = parser.parse_args()

    # 1. Load data
    print(f"\n--- Lade Daten: {args.input} ---")
    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[OK] {len(rows)} Paare geladen.")

    # 2. Load pipeline
    print("\n--- Lade LLpro Pipeline ---")
    nlp = load_pipeline(args.model)

    # 3. Process each pair
    results = []
    total_docs = 0

    for idx, row in enumerate(rows, start=1):
        pair_id = f"pair_{idx:04d}"
        pair_result = {
            "pair_id": pair_id,
            "source": row.get("source", ""),
        }

        for side in ("simple", "normal"):
            text = row.get(f"{side}_text", "")
            if not text or not text.strip():
                pair_result[side] = None
                continue

            text = normalize_text(text)
            print(f"  [{pair_id}/{side}] ({len(text)} chars) ...", end=" ", flush=True)

            try:
                doc = nlp(text)
                pair_result[side] = serialize_doc(doc)
                n_chars = len(pair_result[side]["characters"])
                n_coref = len(pair_result[side]["coref_clusters"])
                n_ents = len(pair_result[side]["entities"])
                print(f"OK ({n_ents} entities, {n_chars} characters, {n_coref} coref clusters)")
            except Exception as e:
                print(f"FEHLER: {e}", file=sys.stderr)
                pair_result[side] = {"error": str(e)}

            total_docs += 1

        results.append(pair_result)

    # 4. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n--- Fertig ---")
    print(f"Paare verarbeitet: {len(results)}")
    print(f"Dokumente annotiert: {total_docs}")
    print(f"Ausgabe: {args.output}")


if __name__ == "__main__":
    main()
