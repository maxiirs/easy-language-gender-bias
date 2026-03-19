#!/usr/bin/env python3
"""Build a spaCy-based working dataset for gender-bias analysis.

The script reads the paired corpus from ``data/prepared/simple-german-articles.csv``,
normalizes the texts, runs German spaCy preprocessing, and exports document-level
and pair-level features that can serve as the base for later bias analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import spacy


DEFAULT_INPUT = Path("data/prepared/simple-german-articles.csv")
DEFAULT_DOC_OUTPUT = Path("data/processed/bias_doc_features.csv")
DEFAULT_PAIR_OUTPUT = Path("data/processed/bias_pair_features.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/processed/bias_dataset_summary.json")

WHITESPACE_RE = re.compile(r"\s+")
INCLUSIVE_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]+([*:/_]|innen$|Innen$)")

MASCULINE_PRONOUNS = {
    "er",
    "ihn",
    "ihm",
    "sein",
    "seine",
    "seiner",
    "seinem",
    "seinen",
}
FEMININE_PRONOUNS = {
    "sie",
    "ihr",
    "ihre",
    "ihrer",
    "ihrem",
    "ihren",
}
NEUTRAL_PRONOUNS = {
    "person",
    "menschen",
    "mensch",
    "jemand",
    "niemand",
    "alle",
}
MASCULINE_NOUNS = {
    "mann",
    "männer",
    "vater",
    "väter",
    "sohn",
    "söhne",
    "bruder",
    "brüder",
    "arzt",
    "ärzte",
    "bürger",
    "mitarbeiter",
    "kollege",
    "kollegen",
}
FEMININE_NOUNS = {
    "frau",
    "frauen",
    "mutter",
    "mütter",
    "tochter",
    "töchter",
    "schwester",
    "schwestern",
    "ärztin",
    "ärztinnen",
    "bürgerin",
    "bürgerinnen",
    "mitarbeiterin",
    "mitarbeiterinnen",
    "kollegin",
    "kolleginnen",
}
NEUTRAL_PERSON_NOUNS = {
    "person",
    "personen",
    "mensch",
    "menschen",
    "fachkraft",
    "fachkräfte",
    "lehrkraft",
    "lehrkräfte",
    "kraft",
    "kräfte",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def count_matching_lemmas(doc, vocabulary: set[str]) -> int:
    count = 0
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if lemma in vocabulary:
            count += 1
    return count


def count_matching_tokens(doc, vocabulary: set[str]) -> int:
    count = 0
    for token in doc:
        token_text = token.text.lower().strip()
        if token_text in vocabulary:
            count += 1
    return count


def count_inclusive_forms(doc) -> int:
    count = 0
    for token in doc:
        if INCLUSIVE_RE.match(token.text):
            count += 1
    return count


def extract_person_entities(doc) -> tuple[int, list[str]]:
    persons = []
    for ent in doc.ents:
        if ent.label_ == "PER":
            persons.append(ent.text.strip())
    unique_persons = sorted(set(persons))
    return len(persons), unique_persons


def document_features(pair_id: str, side: str, source: str, url: str, text: str, doc) -> dict[str, object]:
    token_count = sum(1 for token in doc if not token.is_space)
    alpha_token_count = sum(1 for token in doc if token.is_alpha)
    sentence_count = sum(1 for _ in doc.sents)
    person_entity_count, unique_person_entities = extract_person_entities(doc)

    masculine_pronoun_count = count_matching_tokens(doc, MASCULINE_PRONOUNS)
    feminine_pronoun_count = count_matching_tokens(doc, FEMININE_PRONOUNS)
    neutral_pronoun_count = count_matching_tokens(doc, NEUTRAL_PRONOUNS)

    masculine_noun_count = count_matching_lemmas(doc, MASCULINE_NOUNS)
    feminine_noun_count = count_matching_lemmas(doc, FEMININE_NOUNS)
    neutral_person_noun_count = count_matching_lemmas(doc, NEUTRAL_PERSON_NOUNS)
    inclusive_form_count = count_inclusive_forms(doc)

    return {
        "pair_id": pair_id,
        "side": side,
        "source": source,
        "url": url,
        "text": text,
        "char_count": len(text),
        "token_count": token_count,
        "alpha_token_count": alpha_token_count,
        "sentence_count": sentence_count,
        "person_entity_count": person_entity_count,
        "unique_person_entity_count": len(unique_person_entities),
        "person_entities": " | ".join(unique_person_entities),
        "masculine_pronoun_count": masculine_pronoun_count,
        "feminine_pronoun_count": feminine_pronoun_count,
        "neutral_pronoun_count": neutral_pronoun_count,
        "masculine_noun_count": masculine_noun_count,
        "feminine_noun_count": feminine_noun_count,
        "neutral_person_noun_count": neutral_person_noun_count,
        "inclusive_form_count": inclusive_form_count,
        "gender_marker_total": (
            masculine_pronoun_count
            + feminine_pronoun_count
            + neutral_pronoun_count
            + masculine_noun_count
            + feminine_noun_count
            + neutral_person_noun_count
            + inclusive_form_count
        ),
    }


def pair_features(simple_row: dict[str, object], normal_row: dict[str, object]) -> dict[str, object]:
    simple_tokens = int(simple_row["token_count"])
    normal_tokens = int(normal_row["token_count"])
    simple_sentences = int(simple_row["sentence_count"])
    normal_sentences = int(normal_row["sentence_count"])

    return {
        "pair_id": simple_row["pair_id"],
        "source": simple_row["source"],
        "simple_url": simple_row["url"],
        "normal_url": normal_row["url"],
        "simple_text": simple_row["text"],
        "normal_text": normal_row["text"],
        "simple_token_count": simple_tokens,
        "normal_token_count": normal_tokens,
        "token_ratio_normal_to_simple": round(normal_tokens / simple_tokens, 4) if simple_tokens else "",
        "simple_sentence_count": simple_sentences,
        "normal_sentence_count": normal_sentences,
        "sentence_ratio_normal_to_simple": round(normal_sentences / simple_sentences, 4)
        if simple_sentences
        else "",
        "simple_person_entity_count": simple_row["person_entity_count"],
        "normal_person_entity_count": normal_row["person_entity_count"],
        "delta_person_entity_count": int(normal_row["person_entity_count"]) - int(simple_row["person_entity_count"]),
        "simple_masculine_markers": int(simple_row["masculine_pronoun_count"]) + int(simple_row["masculine_noun_count"]),
        "normal_masculine_markers": int(normal_row["masculine_pronoun_count"]) + int(normal_row["masculine_noun_count"]),
        "simple_feminine_markers": int(simple_row["feminine_pronoun_count"]) + int(simple_row["feminine_noun_count"]),
        "normal_feminine_markers": int(normal_row["feminine_pronoun_count"]) + int(normal_row["feminine_noun_count"]),
        "simple_neutral_markers": int(simple_row["neutral_pronoun_count"])
        + int(simple_row["neutral_person_noun_count"])
        + int(simple_row["inclusive_form_count"]),
        "normal_neutral_markers": int(normal_row["neutral_pronoun_count"])
        + int(normal_row["neutral_person_noun_count"])
        + int(normal_row["inclusive_form_count"]),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_summary(doc_rows: list[dict[str, object]], pair_rows: list[dict[str, object]]) -> dict[str, object]:
    source_counter = Counter(row["source"] for row in pair_rows)
    return {
        "pair_count": len(pair_rows),
        "document_count": len(doc_rows),
        "sources": dict(sorted(source_counter.items())),
        "columns": {
            "doc_features": list(doc_rows[0].keys()) if doc_rows else [],
            "pair_features": list(pair_rows[0].keys()) if pair_rows else [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument("--pair-output", type=Path, default=DEFAULT_PAIR_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model", default="de_core_news_sm")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    nlp = spacy.load(args.model, disable=["textcat", "parser"])
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc_inputs: list[tuple[str, str, str, str, str]] = []
    for index, row in enumerate(rows, start=1):
        pair_id = f"pair_{index:04d}"
        source = row["source"]
        simple_text = normalize_text(row["simple_text"])
        normal_text = normalize_text(row["normal_text"])
        doc_inputs.append((pair_id, "simple", source, row["simple_url"], simple_text))
        doc_inputs.append((pair_id, "normal", source, row["normal_url"], normal_text))

    doc_rows: list[dict[str, object]] = []
    for meta, doc in zip(doc_inputs, nlp.pipe((item[4] for item in doc_inputs), batch_size=16), strict=True):
        pair_id, side, source, url, text = meta
        doc_rows.append(document_features(pair_id, side, source, url, text, doc))

    by_pair_and_side = {(row["pair_id"], row["side"]): row for row in doc_rows}
    pair_rows: list[dict[str, object]] = []
    for index in range(1, len(rows) + 1):
        pair_id = f"pair_{index:04d}"
        simple_row = by_pair_and_side[(pair_id, "simple")]
        normal_row = by_pair_and_side[(pair_id, "normal")]
        pair_rows.append(pair_features(simple_row, normal_row))

    args.doc_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.doc_output, doc_rows)
    write_csv(args.pair_output, pair_rows)

    summary = build_summary(doc_rows, pair_rows)
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
