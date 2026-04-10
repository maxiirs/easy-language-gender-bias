#!/usr/bin/env python3
"""Prepare image-ready gender labels from LLpro-annotated texts.

The script is designed for comparative bias analysis between Easy Language
and Standard German. It maps person mentions to compact labels that can be
reused later for MiVOLO-based image analysis:

    f1  -> one female person
    m1  -> one male person
    fg  -> female group
    mg  -> male group
    mx  -> mixed / generic group
    ux  -> unresolved

The code keeps lexical assumptions outside Python source code. Open lexical
resources should be stored under ``data/resources/`` with explicit provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/annotated_texts.json")
DEFAULT_OUTPUT = Path("data/processed/gender_analysis.csv")
DEFAULT_SUMMARY = Path("data/processed/gender_analysis_summary.json")
DEFAULT_PERSON_LEXICON = Path("data/resources/person_lexicon.csv")

PAIR_FORM_RE = re.compile(
    r"\b\w+innen\b\s+und\s+\b\w+\b|\b\w+\b\s+und\s+\b\w+innen\b|\b\w+innen\b\s+und\b",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")

PERSON_ENTITY_LABEL = "PER"
GENDERED_PRONOUN_TAGS = {"PPER", "PPOSAT", "PPOSS", "PRELS"}
PERSON_NOUN_SUFFIXES = ("er", "erin", "ist", "ent", "ant", "eur", "oge", "ling")
PERSON_PARTICIPLE_SUFFIXES = ("ende", "enden", "ender", "endem")
UMLAUT_REVERSE = {"ä": "a", "ö": "o", "ü": "u"}


@dataclass(frozen=True)
class PersonLexicon:
    """External lexical resources with explicit provenance on disk."""

    explicit_female: frozenset[str]
    explicit_male: frozenset[str]
    collectives: frozenset[str]
    person_hints: frozenset[str]


def normalize_text(text: str) -> str:
    """Normalize text for robust matching."""
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def parse_morph(token: dict, feature: str) -> str | None:
    """Extract one feature from a serialized spaCy morph string."""
    morph = token.get("morph", "")
    for part in morph.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == feature:
            return value
    return None


def load_person_lexicon(path: Path) -> PersonLexicon:
    """Load a person lexicon with transparent provenance fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing lexicon resource: {path}. "
            "Create it under data/resources/ before running the analysis."
        )

    grouped: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"lemma", "class", "source", "license"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain at least the columns: "
                f"{', '.join(sorted(required))}"
            )

        for row in reader:
            lemma = normalize_text(row.get("lemma", ""))
            entry_class = normalize_text(row.get("class", ""))
            if not lemma or not entry_class:
                continue
            grouped[entry_class].add(lemma)

    return PersonLexicon(
        explicit_female=frozenset(grouped["explicit_female"]),
        explicit_male=frozenset(grouped["explicit_male"]),
        collectives=frozenset(grouped["collective"]),
        person_hints=frozenset(grouped["person_hint"]),
    )


def get_head_token(character: dict) -> dict:
    """Pick the main content token of a character span."""
    tokens = character.get("tokens", [])
    for token in tokens:
        if token.get("pos") in {"NOUN", "PROPN", "PRON"}:
            return token
        if token.get("tag") == "NN":
            return token
    return tokens[0] if tokens else {}


def get_number(character: dict) -> str:
    """Return Sing / Plur if available for the character span."""
    for token in character.get("tokens", []):
        number = parse_morph(token, "Number")
        if number:
            return number
    return ""


def looks_like_person_name(text: str) -> bool:
    """Detect simple multi-token names such as 'Dirk Hilbert'."""
    parts = [part for part in re.split(r"[\s-]+", text.strip()) if part]
    if len(parts) < 2:
        return False
    uppercase_parts = sum(part[:1].isupper() for part in parts)
    return uppercase_parts >= 2


def build_movable_set(pairs: list[dict]) -> set[str]:
    """Build a corpus-internal set of movable masculine lemmas."""
    all_lemmas: set[str] = set()

    for pair in pairs:
        for side in ("simple", "normal"):
            data = pair.get(side)
            if not data:
                continue
            for character in data.get("characters", []):
                for token in character.get("tokens", []):
                    lemma = normalize_text(token.get("lemma", ""))
                    if lemma:
                        all_lemmas.add(lemma)

    movable: set[str] = set()
    for lemma in all_lemmas:
        if lemma.endswith("in") and len(lemma) > 3:
            base = lemma[:-2]
            movable.add(base)
            for umlaut, plain in UMLAUT_REVERSE.items():
                if umlaut in base:
                    movable.add(base.replace(umlaut, plain, 1))

    return movable


def get_person_entity_texts(data: dict) -> set[str]:
    """Collect normalized PERSON entity strings for one document."""
    person_entities = set()
    for entity in data.get("entities", []):
        if entity.get("label") == PERSON_ENTITY_LABEL:
            text = normalize_text(entity.get("text", ""))
            if text:
                person_entities.add(text)
    return person_entities


def is_person_name(character: dict, person_entities: set[str]) -> bool:
    """Return True for person names backed by NER or name-like formatting."""
    text = normalize_text(character.get("text", ""))
    if not text:
        return False
    if text in person_entities:
        return True

    text_parts = text.split()
    if len(text_parts) == 1 and len(text_parts[0]) >= 3:
        if any(text_parts[0] in entity.split() for entity in person_entities):
            return True
    elif len(text_parts) > 1:
        if any(text in entity or entity in text for entity in person_entities):
            return True

    return looks_like_person_name(character.get("text", ""))


def infer_coref_gender(character: dict, coref_clusters: list[dict]) -> str:
    """Infer gender from morphologically marked pronouns in the same cluster."""
    char_start = character.get("start")
    char_end = character.get("end")
    if char_start is None or char_end is None:
        return ""

    for cluster in coref_clusters:
        mentions = cluster.get("mentions", [])
        has_overlap = any(
            mention.get("start", -1) < char_end and mention.get("end", -1) > char_start
            for mention in mentions
        )
        if not has_overlap:
            continue

        female_count = 0
        male_count = 0
        for mention in mentions:
            for token in mention.get("tokens", []):
                tag = token.get("tag", "")
                if tag not in GENDERED_PRONOUN_TAGS:
                    continue
                number = parse_morph(token, "Number") or mention.get("morph_number")
                if number != "Sing":
                    continue
                gender = parse_morph(token, "Gender") or mention.get("morph_gender")
                if gender == "Fem":
                    female_count += 1
                elif gender == "Masc":
                    male_count += 1

        if female_count > male_count and female_count > 0:
            return "fem"
        if male_count > female_count and male_count > 0:
            return "male"

    return ""


def is_pair_form(text: str) -> bool:
    """Detect explicit paired forms such as 'Bürgerinnen und Bürger'."""
    return bool(PAIR_FORM_RE.search(text))


def looks_like_person_noun(
    character: dict,
    movable: set[str],
    lexicon: PersonLexicon,
    coref_gender: str,
) -> bool:
    """Heuristically keep only person-denoting common nouns."""
    text = character.get("text", "")
    head = get_head_token(character)
    lemma = normalize_text(head.get("lemma", ""))

    if is_pair_form(text):
        return True
    if coref_gender:
        return True
    if lemma in lexicon.collectives:
        return True
    if lemma in lexicon.explicit_female or lemma in lexicon.explicit_male:
        return True
    if lemma in lexicon.person_hints:
        return True
    if lemma in movable:
        return True
    if lemma.endswith("in") and len(lemma) > 3:
        return True
    if lemma.endswith(PERSON_NOUN_SUFFIXES):
        return True
    if lemma.endswith(PERSON_PARTICIPLE_SUFFIXES):
        return True
    return False


def is_human_actor(
    character: dict,
    person_entities: set[str],
    movable: set[str],
    lexicon: PersonLexicon,
    coref_gender: str,
) -> bool:
    """Keep likely person-denoting spans and remove obvious prompt noise."""
    head = get_head_token(character)
    if not head:
        return False

    if is_person_name(character, person_entities):
        return True

    pos = head.get("pos", "")
    if pos == "PROPN":
        return False
    if pos == "PRON":
        return False
    if pos == "NOUN" or head.get("tag") == "NN":
        return looks_like_person_noun(character, movable, lexicon, coref_gender)
    return False


def is_collective_group(lemma: str, number: str, lexicon: PersonLexicon) -> bool:
    """Treat plural mentions and curated collectives as group prompts."""
    return number == "Plur" or lemma in lexicon.collectives


def infer_form(
    character: dict,
    movable: set[str],
    person_entities: set[str],
    lexicon: PersonLexicon,
    coref_gender: str,
) -> tuple[str, str, str, str]:
    """Infer linguistic frame, number, group flag, and reason."""
    text = character.get("text", "")
    head = get_head_token(character)
    lemma = normalize_text(head.get("lemma", ""))
    pos = head.get("pos", "")
    number = get_number(character)
    group = "g" if is_collective_group(lemma, number, lexicon) or is_pair_form(text) else "1"

    if is_pair_form(text):
        return "pair", number or "Plur", "g", "pair_form"

    if is_person_name(character, person_entities):
        if group == "1" and coref_gender == "fem":
            return "fem", number or "Sing", group, "coref_name_fem"
        if group == "1" and coref_gender == "male":
            return "male", number or "Sing", group, "coref_name_male"
        return "name", number or "Sing", group, "person_name"

    if group == "1" and coref_gender == "fem":
        return "fem", number or "Sing", group, "coref_fem"
    if group == "1" and coref_gender == "male":
        return "male", number or "Sing", group, "coref_male"

    if lemma in lexicon.explicit_female:
        return "fem", number or "Sing", group, "lexicon_fem"
    if lemma in lexicon.explicit_male:
        return "male", number or "Sing", group, "lexicon_male"

    if pos == "NOUN" and lemma.endswith("in") and len(lemma) > 3:
        return "fem", number or "Sing", group, "suffix_in"

    if character.get("morph_gender") == "Masc" and lemma in movable:
        return "gmasc", number or "Sing", group, "movable_masc"

    if group == "g":
        return "epic", number or "Plur", group, "collective_or_plural"

    return "unk", number or "", group, "no_signal"


def map_form_to_label(form: str, group: str) -> str:
    """Map linguistic frames to compact image-ready labels."""
    if form == "fem":
        return "fg" if group == "g" else "f1"
    if form == "male":
        return "mg" if group == "g" else "m1"
    if form in {"pair", "epic"}:
        return "mx"
    if form == "gmasc":
        return "mx" if group == "g" else "ux"
    return "ux"


def process_pair(pair: dict, movable: set[str], lexicon: PersonLexicon) -> list[dict]:
    """Convert one text pair into analysis rows."""
    rows: list[dict] = []
    pair_id = pair.get("pair_id", "")
    source = pair.get("source", "")

    for side in ("simple", "normal"):
        data = pair.get(side)
        if not data or "error" in data:
            continue

        person_entities = get_person_entity_texts(data)
        coref_clusters = data.get("coref_clusters", [])

        for character in data.get("characters", []):
            coref_gender = infer_coref_gender(character, coref_clusters)
            if not is_human_actor(character, person_entities, movable, lexicon, coref_gender):
                continue

            head = get_head_token(character)
            form, number, group, reason = infer_form(
                character=character,
                movable=movable,
                person_entities=person_entities,
                lexicon=lexicon,
                coref_gender=coref_gender,
            )

            rows.append({
                "pair_id": pair_id,
                "source": source,
                "side": side,
                "text": character.get("text", ""),
                "lemma": head.get("lemma", ""),
                "pos": head.get("pos", ""),
                "morph_gender": character.get("morph_gender") or "",
                "number": number,
                "group": group,
                "coref_gender": coref_gender,
                "frame": form,
                "label": map_form_to_label(form, group),
                "reason": reason,
                "sent_text": character.get("sent_text", ""),
            })

    return rows


def has_coref_token_features(pairs: list[dict]) -> bool:
    """Check whether annotated coreference mentions contain token features."""
    for pair in pairs:
        for side in ("simple", "normal"):
            data = pair.get(side) or {}
            for cluster in data.get("coref_clusters", []):
                for mention in cluster.get("mentions", []):
                    if mention.get("tokens"):
                        return True
    return False


def build_summary(rows: list[dict], movable_count: int, lexicon_path: Path) -> dict:
    """Create compact summary statistics for later analysis."""
    summary = {
        "total_mentions": len(rows),
        "mapped_mentions": sum(row["label"] != "ux" for row in rows),
        "movable_lemmas": movable_count,
        "person_lexicon": str(lexicon_path),
        "by_side": {},
        "by_side_label": {},
        "by_side_frame": {},
    }

    for side in ("simple", "normal"):
        side_rows = [row for row in rows if row["side"] == side]
        summary["by_side"][side] = len(side_rows)
        summary["by_side_label"][side] = dict(
            Counter(row["label"] for row in side_rows).most_common()
        )
        summary["by_side_frame"][side] = dict(
            Counter(row["frame"] for row in side_rows).most_common()
        )

    return summary


def print_summary(summary: dict) -> None:
    """Print high-level distribution tables."""
    print("\n" + "=" * 60)
    print("TEXT LABEL DISTRIBUTION")
    print("=" * 60)
    print(f"Total mentions:  {summary['total_mentions']}")
    print(f"Mapped mentions: {summary['mapped_mentions']}")
    print(f"Movable lemmas:  {summary['movable_lemmas']}")
    print(f"Lexicon:         {summary['person_lexicon']}")

    for side, label in (("simple", "Easy Language"), ("normal", "Standard German")):
        total = summary["by_side"][side]
        print(f"\n{label}: {total} mentions")
        print("  Labels:")
        for key, count in summary["by_side_label"][side].items():
            share = count / total * 100 if total else 0
            print(f"    {key:>2}: {count:5d} ({share:5.1f}%)")
        print("  Frames:")
        for key, count in summary["by_side_frame"][side].items():
            share = count / total * 100 if total else 0
            print(f"    {key:>5}: {count:5d} ({share:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare image-ready gender labels from annotated texts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--person-lexicon", type=Path, default=DEFAULT_PERSON_LEXICON)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        pairs = json.load(handle)
    print(f"Loaded pairs: {len(pairs)}")

    if not has_coref_token_features(pairs):
        print(
            "Warning: coreference mentions do not contain token features. "
            "Pronoun-based gender inference is disabled for this annotation file. "
            "Re-run src/pipeline/annotate_texts.py for full morphology-based coref evidence."
        )

    lexicon = load_person_lexicon(args.person_lexicon)
    movable = build_movable_set(pairs)
    print(f"Corpus-internal movable lemmas: {len(movable)}")

    rows: list[dict] = []
    for pair in pairs:
        rows.extend(process_pair(pair, movable, lexicon))
    print(f"Kept mentions: {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = build_summary(rows, len(movable), args.person_lexicon)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nCSV:     {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
