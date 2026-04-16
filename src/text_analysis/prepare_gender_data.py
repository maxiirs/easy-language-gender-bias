#!/usr/bin/env python3
"""Prepare text-side gender analysis data and review queues.

This script applies automatic rules to LLpro + Kaikki data, writes one row per
mention, and exports the remaining ambiguous cases for manual review.

Automatic fine-grained frames:
    fem
    male_specific
    gmasc_generic
    epic_individual
    epic_group
    pair
    named_person
    unk

Image-ready labels:
    f1, m1, fg, mg, mx, ux
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/annotated_texts.json")
DEFAULT_OUTPUT = Path("data/processed/gender_analysis_auto.csv")
DEFAULT_SUMMARY = Path("data/processed/gender_analysis_auto_summary.json")
DEFAULT_REVIEW = Path("data/annotations/gender_review.csv")
DEFAULT_KAIKKI_CACHE = Path("data/resources/kaikki_de_lookup.jsonl")

PAIR_FORM_RE = re.compile(
    r"\b\w+innen\b\s+und\s+\b\w+\b|"
    r"\b\w+\b\s+und\s+\b\w+innen\b|"
    r"\b\w+innen\b\s+und\b|"
    r"\b\w+/\s*innen\b",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")
HUMAN_GLOSS_RE = re.compile(r"\b(mensch(?:en)?|person(?:en)?)\b", re.IGNORECASE)
COLLECTIVE_GLOSS_RE = re.compile(
    r"\b("
    r"gruppe\b.*\bpersonen|"
    r"aus mehreren personen|"
    r"gremium|"
    r"organisationseinheit|"
    r"eltern und kinder|"
    r"großeltern|"
    r"partnern mit nachwuchs|"
    r"kammer des parlaments|"
    r"entscheidungsgremium|"
    r"verwaltungsbehörde|"
    r"regierungsbehörde|"
    r"gemeinschaft\b.*\bpersonen"
    r")\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PERSON_NAME_TOKEN_RE = re.compile(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'-]+\.?$")
DOT_COMPACT_RE = re.compile(r"\S\.\S")
METADATA_SENTENCE_RE = re.compile(
    r"(hat den text in leichter sprache geschrieben|"
    r"hat den text geschrieben|"
    r"hat die bilder gemalt|"
    r"hat die bilder gemacht|"
    r"hat den text geprüft|"
    r"hat die bilder gezeichnet|"
    r"text in leichter sprache|"
    r"bilder gemalt|"
    r"stock\.adobe\.com|"
    r"pexels/|"
    r"©|"
    r"copyright)",
    re.IGNORECASE,
)

PERSON_ENTITY_LABEL = "PER"
NON_PERSON_ENTITY_LABELS = {"LOC", "ORG"}
GENDERED_PRONOUN_TAGS = {"PPER", "PPOSAT", "PPOSS", "PRELS"}
AUTO_FRAME_VALUES = {
    "fem",
    "male_specific",
    "gmasc_generic",
    "epic_individual",
    "epic_group",
    "pair",
    "named_person",
    "unk",
}


@dataclass(frozen=True)
class KaikkiLexicon:
    """Compact lexical signals derived from the local Kaikki cache."""

    human_nouns: frozenset[str]
    kaikki_collective_nouns: frozenset[str]
    explicit_female: frozenset[str]
    gmasc_candidates: frozenset[str]
    female_names: frozenset[str]
    male_names: frozenset[str]


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


def load_kaikki_cache(path: Path) -> KaikkiLexicon:
    """Load a local Kaikki cache and derive lexical sets."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Kaikki cache: {path}. "
            "Build it first with src/lexicon/build_kaikki_lookup.py."
        )

    human_nouns: set[str] = set()
    kaikki_collective_nouns: set[str] = set()
    explicit_female: set[str] = set()
    gmasc_candidates: set[str] = set()
    female_names: set[str] = set()
    male_names: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("lang") != "Deutsch":
                continue

            lemma = normalize_text(entry.get("word", ""))
            if not lemma:
                continue

            tags = {normalize_text(tag) for tag in entry.get("tags", [])}
            pos = entry.get("pos")

            if pos == "name":
                if "feminine" in tags and "masculine" not in tags:
                    female_names.add(lemma)
                elif "masculine" in tags and "feminine" not in tags:
                    male_names.add(lemma)
                continue

            if pos != "noun":
                continue

            forms = entry.get("forms", [])
            glosses = []
            for sense in entry.get("senses", []):
                glosses.extend(sense.get("glosses", []))
            gloss_text = normalize_text(" ".join(glosses))
            is_human_noun = bool(HUMAN_GLOSS_RE.search(gloss_text))
            is_collective_noun = bool(COLLECTIVE_GLOSS_RE.search(gloss_text))

            if is_human_noun:
                human_nouns.add(lemma)
            if is_collective_noun:
                kaikki_collective_nouns.add(lemma)

            has_feminine_counterpart = any(
                "feminine" in {normalize_text(tag) for tag in form.get("tags", [])}
                and normalize_text(form.get("form", "")) != lemma
                for form in forms
            )
            has_masculine_counterpart = any(
                "masculine" in {normalize_text(tag) for tag in form.get("tags", [])}
                and normalize_text(form.get("form", "")) != lemma
                for form in forms
            )

            if "feminine" in tags and has_masculine_counterpart:
                explicit_female.add(lemma)
            if "masculine" in tags and has_feminine_counterpart:
                gmasc_candidates.add(lemma)

    return KaikkiLexicon(
        human_nouns=frozenset(human_nouns),
        kaikki_collective_nouns=frozenset(kaikki_collective_nouns),
        explicit_female=frozenset(explicit_female),
        gmasc_candidates=frozenset(gmasc_candidates),
        female_names=frozenset(female_names),
        male_names=frozenset(male_names),
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


def has_disallowed_surface(text: str) -> bool:
    """Reject obvious non-actor surface forms such as emails or handles."""
    lowered = text.strip().lower()
    if not lowered:
        return True
    if EMAIL_RE.search(text):
        return True
    if "@" in text or "_" in text or "://" in text or lowered.startswith("www."):
        return True
    if DOT_COMPACT_RE.search(text):
        return True
    return False


def is_metadata_sentence(text: str) -> bool:
    """Detect credit/footer sentences that should not enter actor analysis."""
    return bool(METADATA_SENTENCE_RE.search(text or ""))


def get_person_entity_texts(data: dict) -> set[str]:
    """Collect normalized PERSON entity strings for one document."""
    person_entities = set()
    for entity in data.get("entities", []):
        if entity.get("label") == PERSON_ENTITY_LABEL:
            text = normalize_text(entity.get("text", ""))
            if text:
                person_entities.add(text)
    return person_entities


def has_propn_token(character: dict) -> bool:
    """Return True if the mention contains at least one proper noun token."""
    return any(token.get("pos") == "PROPN" for token in character.get("tokens", []))


def looks_like_person_name(character: dict) -> bool:
    """Detect person-like proper-name spans without relying on lemma lists."""
    if has_disallowed_surface(character.get("text", "")):
        return False

    content_tokens = [
        token for token in character.get("tokens", [])
        if token.get("pos") != "PUNCT" and token.get("text", "").strip()
    ]
    if len(content_tokens) < 2:
        return False
    if not all(token.get("pos") == "PROPN" for token in content_tokens):
        return False

    matched_tokens = 0
    for token in content_tokens:
        token_text = token.get("text", "")
        if token_text.isupper() and len(token_text) > 1:
            return False
        if not PERSON_NAME_TOKEN_RE.match(token_text):
            return False
        matched_tokens += 1
    return matched_tokens >= 2


def is_person_name(character: dict, person_entities: set[str], entity_conflict: str) -> bool:
    """Return True for person names backed by NER or name-like formatting."""
    text = normalize_text(character.get("text", ""))
    if not text:
        return False
    if has_disallowed_surface(character.get("text", "")):
        return False
    if entity_conflict:
        return False
    if text in person_entities:
        return True
    if not has_propn_token(character):
        return False

    text_parts = text.split()
    if len(text_parts) == 1 and len(text_parts[0]) >= 3:
        if any(text_parts[0] in entity.split() for entity in person_entities):
            return True
    elif len(text_parts) > 1:
        if any(text in entity or entity in text for entity in person_entities):
            return True

    return looks_like_person_name(character)


def get_non_person_entity_conflict(character: dict, entities: list[dict]) -> str:
    """Return LOC/ORG label if the mention appears to be embedded in a non-person entity."""
    mention_text = normalize_text(character.get("text", ""))
    sent_text = normalize_text(character.get("sent_text", ""))
    if not mention_text:
        return ""

    for entity in entities:
        label = entity.get("label", "")
        if label not in NON_PERSON_ENTITY_LABELS:
            continue
        entity_text = normalize_text(entity.get("text", ""))
        if not entity_text:
            continue
        if sent_text and entity_text not in sent_text:
            continue
        if mention_text == entity_text:
            return label.lower()
        if len(mention_text) >= 4 and mention_text in entity_text:
            return label.lower()
    return ""


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


def infer_named_person_gender(character: dict, lexicon: KaikkiLexicon) -> str:
    """Infer a named person's referential gender from LLpro or Kaikki name data."""
    morph_genders = {
        parse_morph(token, "Gender")
        for token in character.get("tokens", [])
        if token.get("pos") == "PROPN" and parse_morph(token, "Gender")
    }
    if len(morph_genders) == 1:
        gender = next(iter(morph_genders))
        if gender == "Fem":
            return "fem"
        if gender == "Masc":
            return "male"

    span_gender = character.get("morph_gender")
    if span_gender == "Fem":
        return "fem"
    if span_gender == "Masc":
        return "male"

    first_propn = next(
        (
            normalize_text(token.get("lemma", "") or token.get("text", ""))
            for token in character.get("tokens", [])
            if token.get("pos") == "PROPN"
        ),
        "",
    )
    if not first_propn:
        return ""
    if first_propn in lexicon.female_names and first_propn not in lexicon.male_names:
        return "fem"
    if first_propn in lexicon.male_names and first_propn not in lexicon.female_names:
        return "male"
    return ""


def is_pair_form(text: str) -> bool:
    """Detect explicit paired forms such as 'Bürgerinnen und Bürger'."""
    return bool(PAIR_FORM_RE.search(text))


def is_human_actor(
    character: dict,
    entities: list[dict],
    person_entities: set[str],
    lexicon: KaikkiLexicon,
    coref_gender: str,
) -> bool:
    """Keep likely human actor mentions."""
    head = get_head_token(character)
    if not head:
        return False
    if is_metadata_sentence(character.get("sent_text", "")):
        return False
    if has_disallowed_surface(character.get("text", "")):
        return False

    entity_conflict = get_non_person_entity_conflict(character, entities)

    if is_person_name(character, person_entities, entity_conflict):
        return True

    pos = head.get("pos", "")
    if pos == "PROPN":
        return False
    if pos == "PRON":
        return False
    if is_pair_form(character.get("text", "")):
        return True
    if coref_gender:
        return True
    if pos == "NOUN" or head.get("tag") == "NN":
        lemma = normalize_text(head.get("lemma", ""))
        is_human = (
            lemma in lexicon.human_nouns
            or lemma in lexicon.kaikki_collective_nouns
        )
        if entity_conflict and not is_human:
            return False
        return is_human
    return False


def map_frame_to_t2i(frame: str, is_group: bool, is_named_person: bool) -> str:
    """Map fine-grained text frames to compact image-ready labels."""
    if is_named_person:
        return "ux"
    if frame == "fem":
        return "fg" if is_group else "f1"
    if frame == "male_specific":
        return "mg" if is_group else "m1"
    if frame in {"pair", "epic_group"}:
        return "mx"
    if frame in {"gmasc_generic", "epic_individual", "named_person", "unk"}:
        return "ux"
    return "ux"


def needs_review(auto_frame: str, is_named_person: bool, is_group: bool) -> bool:
    """Decide whether a mention should be exported for manual review."""
    if auto_frame == "unk":
        return True
    if auto_frame == "named_person":
        return True
    if auto_frame == "gmasc_generic" and not is_group:
        return True
    return False


def get_review_reason(auto_frame: str, is_named_person: bool, is_group: bool) -> str:
    """Explain why a mention was exported for manual review."""
    if auto_frame == "unk":
        return "no_automatic_signal"
    if auto_frame == "named_person":
        return "named_person_separate_text_track"
    if auto_frame == "gmasc_generic" and not is_group:
        return "singular_gmasc_needs_generic_vs_specific_decision"
    return ""


def classify_mention(
    *,
    character: dict,
    pair_id: str,
    source: str,
    side: str,
    person_entities: set[str],
    lexicon: KaikkiLexicon,
    coref_clusters: list[dict],
    entities: list[dict],
) -> dict:
    """Classify one mention and return a flat row."""
    head = get_head_token(character)
    lemma = normalize_text(head.get("lemma", ""))
    text = character.get("text", "")
    number = get_number(character)
    is_group = number == "Plur" or is_pair_form(text)
    entity_conflict = get_non_person_entity_conflict(character, entities)
    named_person = is_person_name(character, person_entities, entity_conflict)
    coref_gender = infer_coref_gender(character, coref_clusters)
    named_person_gender = infer_named_person_gender(character, lexicon) if named_person else ""

    if is_pair_form(text):
        auto_frame = "pair"
        auto_reason = "pair_form"
    elif coref_gender == "male":
        auto_frame = "male_specific"
        auto_reason = "coref_male"
    elif coref_gender == "fem":
        auto_frame = "fem"
        auto_reason = "coref_fem"
    elif named_person and named_person_gender == "male":
        auto_frame = "male_specific"
        auto_reason = "name_signal_male"
    elif named_person and named_person_gender == "fem":
        auto_frame = "fem"
        auto_reason = "name_signal_fem"
    elif named_person:
        auto_frame = "named_person"
        auto_reason = "person_name_unresolved"
    elif lemma in lexicon.explicit_female:
        auto_frame = "fem"
        auto_reason = "kaikki_fem"
    elif lemma in lexicon.gmasc_candidates and character.get("morph_gender") == "Masc":
        auto_frame = "gmasc_generic"
        auto_reason = "kaikki_gmasc"
    elif lemma in lexicon.kaikki_collective_nouns:
        auto_frame = "epic_group"
        auto_reason = "kaikki_collective_noun"
    elif lemma in lexicon.human_nouns:
        auto_frame = "epic_group" if is_group else "epic_individual"
        auto_reason = "kaikki_human_noun"
    elif entity_conflict:
        auto_frame = "unk"
        auto_reason = f"entity_conflict_{entity_conflict}"
    else:
        auto_frame = "unk"
        auto_reason = "no_signal"

    if auto_frame not in AUTO_FRAME_VALUES:
        raise ValueError(f"Unexpected auto frame: {auto_frame}")

    mention_id = f"{pair_id}:{side}:{character.get('start')}:{character.get('end')}"
    review_flag = needs_review(auto_frame, named_person, is_group)
    review_reason = get_review_reason(auto_frame, named_person, is_group) if review_flag else ""

    return {
        "mention_id": mention_id,
        "pair_id": pair_id,
        "source": source,
        "side": side,
        "start": character.get("start", ""),
        "end": character.get("end", ""),
        "text": text,
        "lemma": head.get("lemma", ""),
        "pos": head.get("pos", ""),
        "morph_gender": character.get("morph_gender") or "",
        "number": number,
        "is_group": int(is_group),
        "is_named_person": int(named_person),
        "coref_gender": coref_gender,
        "entity_conflict": entity_conflict,
        "auto_frame": auto_frame,
        "auto_t2i_label": map_frame_to_t2i(auto_frame, is_group, named_person),
        "auto_reason": auto_reason,
        "review_needed": int(review_flag),
        "review_reason": review_reason,
        "sent_text": character.get("sent_text", ""),
    }


def process_pair(
    pair: dict,
    lexicon: KaikkiLexicon,
) -> list[dict]:
    """Convert one text pair into mention-level analysis rows."""
    rows: list[dict] = []
    pair_id = pair.get("pair_id", "")
    source = pair.get("source", "")

    for side in ("simple", "normal"):
        data = pair.get(side)
        if not data or "error" in data:
            continue

        person_entities = get_person_entity_texts(data)
        entities = data.get("entities", [])
        coref_clusters = data.get("coref_clusters", [])

        for character in data.get("characters", []):
            coref_gender = infer_coref_gender(character, coref_clusters)
            if not is_human_actor(
                character,
                entities,
                person_entities,
                lexicon,
                coref_gender,
            ):
                continue

            rows.append(
                classify_mention(
                    character=character,
                    pair_id=pair_id,
                    source=source,
                    side=side,
                    person_entities=person_entities,
                    lexicon=lexicon,
                    coref_clusters=coref_clusters,
                    entities=entities,
                )
            )

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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_review_rows(rows: list[dict]) -> list[dict]:
    """Create a lemma-level manual review template for uncertain cases."""
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if not int(row["review_needed"]):
            continue
        key = (row["lemma"], row["auto_frame"], row["review_reason"])
        entry = grouped.setdefault(
            key,
            {
                "review_key": f"{row['lemma']}|{row['auto_frame']}|{row['review_reason']}",
                "lemma": row["lemma"],
                "auto_frame": row["auto_frame"],
                "auto_t2i_label": row["auto_t2i_label"],
                "review_reason": row["review_reason"],
                "is_group": row["is_group"],
                "is_named_person": row["is_named_person"],
                "example_count": 0,
                "example_texts": [],
                "example_sentences": [],
                "manual_frame": "",
                "manual_t2i_label": "",
                "manual_note": "",
            },
        )
        entry["example_count"] += 1
        if row["text"] not in entry["example_texts"] and len(entry["example_texts"]) < 5:
            entry["example_texts"].append(row["text"])
        if row["sent_text"] not in entry["example_sentences"] and len(entry["example_sentences"]) < 5:
            entry["example_sentences"].append(row["sent_text"])

    review_rows = []
    for entry in grouped.values():
        review_rows.append(
            {
                "review_key": entry["review_key"],
                "lemma": entry["lemma"],
                "auto_frame": entry["auto_frame"],
                "auto_t2i_label": entry["auto_t2i_label"],
                "review_reason": entry["review_reason"],
                "is_group": entry["is_group"],
                "is_named_person": entry["is_named_person"],
                "example_count": entry["example_count"],
                "example_texts": " || ".join(entry["example_texts"]),
                "example_sentences": " || ".join(entry["example_sentences"]),
                "manual_frame": "",
                "manual_t2i_label": "",
                "manual_note": "",
            }
        )
    review_rows.sort(key=lambda row: (-int(row["example_count"]), row["lemma"].lower(), row["review_reason"]))
    return review_rows


def build_summary(
    rows: list[dict],
    lexicon: KaikkiLexicon,
    cache_path: Path,
    review_rows: list[dict],
) -> dict:
    """Create compact summary statistics for later analysis."""
    summary = {
        "total_mentions": len(rows),
        "mapped_mentions": sum(row["auto_t2i_label"] != "ux" for row in rows),
        "review_mentions": len(review_rows),
        "kaikki_cache": str(cache_path),
        "lexicon_counts": {
            "human_nouns": len(lexicon.human_nouns),
            "kaikki_collective_nouns": len(lexicon.kaikki_collective_nouns),
            "explicit_female": len(lexicon.explicit_female),
            "gmasc_candidate": len(lexicon.gmasc_candidates),
        },
        "by_side": {},
        "by_side_t2i": {},
        "by_side_frame": {},
    }

    for side in ("simple", "normal"):
        side_rows = [row for row in rows if row["side"] == side]
        summary["by_side"][side] = len(side_rows)
        summary["by_side_t2i"][side] = dict(
            Counter(row["auto_t2i_label"] for row in side_rows).most_common()
        )
        summary["by_side_frame"][side] = dict(
            Counter(row["auto_frame"] for row in side_rows).most_common()
        )

    summary["review_reasons"] = dict(
        Counter(row["review_reason"] for row in review_rows).most_common()
    )
    return summary


def print_summary(summary: dict) -> None:
    """Print high-level distribution tables."""
    print("\n" + "=" * 60)
    print("TEXT LABEL DISTRIBUTION")
    print("=" * 60)
    print(f"Total mentions:   {summary['total_mentions']}")
    print(f"Mapped mentions:  {summary['mapped_mentions']}")
    print(f"Review mentions:  {summary['review_mentions']}")
    print(f"Kaikki cache:     {summary['kaikki_cache']}")
    print(
        "Lexicon counts:   "
        f"human={summary['lexicon_counts']['human_nouns']} "
        f"kaikki_collective={summary['lexicon_counts']['kaikki_collective_nouns']} "
        f"f={summary['lexicon_counts']['explicit_female']} "
        f"gmasc={summary['lexicon_counts']['gmasc_candidate']}"
    )

    for side, label in (("simple", "Easy Language"), ("normal", "Standard German")):
        total = summary["by_side"][side]
        print(f"\n{label}: {total} mentions")
        print("  T2I labels:")
        for key, count in summary["by_side_t2i"][side].items():
            share = count / total * 100 if total else 0
            print(f"    {key:>2}: {count:5d} ({share:5.1f}%)")
        print("  Fine frames:")
        for key, count in summary["by_side_frame"][side].items():
            share = count / total * 100 if total else 0
            print(f"    {key:>15}: {count:5d} ({share:5.1f}%)")

    if summary["review_reasons"]:
        print("\nReview reasons:")
        for key, count in summary["review_reasons"].items():
            print(f"  {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare mention-level gender analysis data and review queues."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--kaikki-cache", type=Path, default=DEFAULT_KAIKKI_CACHE)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        pairs = json.load(handle)
    print(f"Loaded pairs: {len(pairs)}")

    if not has_coref_token_features(pairs):
        print(
            "Warning: coreference mentions do not contain token features. "
            "Pronoun-based gender inference is disabled for this annotation file. "
            "Re-run src/annotation/annotate_texts.py for full morphology-based coref evidence."
        )

    lexicon = load_kaikki_cache(args.kaikki_cache)
    rows: list[dict] = []
    for pair in pairs:
        rows.extend(process_pair(pair, lexicon))
    print(f"Kept mentions: {len(rows)}")

    review_rows = build_review_rows(rows)
    write_csv(args.output, rows)
    write_csv(args.review_output, review_rows)

    summary = build_summary(
        rows,
        lexicon,
        args.kaikki_cache,
        review_rows,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nAuto CSV:   {args.output}")
    print(f"Review CSV: {args.review_output}")
    print(f"Summary:    {args.summary}")


if __name__ == "__main__":
    main()
