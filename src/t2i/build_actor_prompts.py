#!/usr/bin/env python3
"""Build actor-level prompt tables from final gender-analysis annotations.

The script aggregates mention-level rows into one actor row per
`pair_id + side + actor_key`, then builds short prompt variants that can be
used for text-to-image generation.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/gender_analysis_final.csv")
DEFAULT_PAIRS = Path("data/processed/paired_dataset_accepted.csv")
DEFAULT_OUTPUT = Path("data/processed/actor_prompts.csv")
DEFAULT_PAIRED_OUTPUT = Path("data/processed/actor_prompts_paired.csv")

WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for grouping and comparisons."""
    return WS_RE.sub(" ", text.strip().lower())


def capitalize_label(text: str) -> str:
    """Capitalize the first character without altering the rest."""
    text = text.strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def clean_sentence(text: str) -> str:
    """Normalize sentence whitespace."""
    return WS_RE.sub(" ", text.strip())


def finalize_prompt(*parts: str) -> str:
    """Join prompt parts with sentence punctuation."""
    cleaned = []
    for part in parts:
        part = clean_sentence(part)
        if not part:
            continue
        if part[-1] not in ".!?":
            part = f"{part}."
        cleaned.append(part)
    return " ".join(cleaned)


def load_csv_rows(path: Path) -> list[dict]:
    """Load a CSV file into memory."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pair_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Map pair_id to metadata from the paired corpus CSV."""
    rows = load_csv_rows(path)
    metadata: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=1):
        pair_id = f"pair_{index:04d}"
        metadata[pair_id] = {
            "topic": row.get("topic", ""),
            "simple_url": row.get("simple_url", ""),
            "normal_url": row.get("normal_url", ""),
            "source": row.get("source", ""),
        }
    return metadata


def get_actor_key(row: dict) -> str:
    """Build a within-document actor key for mention aggregation."""
    if row.get("is_named_person") == "1":
        base = normalize_text(row.get("text", ""))
    elif row.get("final_frame") == "pair":
        base = normalize_text(row.get("text", ""))
    else:
        base = normalize_text(row.get("lemma", "")) or normalize_text(row.get("text", ""))
    return base


def choose_display_label(group_rows: list[dict]) -> str:
    """Select a stable actor label for prompt generation."""
    final_frame = group_rows[0]["final_frame"]
    is_named = group_rows[0]["is_named_person"] == "1"
    is_group = group_rows[0]["is_group"] == "1"

    if is_group:
        counts = Counter(clean_sentence(row["text"]) for row in group_rows if row.get("text"))
        if counts:
            return counts.most_common(1)[0][0]

    if is_named or final_frame == "pair":
        counts = Counter(clean_sentence(row["text"]) for row in group_rows if row.get("text"))
        if counts:
            return counts.most_common(1)[0][0]

    lemma = group_rows[0].get("lemma", "")
    if lemma:
        return capitalize_label(lemma)

    counts = Counter(clean_sentence(row["text"]) for row in group_rows if row.get("text"))
    if counts:
        return counts.most_common(1)[0][0]
    return ""


def score_sentence(sentence: str, actor_label: str, actor_key: str) -> tuple[int, int]:
    """Score context sentences: prefer complete medium-length sentences."""
    sentence_norm = normalize_text(sentence)
    label_norm = normalize_text(actor_label)
    contains_actor = int(bool(label_norm and label_norm in sentence_norm) or bool(actor_key and actor_key in sentence_norm))
    length_penalty = abs(len(sentence) - 120)
    return contains_actor, -length_penalty


def choose_context_sentence(group_rows: list[dict], actor_label: str, actor_key: str) -> str:
    """Pick one representative context sentence for the actor."""
    candidates = []
    seen = set()
    for row in group_rows:
        sentence = clean_sentence(row.get("sent_text", ""))
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        candidates.append(sentence)
    if not candidates:
        return ""
    candidates.sort(key=lambda sent: score_sentence(sent, actor_label, actor_key), reverse=True)
    return candidates[0]


def build_actor_rows(final_rows: list[dict], pair_metadata: dict[str, dict[str, str]]) -> list[dict]:
    """Aggregate mention-level rows into actor-level rows."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in final_rows:
        if row.get("final_frame") == "unk":
            continue
        if row.get("pos") not in {"NOUN", "PROPN"}:
            continue
        group_key = (row["pair_id"], row["side"], get_actor_key(row))
        grouped[group_key].append(row)

    actor_rows: list[dict] = []
    for (pair_id, side, actor_key), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["start"]))
        sample = rows[0]
        metadata = pair_metadata.get(pair_id, {})
        actor_label = choose_display_label(rows)
        context_sentence = choose_context_sentence(rows, actor_label, actor_key)
        topic = metadata.get("topic", "")

        prompt_actor_only = finalize_prompt(actor_label)
        prompt_actor_context = finalize_prompt(actor_label, context_sentence)
        prompt_topic_actor_context = finalize_prompt(topic, actor_label, context_sentence)

        include_in_t2i_default = int(
            sample.get("is_named_person") != "1"
            and sample.get("final_frame") in {
                "fem",
                "male_specific",
                "gmasc_generic",
                "epic_individual",
                "epic_group",
                "pair",
            }
        )

        unique_texts = []
        seen_texts = set()
        for row in rows:
            text = clean_sentence(row.get("text", ""))
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append(text)

        actor_rows.append(
            {
                "actor_id": f"{pair_id}:{side}:{actor_key}",
                "pair_id": pair_id,
                "side": side,
                "source": metadata.get("source", sample.get("source", "")),
                "topic": topic,
                "url": metadata.get(f"{side}_url", ""),
                "actor_key": actor_key,
                "actor_label": actor_label,
                "final_frame": sample.get("final_frame", ""),
                "final_t2i_label": sample.get("final_t2i_label", ""),
                "is_group": sample.get("is_group", "0"),
                "is_named_person": sample.get("is_named_person", "0"),
                "mention_count": len(rows),
                "surface_forms": " || ".join(unique_texts[:10]),
                "context_sentence": context_sentence,
                "prompt_actor_only": prompt_actor_only,
                "prompt_actor_context": prompt_actor_context,
                "prompt_topic_actor_context": prompt_topic_actor_context,
                "include_in_t2i_default": include_in_t2i_default,
            }
        )

    return actor_rows


def build_paired_rows(actor_rows: list[dict]) -> list[dict]:
    """Align actor rows across simple and normal texts within one pair."""
    by_pair_and_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in actor_rows:
        by_pair_and_key[(row["pair_id"], row["actor_key"])][row["side"]] = row

    paired_rows: list[dict] = []
    for (pair_id, actor_key), sides in sorted(by_pair_and_key.items()):
        simple = sides.get("simple")
        normal = sides.get("normal")
        topic = (simple or normal or {}).get("topic", "")
        source = (simple or normal or {}).get("source", "")
        paired_rows.append(
            {
                "pair_actor_id": f"{pair_id}:{actor_key}",
                "pair_id": pair_id,
                "actor_key": actor_key,
                "topic": topic,
                "source": source,
                "has_simple": int(simple is not None),
                "has_normal": int(normal is not None),
                "simple_actor_id": simple.get("actor_id", "") if simple else "",
                "simple_actor_label": simple.get("actor_label", "") if simple else "",
                "simple_final_frame": simple.get("final_frame", "") if simple else "",
                "simple_final_t2i_label": simple.get("final_t2i_label", "") if simple else "",
                "simple_is_named_person": simple.get("is_named_person", "") if simple else "",
                "simple_context_sentence": simple.get("context_sentence", "") if simple else "",
                "simple_prompt_actor_only": simple.get("prompt_actor_only", "") if simple else "",
                "simple_prompt_actor_context": simple.get("prompt_actor_context", "") if simple else "",
                "simple_prompt_topic_actor_context": simple.get("prompt_topic_actor_context", "") if simple else "",
                "simple_include_in_t2i_default": simple.get("include_in_t2i_default", "") if simple else "",
                "normal_actor_id": normal.get("actor_id", "") if normal else "",
                "normal_actor_label": normal.get("actor_label", "") if normal else "",
                "normal_final_frame": normal.get("final_frame", "") if normal else "",
                "normal_final_t2i_label": normal.get("final_t2i_label", "") if normal else "",
                "normal_is_named_person": normal.get("is_named_person", "") if normal else "",
                "normal_context_sentence": normal.get("context_sentence", "") if normal else "",
                "normal_prompt_actor_only": normal.get("prompt_actor_only", "") if normal else "",
                "normal_prompt_actor_context": normal.get("prompt_actor_context", "") if normal else "",
                "normal_prompt_topic_actor_context": normal.get("prompt_topic_actor_context", "") if normal else "",
                "normal_include_in_t2i_default": normal.get("include_in_t2i_default", "") if normal else "",
            }
        )
    return paired_rows


def print_summary(actor_rows: list[dict], paired_rows: list[dict]) -> None:
    """Print compact stats for the generated prompt tables."""
    side_counts = Counter(row["side"] for row in actor_rows)
    frame_counts = Counter(row["final_frame"] for row in actor_rows)
    paired_both = sum(row["has_simple"] == 1 and row["has_normal"] == 1 for row in paired_rows)
    t2i_default = sum(int(row["include_in_t2i_default"]) for row in actor_rows)

    print(f"Actor rows:              {len(actor_rows)}")
    print(f"Paired actor rows:       {len(paired_rows)}")
    print(f"Paired on both sides:    {paired_both}")
    print(f"Include in T2I default:  {t2i_default}")
    print(f"By side:                 {dict(side_counts)}")
    print(f"By final_frame:          {dict(frame_counts.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build actor-level prompt tables from final gender annotations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paired-output", type=Path, default=DEFAULT_PAIRED_OUTPUT)
    args = parser.parse_args()

    final_rows = load_csv_rows(args.input)
    pair_metadata = load_pair_metadata(args.pairs)

    actor_rows = build_actor_rows(final_rows, pair_metadata)
    paired_rows = build_paired_rows(actor_rows)

    if actor_rows:
        write_csv(args.output, actor_rows, list(actor_rows[0].keys()))
    if paired_rows:
        write_csv(args.paired_output, paired_rows, list(paired_rows[0].keys()))

    print_summary(actor_rows, paired_rows)
    print(f"Actor prompt CSV:        {args.output}")
    print(f"Paired actor prompt CSV: {args.paired_output}")


if __name__ == "__main__":
    main()
