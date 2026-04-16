#!/usr/bin/env python3
"""Build a local Kaikki.org lookup cache for lemmas in annotated texts.

The script extracts the unique lemmas that occur in LLpro character mentions,
downloads the current German Kaikki raw data, filters it to the relevant
lemmas, and writes a compact local JSONL cache for reproducible downstream
analysis.
"""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/annotated_texts.json")
DEFAULT_OUTPUT = Path("data/resources/kaikki_de_lookup.jsonl")
DEFAULT_METADATA = Path("data/resources/kaikki_de_lookup_metadata.json")
DEFAULT_KAIKKI_URL = "https://kaikki.org/dewiktionary/raw-wiktextract-data.jsonl.gz"


def normalize_text(text: str) -> str:
    """Normalize text for robust lemma matching."""
    return " ".join(text.strip().split()).lower()


def get_head_token(character: dict) -> dict:
    """Pick the main content token of a character span."""
    tokens = character.get("tokens", [])
    for token in tokens:
        if token.get("pos") in {"NOUN", "PROPN", "PRON"}:
            return token
        if token.get("tag") == "NN":
            return token
    return tokens[0] if tokens else {}


def collect_target_lemmas(pairs: list[dict]) -> set[str]:
    """Collect unique candidate lemmas from character mentions."""
    lemmas: set[str] = set()
    for pair in pairs:
        for side in ("simple", "normal"):
            data = pair.get(side) or {}
            for character in data.get("characters", []):
                head = get_head_token(character)
                pos = head.get("pos", "")
                if pos not in {"NOUN", "PROPN"}:
                    continue
                lemma = normalize_text(head.get("lemma", ""))
                if lemma:
                    lemmas.add(lemma)
    return lemmas


def download_kaikki_dump(url: str) -> str:
    """Download the Kaikki raw JSONL dump into a temporary file."""
    handle = tempfile.NamedTemporaryFile(prefix="kaikki_de_", suffix=".jsonl.gz", delete=False)
    handle.close()
    urllib.request.urlretrieve(url, handle.name)
    return handle.name


def build_cache(
    dump_path: str,
    target_lemmas: set[str],
    output_path: Path,
) -> tuple[int, Counter]:
    """Filter the Kaikki dump to the target lemmas and write JSONL output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_count = 0
    pos_counts: Counter = Counter()

    with gzip.open(dump_path, "rt", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            entry = json.loads(line)
            if entry.get("lang") != "Deutsch":
                continue

            lemma = normalize_text(entry.get("word", ""))
            if lemma not in target_lemmas:
                continue

            pos = entry.get("pos", "")
            if pos not in {"noun", "name"}:
                continue

            target.write(json.dumps(entry, ensure_ascii=False) + "\n")
            match_count += 1
            pos_counts[pos] += 1

    return match_count, pos_counts


def write_metadata(
    path: Path,
    *,
    source_url: str,
    input_path: Path,
    output_path: Path,
    target_lemma_count: int,
    matched_entry_count: int,
    pos_counts: Counter,
) -> None:
    """Write metadata for reproducibility."""
    metadata = {
        "source": "kaikki.org",
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_annotations": str(input_path),
        "output_cache": str(output_path),
        "target_lemma_count": target_lemma_count,
        "matched_entry_count": matched_entry_count,
        "matched_pos_counts": dict(pos_counts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local Kaikki lookup cache for annotated text lemmas."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--kaikki-url", default=DEFAULT_KAIKKI_URL)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        pairs = json.load(handle)
    print(f"Loaded pairs: {len(pairs)}")

    target_lemmas = collect_target_lemmas(pairs)
    print(f"Target lemmas: {len(target_lemmas)}")

    dump_path = download_kaikki_dump(args.kaikki_url)
    print("Downloaded Kaikki raw data to a temporary file.")

    try:
        matched_entry_count, pos_counts = build_cache(
            dump_path=dump_path,
            target_lemmas=target_lemmas,
            output_path=args.output,
        )
    finally:
        Path(dump_path).unlink(missing_ok=True)

    write_metadata(
        path=args.metadata,
        source_url=args.kaikki_url,
        input_path=args.input,
        output_path=args.output,
        target_lemma_count=len(target_lemmas),
        matched_entry_count=matched_entry_count,
        pos_counts=pos_counts,
    )

    print(f"Matched entries: {matched_entry_count}")
    print(f"POS counts:       {dict(pos_counts)}")
    print(f"Lookup cache:     {args.output}")
    print(f"Metadata:         {args.metadata}")


if __name__ == "__main__":
    main()
