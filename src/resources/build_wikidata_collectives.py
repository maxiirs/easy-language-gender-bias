#!/usr/bin/env python3
"""Build a local Wikidata cache for collective human nouns in the corpus.

The script reads noun lemmas from the local Kaikki cache, queries Wikidata for
exact German label matches, and stores positive collective matches in a compact
CSV file for reproducible downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

DEFAULT_INPUT = Path("data/resources/kaikki_de_lookup.jsonl")
DEFAULT_OUTPUT = Path("data/resources/wikidata_collective_nouns.csv")
DEFAULT_METADATA = Path("data/resources/wikidata_collective_nouns_metadata.json")
DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_CHUNK_SIZE = 20

DIRECT_COLLECTIVE_CLASS_QIDS = {
    "Q16334295",  # group of humans
    "Q874405",    # social group
    "Q1752346",   # formal assembly
    "Q2570643",   # senate
    "Q637846",    # upper house
    "Q640506",    # cabinet
}
COLLECTIVE_CLASS_LABEL_RE = re.compile(
    r"\b(grupp|versammlung|senat|oberhaus|kabinett|gremium)\b",
    re.IGNORECASE,
)
COLLECTIVE_ITEM_DESC_RE = re.compile(
    r"\b(personen|parlament|gremium|regierung|lebensgemeinschaft|zusammenschluss)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize text for robust lemma matching."""
    return " ".join(text.strip().split()).lower()


def load_candidate_lemmas(path: Path) -> list[str]:
    """Collect noun lemmas from the local Kaikki cache."""
    lemmas: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("lang") != "Deutsch" or entry.get("pos") != "noun":
                continue
            lemma = normalize_text(entry.get("word", ""))
            if lemma:
                lemmas.add(lemma)
    return sorted(lemmas)


def chunked(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    return [items[index:index + size] for index in range(0, len(items), size)]


def build_query(lemmas: list[str]) -> str:
    """Build one SPARQL query for a batch of exact German labels."""
    lemma_values = " ".join(json.dumps(lemma, ensure_ascii=False) for lemma in lemmas)
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?lemma_text ?item ?itemLabel ?itemDescription ?class ?classLabel WHERE {{
  VALUES ?lemma_text {{ {lemma_values} }}
  ?item rdfs:label ?lemma_label .
  FILTER(LANG(?lemma_label) = "de")
  FILTER(LCASE(STR(?lemma_label)) = ?lemma_text)
  ?item wdt:P31 ?class .
  FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q4167410 }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de,en". }}
}}
ORDER BY ?lemma_text ?item ?class
""".strip()


def run_query(endpoint: str, query: str, max_retries: int = 3, timeout: int = 30) -> dict:
    """Execute a SPARQL query and return JSON results."""
    url = f"{endpoint}?{urllib.parse.urlencode({'query': query, 'format': 'json'})}"
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "easy-language-gender-bias/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if attempt >= max_retries:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Unreachable retry loop")


def extract_qid(entity_url: str) -> str:
    """Extract a Wikidata QID from a full entity URL."""
    return entity_url.rsplit("/", 1)[-1]


def is_collective_match(class_qid: str, class_label: str, item_description: str) -> bool:
    """Decide whether a Wikidata item denotes a human collective."""
    class_label = normalize_text(class_label)
    item_description = normalize_text(item_description)
    if class_qid in DIRECT_COLLECTIVE_CLASS_QIDS:
        return True
    if COLLECTIVE_CLASS_LABEL_RE.search(class_label):
        return True
    if class_label == "organisation" and COLLECTIVE_ITEM_DESC_RE.search(item_description):
        return True
    return False


def collect_matches(lemmas: list[str], endpoint: str, chunk_size: int = 100) -> dict[str, dict]:
    """Query Wikidata in batches and aggregate positive collective matches."""
    matches: dict[str, dict] = {}
    batches = chunked(lemmas, chunk_size)
    for index, batch in enumerate(batches, start=1):
        print(f"Query batch {index}/{len(batches)} ({len(batch)} lemmas)...", flush=True)
        data = run_query(endpoint, build_query(batch))
        for row in data.get("results", {}).get("bindings", []):
            lemma = row["lemma_text"]["value"]
            item_qid = extract_qid(row["item"]["value"])
            class_qid = extract_qid(row["class"]["value"])
            class_label = row.get("classLabel", {}).get("value", "")
            description = row.get("itemDescription", {}).get("value", "")
            if not is_collective_match(class_qid, class_label, description):
                continue
            record = matches.setdefault(
                lemma,
                {
                    "lemma": lemma,
                    "is_collective": "1",
                    "item_ids": set(),
                    "item_labels": set(),
                    "item_descriptions": set(),
                    "class_qids": set(),
                    "class_labels": set(),
                },
            )
            record["item_ids"].add(item_qid)
            record["item_labels"].add(row.get("itemLabel", {}).get("value", ""))
            if description:
                record["item_descriptions"].add(description)
            record["class_qids"].add(class_qid)
            record["class_labels"].add(class_label)
        time.sleep(0.1)
    return matches


def write_matches(path: Path, matches: dict[str, dict]) -> None:
    """Write positive collective matches to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lemma",
        "is_collective",
        "item_ids",
        "item_labels",
        "item_descriptions",
        "class_qids",
        "class_labels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for lemma in sorted(matches):
            row = matches[lemma].copy()
            for key in fieldnames[2:]:
                row[key] = " | ".join(sorted(row[key]))
            writer.writerow(row)


def write_metadata(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    endpoint: str,
    candidate_count: int,
    matched_lemma_count: int,
) -> None:
    """Write metadata for reproducibility."""
    metadata = {
        "source": "wikidata",
        "endpoint": endpoint,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_cache": str(input_path),
        "output_cache": str(output_path),
        "candidate_lemma_count": candidate_count,
        "matched_collective_lemma_count": matched_lemma_count,
        "direct_collective_class_qids": sorted(DIRECT_COLLECTIVE_CLASS_QIDS),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local Wikidata cache for collective human nouns."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    lemmas = load_candidate_lemmas(args.input)
    print(f"Candidate lemmas: {len(lemmas)}")

    matches = collect_matches(lemmas, args.endpoint, args.chunk_size)
    write_matches(args.output, matches)
    write_metadata(
        args.metadata,
        input_path=args.input,
        output_path=args.output,
        endpoint=args.endpoint,
        candidate_count=len(lemmas),
        matched_lemma_count=len(matches),
    )

    print(f"Matched collective lemmas: {len(matches)}")
    print(f"Collective cache:          {args.output}")
    print(f"Metadata:                  {args.metadata}")


if __name__ == "__main__":
    main()
