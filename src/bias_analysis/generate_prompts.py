import argparse
import csv
from collections import defaultdict
from pathlib import Path


def get_prefix(actor: str) -> str:
    lower_actor = actor.lower()
    if lower_actor.endswith("innen") or lower_actor.endswith("kräfte") or lower_actor.endswith("leute") or lower_actor.endswith("gruppen"):
        return ""  # Plural has no indefinite article
    elif lower_actor.endswith("in") or lower_actor.endswith("frau"):
        return "Eine "  # Feminine singular
    elif lower_actor in ["personal", "publikum"]:
        return ""  # Uncountable
    elif lower_actor in ["polizei", "feuerwehr", "verwaltung"]:
        return "Die "  # Specific feminine institutions
    else:
        return "Ein "  # Masculine/Neuter singular generic fallback


def build_prompts(row: dict) -> tuple[str, str]:
    """Returns (prompt_lexical_baseline, prompt_minimal_context) for a row."""
    actor = row.get("actor_lemma", row.get("actor_label", "")).strip()
    if actor:
        actor = actor[0].upper() + actor[1:]

    predications = row["predications"].split(" || ")
    sentence = predications[0].strip() if predications else ""

    prefix = get_prefix(actor)
    prompt_lexical = f"{prefix}{actor}.".strip() if actor else ""

    if actor and sentence:
        prompt_context = f"{prefix}{actor}. {sentence}".strip()
    elif actor:
        prompt_context = f"{prefix}{actor}.".strip()
    else:
        prompt_context = sentence

    return prompt_lexical, prompt_context


def generate_flat(rows: list[dict], output_path: Path):
    """Original flat actor-level CSV with prompt columns appended."""
    fieldnames = list(rows[0].keys())
    if "prompt_lexical_baseline" not in fieldnames:
        fieldnames.append("prompt_lexical_baseline")
    if "prompt_minimal_context" not in fieldnames:
        fieldnames.append("prompt_minimal_context")

    for row in rows:
        row["prompt_lexical_baseline"], row["prompt_minimal_context"] = build_prompts(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Flat CSV:   {len(rows)} actors → {output_path}")


def generate_paired(rows: list[dict], output_path: Path):
    """
    Paired topic-level CSV with one row per pair_id.
    Each row contains ';'-separated lists of actors and prompts for both sides.
    This CSV is the direct input for generate_t2i_images.py.
    """
    # Ensure prompts are already built on the rows
    for row in rows:
        if "prompt_minimal_context" not in row:
            row["prompt_lexical_baseline"], row["prompt_minimal_context"] = build_prompts(row)

    # Group by pair_id, then by side
    pairs: dict[str, dict] = defaultdict(lambda: {
        "topic": "", "source": "",
        "simple": [], "normal": []
    })

    for row in rows:
        pid = row["pair_id"]
        pairs[pid]["topic"] = row.get("topic", "")
        pairs[pid]["source"] = row.get("source", "")
        side = row["side"]
        pairs[pid][side].append({
            "actor_id": row.get("actor_id", ""),
            "actor_label": row.get("actor_label", ""),
            "actor_lemma": row.get("actor_lemma", row.get("actor_label", "")),
            "predicted_gender": row.get("predicted_gender", ""),
            "prompt_lexical": row.get("prompt_lexical_baseline", ""),
            "prompt_context": row.get("prompt_minimal_context", ""),
        })

    fieldnames = [
        "pair_id", "topic", "source",
        "simple_actor_count", "normal_actor_count",
        "simple_actor_ids", "simple_actor_labels", "simple_predicted_genders", "simple_prompts_lexical", "simple_prompts_context",
        "normal_actor_ids", "normal_actor_labels", "normal_predicted_genders", "normal_prompts_lexical", "normal_prompts_context",
    ]

    paired_rows = []
    for pid, data in sorted(pairs.items()):
        simple = data["simple"]
        normal = data["normal"]

        def join(lst, key): return ";".join(str(d[key]) for d in lst)

        paired_rows.append({
            "pair_id": pid,
            "topic": data["topic"],
            "source": data["source"],
            "simple_actor_count": len(simple),
            "normal_actor_count": len(normal),
            "simple_actor_ids":         join(simple, "actor_id"),
            "simple_actor_labels":      join(simple, "actor_lemma"),
            "simple_predicted_genders": join(simple, "predicted_gender"),
            "simple_prompts_lexical":   join(simple, "prompt_lexical"),
            "simple_prompts_context":   join(simple, "prompt_context"),
            "normal_actor_ids":         join(normal, "actor_id"),
            "normal_actor_labels":      join(normal, "actor_lemma"),
            "normal_predicted_genders": join(normal, "predicted_gender"),
            "normal_prompts_lexical":   join(normal, "prompt_lexical"),
            "normal_prompts_context":   join(normal, "prompt_context"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    both_sides = sum(1 for r in paired_rows if r["simple_actor_count"] > 0 and r["normal_actor_count"] > 0)
    print(f"Paired CSV: {len(paired_rows)} topic pairs ({both_sides} with actors on both sides) → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate text-to-image prompts from extracted actors.")
    parser.add_argument("--input",  type=Path, default=Path("data/processed/actor_features.csv"), help="Input CSV.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/actor_prompts.csv"), help="Flat output CSV.")
    parser.add_argument("--paired-output", type=Path, default=Path("data/processed/actor_prompts_paired.csv"), help="Paired output CSV.")

    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data found in the input CSV.")
        return

    generate_flat(rows, args.output)
    generate_paired(rows, args.paired_output)

if __name__ == "__main__":
    main()
