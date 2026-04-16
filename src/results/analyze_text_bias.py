#!/usr/bin/env python3
"""Compute structured text-side actor bias results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scipy.stats import chi2_contingency
except ImportError:  # pragma: no cover - optional dependency
    chi2_contingency = None

try:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = np = plt = mticker = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTORS = PROJECT_ROOT / "data" / "processed" / "actor_prompts.csv"
DEFAULT_PAIRED = PROJECT_ROOT / "data" / "processed" / "actor_prompts_paired.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "results" / "text_bias"

SIDES = ["simple", "normal"]
SIDE_LABELS = {"simple": "Easy Language", "normal": "Standard German"}
COLORS = {"simple": "#5B8DB8", "normal": "#E07B54"}

FRAME_ORDER = [
    "fem",
    "male_specific",
    "gmasc_generic",
    "epic_individual",
    "epic_group",
    "pair",
    "named_person",
    "unk",
]
T2I_LABEL_ORDER = ["f1", "m1", "fg", "mg", "mx", "ux"]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    by_side: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_side[row["side"]].append(row)
    return by_side


def ordered_counts(rows: list[dict], key: str, order: list[str]) -> Counter:
    counts = Counter(row.get(key, "") for row in rows)
    for label in order:
        counts.setdefault(label, 0)
    return counts


def count_rows(labels: list[str], simple_counts: Counter, normal_counts: Counter) -> list[dict]:
    simple_total = sum(simple_counts.values())
    normal_total = sum(normal_counts.values())
    rows = []
    for label in labels:
        simple_count = simple_counts.get(label, 0)
        normal_count = normal_counts.get(label, 0)
        rows.append(
            {
                "label": label,
                "simple_count": simple_count,
                "simple_share": round(simple_count / simple_total, 6) if simple_total else 0.0,
                "normal_count": normal_count,
                "normal_share": round(normal_count / normal_total, 6) if normal_total else 0.0,
            }
        )
    return rows


def chi_square(labels: list[str], simple_counts: Counter, normal_counts: Counter) -> dict:
    active = [label for label in labels if simple_counts.get(label, 0) + normal_counts.get(label, 0) > 0]
    table = [[simple_counts.get(label, 0) for label in active], [normal_counts.get(label, 0) for label in active]]
    result = {"labels": active, "table": table, "available": chi2_contingency is not None}
    if chi2_contingency is None or len(active) < 2:
        return result
    chi2, p, dof, expected = chi2_contingency(table)
    result.update(
        {
            "chi2": float(chi2),
            "p": float(p),
            "dof": int(dof),
            "expected": [[float(value) for value in row] for row in expected],
        }
    )
    return result


def paired_shifts(paired_rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    paired = [row for row in paired_rows if row.get("has_simple") == "1" and row.get("has_normal") == "1"]
    same_frame = sum(1 for row in paired if row.get("simple_final_frame") == row.get("normal_final_frame"))
    same_t2i = sum(1 for row in paired if row.get("simple_final_t2i_label") == row.get("normal_final_t2i_label"))
    frame_shifts = Counter()
    t2i_shifts = Counter()
    for row in paired:
        simple_frame = row.get("simple_final_frame", "")
        normal_frame = row.get("normal_final_frame", "")
        simple_label = row.get("simple_final_t2i_label", "")
        normal_label = row.get("normal_final_t2i_label", "")
        if simple_frame != normal_frame:
            frame_shifts[(simple_frame, normal_frame)] += 1
        if simple_label != normal_label:
            t2i_shifts[(simple_label, normal_label)] += 1
    summary = {
        "paired_actors_with_both_sides": len(paired),
        "same_final_frame": same_frame,
        "same_final_frame_share": round(same_frame / len(paired), 6) if paired else 0.0,
        "same_final_t2i_label": same_t2i,
        "same_final_t2i_label_share": round(same_t2i / len(paired), 6) if paired else 0.0,
    }
    frame_rows = [{"simple_final_frame": s, "normal_final_frame": n, "count": count} for (s, n), count in frame_shifts.items()]
    t2i_rows = [{"simple_final_t2i_label": s, "normal_final_t2i_label": n, "count": count} for (s, n), count in t2i_shifts.items()]
    return summary, sorted(frame_rows, key=lambda row: row["count"], reverse=True), sorted(t2i_rows, key=lambda row: row["count"], reverse=True)


def t2i_main_set_rows(by_side: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for side in SIDES:
        subset = [row for row in by_side.get(side, []) if row.get("include_in_t2i_default") == "1"]
        counts = ordered_counts(subset, "final_t2i_label", T2I_LABEL_ORDER)
        record = {"side": side, "total": len(subset)}
        record.update({label: counts[label] for label in T2I_LABEL_ORDER})
        rows.append(record)
    return rows


def plot_grouped_distribution(labels: list[str], simple_counts: Counter, normal_counts: Counter, title: str, out_path: Path) -> None:
    if plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    simple_total = sum(simple_counts.values())
    normal_total = sum(normal_counts.values())
    simple_vals = [simple_counts.get(label, 0) / simple_total if simple_total else 0 for label in labels]
    normal_vals = [normal_counts.get(label, 0) / normal_total if normal_total else 0 for label in labels]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, simple_vals, width, label=SIDE_LABELS["simple"], color=COLORS["simple"])
    ax.bar(x + width / 2, normal_vals, width, label=SIDE_LABELS["normal"], color=COLORS["normal"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of actors")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_paired_heatmap(paired_rows: list[dict], out_path: Path) -> None:
    if plt is None:
        return
    paired = [row for row in paired_rows if row.get("has_simple") == "1" and row.get("has_normal") == "1"]
    if not paired:
        return
    matrix = np.zeros((len(T2I_LABEL_ORDER), len(T2I_LABEL_ORDER)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(T2I_LABEL_ORDER)}
    for row in paired:
        simple = row.get("simple_final_t2i_label", "")
        normal = row.get("normal_final_t2i_label", "")
        if simple in label_to_idx and normal in label_to_idx:
            matrix[label_to_idx[simple], label_to_idx[normal]] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(T2I_LABEL_ORDER)))
    ax.set_yticks(np.arange(len(T2I_LABEL_ORDER)))
    ax.set_xticklabels(T2I_LABEL_ORDER)
    ax.set_yticklabels(T2I_LABEL_ORDER)
    ax.set_xlabel(SIDE_LABELS["normal"])
    ax.set_ylabel(SIDE_LABELS["simple"])
    ax.set_title("Paired actor label transitions")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    actor_rows = load_csv(args.actors)
    paired_rows = load_csv(args.paired)
    by_side = partition(actor_rows)

    frame_simple = ordered_counts(by_side.get("simple", []), "final_frame", FRAME_ORDER)
    frame_normal = ordered_counts(by_side.get("normal", []), "final_frame", FRAME_ORDER)
    t2i_simple = ordered_counts(by_side.get("simple", []), "final_t2i_label", T2I_LABEL_ORDER)
    t2i_normal = ordered_counts(by_side.get("normal", []), "final_t2i_label", T2I_LABEL_ORDER)
    paired_summary, frame_shift_rows, t2i_shift_rows = paired_shifts(paired_rows)

    summary = {
        "input_files": {"actors": str(args.actors), "paired": str(args.paired)},
        "dataset": {
            "actor_rows": len(actor_rows),
            "paired_actor_rows": len(paired_rows),
            "paired_actor_rows_with_both_sides": paired_summary["paired_actors_with_both_sides"],
            "by_side": {
                side: {
                    "actors": len(by_side.get(side, [])),
                    "named_persons": sum(1 for row in by_side.get(side, []) if row.get("is_named_person") == "1"),
                    "groups": sum(1 for row in by_side.get(side, []) if row.get("is_group") == "1"),
                    "included_in_t2i_default": sum(1 for row in by_side.get(side, []) if row.get("include_in_t2i_default") == "1"),
                }
                for side in SIDES
            },
        },
        "chi_square": {
            "final_frame": chi_square(FRAME_ORDER, frame_simple, frame_normal),
            "final_t2i_label": chi_square(T2I_LABEL_ORDER, t2i_simple, t2i_normal),
        },
        "paired": paired_summary,
    }

    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "final_frame_counts.csv", count_rows(FRAME_ORDER, frame_simple, frame_normal))
    write_csv(out_dir / "t2i_label_counts.csv", count_rows(T2I_LABEL_ORDER, t2i_simple, t2i_normal))
    write_csv(out_dir / "t2i_main_set_counts.csv", t2i_main_set_rows(by_side))
    write_csv(out_dir / "paired_frame_shifts.csv", frame_shift_rows)
    write_csv(out_dir / "paired_t2i_label_shifts.csv", t2i_shift_rows)

    plot_grouped_distribution(FRAME_ORDER, frame_simple, frame_normal, "Actor final-frame distribution by language side", plots_dir / "final_frame_distribution.png")
    plot_grouped_distribution(T2I_LABEL_ORDER, t2i_simple, t2i_normal, "Actor T2I-label distribution by language side", plots_dir / "t2i_label_distribution.png")
    plot_paired_heatmap(paired_rows, plots_dir / "paired_t2i_heatmap.png")
    print(f"Wrote text-bias results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute structured actor-level text-bias results.")
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    run(parser.parse_args())
