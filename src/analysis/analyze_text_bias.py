#!/usr/bin/env python3
"""Analyse actor-level text bias in Easy Language vs Standard German.

The script operates on actor-level prompt tables and compares the distribution
of actor classes between the two language varieties. It mirrors the image-side
analysis style so that text and image bias can later be interpreted together.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scipy.stats import chi2_contingency

    SCIPY = True
except ImportError:
    SCIPY = False
    print("[WARN] scipy not installed – Chi-Square tests will be skipped.")
    print("       Install with: pip install scipy\n")

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    PLOTTING = True
except ImportError:
    PLOTTING = False
    print("[WARN] matplotlib/numpy not installed – plots will not be generated.")
    print("       Install with: pip install matplotlib numpy\n")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTORS = PROJECT_ROOT / "data" / "processed" / "actor_prompts.csv"
DEFAULT_PAIRED = PROJECT_ROOT / "data" / "processed" / "actor_prompts_paired.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "text_bias_analysis.txt"
PLOTS_DIR = PROJECT_ROOT / "data" / "processed" / "plots"
TABLES_DIR = PROJECT_ROOT / "data" / "processed"

SIDES = ["simple", "normal"]
SIDE_LABELS = {"simple": "Leichte Sprache", "normal": "Standard Deutsch"}
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


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    by_side: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_side[row["side"]].append(row)
    return by_side


def separator(char: str = "─", width: int = 76) -> str:
    return char * width


def section_header(title: str) -> str:
    line = separator()
    return f"\n{line}\n  {title}\n{line}\n"


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * n / total:5.1f}%"


def ordered_counts(rows: list[dict], key: str, order: list[str]) -> Counter:
    counts = Counter(row.get(key, "") for row in rows)
    for label in order:
        counts.setdefault(label, 0)
    return counts


def contingency_table(simple_counts: Counter, normal_counts: Counter, labels: list[str]) -> list[list[int]]:
    return [
        [simple_counts.get(label, 0) for label in labels],
        [normal_counts.get(label, 0) for label in labels],
    ]


def write_count_table(path: Path, labels: list[str], simple_counts: Counter, normal_counts: Counter) -> None:
    rows = []
    simple_total = sum(simple_counts.values())
    normal_total = sum(normal_counts.values())
    for label in labels:
        rows.append(
            {
                "label": label,
                "simple_count": simple_counts.get(label, 0),
                "simple_share": round(simple_counts.get(label, 0) / simple_total, 6) if simple_total else 0.0,
                "normal_count": normal_counts.get(label, 0),
                "normal_share": round(normal_counts.get(label, 0) / normal_total, 6) if normal_total else 0.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dataset_overview(actor_rows: list[dict], by_side: dict[str, list[dict]], paired_rows: list[dict], out: list[str]) -> None:
    out.append(section_header("1. Dataset Overview"))
    out.append(f"  Total actor rows: {len(actor_rows)}")
    out.append(f"  Paired actor rows: {len(paired_rows)}")
    both_sides = sum(1 for row in paired_rows if row.get('has_simple') == '1' and row.get('has_normal') == '1')
    out.append(f"  Paired actor rows on both sides: {both_sides}")
    out.append("")
    out.append(f"  {'Side':<18} {'Actors':>8} {'Named':>8} {'Groups':>8} {'T2I default':>12}")
    out.append(f"  {separator('-', 60)}")
    for side in SIDES:
        rows = by_side.get(side, [])
        named = sum(1 for row in rows if row.get("is_named_person") == "1")
        groups = sum(1 for row in rows if row.get("is_group") == "1")
        t2i = sum(1 for row in rows if row.get("include_in_t2i_default") == "1")
        out.append(f"  {SIDE_LABELS[side]:<18} {len(rows):>8} {named:>8} {groups:>8} {t2i:>12}")
    out.append("")


def distribution_section(by_side: dict[str, list[dict]], key: str, labels: list[str], title: str, out: list[str]) -> tuple[Counter, Counter]:
    out.append(section_header(title))
    out.append(f"  {'Category':<28} {'Leichte Sprache':>20}   {'Standard Dt.':>20}")
    out.append(f"  {separator('-', 72)}")

    simple_counts = ordered_counts(by_side.get("simple", []), key, labels)
    normal_counts = ordered_counts(by_side.get("normal", []), key, labels)
    simple_total = len(by_side.get("simple", []))
    normal_total = len(by_side.get("normal", []))

    for label in labels:
        s = simple_counts.get(label, 0)
        n = normal_counts.get(label, 0)
        out.append(f"  {label:<28} {s:>6} ({pct(s, simple_total)})   {n:>6} ({pct(n, normal_total)})")
    out.append("")
    return simple_counts, normal_counts


def chi_square_section(simple_counts: Counter, normal_counts: Counter, labels: list[str], title: str, out: list[str]) -> None:
    out.append(section_header(title))
    active_labels = [
        label for label in labels
        if simple_counts.get(label, 0) + normal_counts.get(label, 0) > 0
    ]
    table = contingency_table(simple_counts, normal_counts, active_labels)
    out.append("  Contingency table (rows = LS / StD):")
    header = "  {:<4} ".format("") + "  ".join(f"{label[:14]:>14}" for label in active_labels)
    out.append(header)
    for row_label, row in zip(["LS", "StD"], table):
        out.append("  {:<4} ".format(row_label) + "  ".join(f"{val:>14}" for val in row))
    out.append("")
    if not SCIPY:
        out.append("  [scipy not available]\n")
        return
    if len(active_labels) < 2:
        out.append("  [not enough active categories for chi-square]\n")
        return
    chi2, p, dof, _ = chi2_contingency(table)
    sig = "*** (p<0.001)" if p < 0.001 else "** (p<0.01)" if p < 0.01 else "* (p<0.05)" if p < 0.05 else "(not significant)"
    out.append(f"  chi2 = {chi2:.4f}, df = {dof}, p = {p:.6f} {sig}\n")


def paired_comparison_section(paired_rows: list[dict], out: list[str]) -> None:
    out.append(section_header("5. Paired Actor Comparison"))
    paired = [row for row in paired_rows if row.get("has_simple") == "1" and row.get("has_normal") == "1"]
    if not paired:
        out.append("  No paired actor rows with both sides available.\n")
        return

    same_frame = sum(1 for row in paired if row.get("simple_final_frame") == row.get("normal_final_frame"))
    same_t2i = sum(1 for row in paired if row.get("simple_final_t2i_label") == row.get("normal_final_t2i_label"))

    out.append(f"  Paired actors with both sides: {len(paired)}")
    out.append(f"  Same final_frame: {same_frame} ({pct(same_frame, len(paired))})")
    out.append(f"  Same final_t2i_label: {same_t2i} ({pct(same_t2i, len(paired))})")
    out.append("")

    frame_shifts = Counter()
    t2i_shifts = Counter()
    for row in paired:
        simple_frame = row.get("simple_final_frame", "")
        normal_frame = row.get("normal_final_frame", "")
        simple_label = row.get("simple_final_t2i_label", "")
        normal_label = row.get("normal_final_t2i_label", "")
        if simple_frame != normal_frame:
            frame_shifts[f"{simple_frame} -> {normal_frame}"] += 1
        if simple_label != normal_label:
            t2i_shifts[f"{simple_label} -> {normal_label}"] += 1

    out.append("  Most frequent frame shifts (LS -> StD):")
    for shift, count in frame_shifts.most_common(10):
        out.append(f"    {shift}: {count}")
    if not frame_shifts:
        out.append("    none")
    out.append("")

    out.append("  Most frequent T2I-label shifts (LS -> StD):")
    for shift, count in t2i_shifts.most_common(10):
        out.append(f"    {shift}: {count}")
    if not t2i_shifts:
        out.append("    none")
    out.append("")


def t2i_subset_section(by_side: dict[str, list[dict]], out: list[str]) -> None:
    out.append(section_header("6. T2I Main-Set Composition"))
    subset_by_side = {
        side: [row for row in rows if row.get("include_in_t2i_default") == "1"]
        for side, rows in by_side.items()
    }
    out.append(f"  {'Side':<18} {'Actors':>8} {'f1':>6} {'m1':>6} {'fg':>6} {'mg':>6} {'mx':>6} {'ux':>6}")
    out.append(f"  {separator('-', 72)}")
    for side in SIDES:
        rows = subset_by_side.get(side, [])
        counts = ordered_counts(rows, "final_t2i_label", T2I_LABEL_ORDER)
        out.append(
            f"  {SIDE_LABELS[side]:<18} {len(rows):>8} "
            f"{counts['f1']:>6} {counts['m1']:>6} {counts['fg']:>6} "
            f"{counts['mg']:>6} {counts['mx']:>6} {counts['ux']:>6}"
        )
    out.append("")


def plot_grouped_distribution(labels: list[str], simple_counts: Counter, normal_counts: Counter, title: str, out_path: Path) -> None:
    if not PLOTTING:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    simple_total = sum(simple_counts.values())
    normal_total = sum(normal_counts.values())
    simple_vals = [simple_counts.get(label, 0) / simple_total if simple_total else 0 for label in labels]
    normal_vals = [normal_counts.get(label, 0) / normal_total if normal_total else 0 for label in labels]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, simple_vals, width, label="Leichte Sprache", color=COLORS["simple"])
    ax.bar(x + width / 2, normal_vals, width, label="Standard Deutsch", color=COLORS["normal"])
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
    if not PLOTTING:
        return
    paired = [row for row in paired_rows if row.get("has_simple") == "1" and row.get("has_normal") == "1"]
    if not paired:
        return

    matrix = np.zeros((len(T2I_LABEL_ORDER), len(T2I_LABEL_ORDER)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(T2I_LABEL_ORDER)}
    for row in paired:
        s = row.get("simple_final_t2i_label", "")
        n = row.get("normal_final_t2i_label", "")
        if s in label_to_idx and n in label_to_idx:
            matrix[label_to_idx[s], label_to_idx[n]] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(T2I_LABEL_ORDER)))
    ax.set_yticks(np.arange(len(T2I_LABEL_ORDER)))
    ax.set_xticklabels(T2I_LABEL_ORDER)
    ax.set_yticklabels(T2I_LABEL_ORDER)
    ax.set_xlabel("Standard Deutsch")
    ax.set_ylabel("Leichte Sprache")
    ax.set_title("Paired actor label transitions")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse actor-level text bias between Easy Language and Standard German.")
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS, help=f"Path to actor_prompts.csv (default: {DEFAULT_ACTORS})")
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED, help=f"Path to actor_prompts_paired.csv (default: {DEFAULT_PAIRED})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"Path to text report (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    actor_rows = load_csv(args.actors)
    paired_rows = load_csv(args.paired)
    by_side = partition(actor_rows)

    out: list[str] = []
    out.append("TEXT BIAS ANALYSIS\n")
    out.append("Actor-level comparison of Easy Language vs Standard German.\n")

    dataset_overview(actor_rows, by_side, paired_rows, out)

    frame_simple, frame_normal = distribution_section(by_side, "final_frame", FRAME_ORDER, "2. Final-Frame Distribution", out)
    chi_square_section(frame_simple, frame_normal, FRAME_ORDER, "3. Chi-Square Test: Final Frames", out)

    t2i_simple, t2i_normal = distribution_section(by_side, "final_t2i_label", T2I_LABEL_ORDER, "4. T2I-Label Distribution", out)
    chi_square_section(t2i_simple, t2i_normal, T2I_LABEL_ORDER, "4b. Chi-Square Test: T2I Labels", out)

    paired_comparison_section(paired_rows, out)
    t2i_subset_section(by_side, out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote report to {args.output}")

    write_count_table(TABLES_DIR / "text_bias_final_frame_counts.csv", FRAME_ORDER, frame_simple, frame_normal)
    write_count_table(TABLES_DIR / "text_bias_t2i_label_counts.csv", T2I_LABEL_ORDER, t2i_simple, t2i_normal)
    print(f"Wrote tables to {TABLES_DIR / 'text_bias_final_frame_counts.csv'} and {TABLES_DIR / 'text_bias_t2i_label_counts.csv'}")

    if PLOTTING:
        plot_grouped_distribution(
            FRAME_ORDER,
            frame_simple,
            frame_normal,
            "Actor final-frame distribution by language side",
            PLOTS_DIR / "text_bias_final_frame_distribution.png",
        )
        plot_grouped_distribution(
            T2I_LABEL_ORDER,
            t2i_simple,
            t2i_normal,
            "Actor T2I-label distribution by language side",
            PLOTS_DIR / "text_bias_t2i_label_distribution.png",
        )
        plot_paired_heatmap(
            paired_rows,
            PLOTS_DIR / "text_bias_paired_t2i_heatmap.png",
        )
        print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
