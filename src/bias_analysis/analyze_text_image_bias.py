#!/usr/bin/env python3
"""Compare text-side actor categories with image-side MiVOLO outcomes.

The script focuses on the transformation from text category to image category:
- Which image categories emerge from a given text category?
- Does this transformation differ between Easy Language and Standard German?
- Where do the largest divergences between text intention and image output occur?
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
DEFAULT_IN = PROJECT_ROOT / "data" / "processed" / "t2i_mivolo_results.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "text_image_bias_analysis.txt"
PLOTS_DIR = PROJECT_ROOT / "data" / "processed" / "plots"
TABLES_DIR = PROJECT_ROOT / "data" / "processed"

SIDES = ["simple", "normal"]
SIDE_LABELS = {"simple": "Leichte Sprache", "normal": "Standard Deutsch"}
COLORS = {"simple": "#5B8DB8", "normal": "#E07B54"}

IMAGE_CATEGORIES = [
    "weiblich (Singular)",
    "maskulin (Singular)",
    "Frauengruppe",
    "Männergruppe",
    "gemischte/generische Gruppe",
    "Keine Person",
]

TEXT_CATEGORY_MAP = {
    "f1": "weiblich (Singular)",
    "m1": "maskulin (Singular)",
    "fg": "Frauengruppe",
    "mg": "Männergruppe",
    "mx": "gemischte/generische Gruppe",
    "ux": "gemischte/generische Gruppe",
    "weiblich (Singular)": "weiblich (Singular)",
    "maskulin (Singular)": "maskulin (Singular)",
    "Frauengruppe": "Frauengruppe",
    "Männergruppe": "Männergruppe",
    "gemischte/generische Gruppe": "gemischte/generische Gruppe",
    "generisch maskulin (Singular)": "maskulin (Singular)",
}


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def separator(char: str = "─", width: int = 80) -> str:
    return char * width


def section_header(title: str) -> str:
    line = separator()
    return f"\n{line}\n  {title}\n{line}\n"


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * n / total:5.1f}%"


def canonical_text_category(row: dict) -> str:
    value = (row.get("text_intended_category") or row.get("final_t2i_label") or row.get("predicted_gender") or "").strip()
    return TEXT_CATEGORY_MAP.get(value, value)


def prepare_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        out = dict(row)
        out["text_category"] = canonical_text_category(row)
        prepared.append(out)
    return prepared


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    by_side: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_side[row["side"]].append(row)
    return by_side


def active_text_categories(rows: list[dict]) -> list[str]:
    cats = sorted({row["text_category"] for row in rows if row["text_category"]})
    ordered = [cat for cat in IMAGE_CATEGORIES if cat in cats]
    extras = [cat for cat in cats if cat not in ordered]
    return ordered + extras


def dataset_overview(rows: list[dict], by_side: dict[str, list[dict]], out: list[str]) -> list[str]:
    text_cats = active_text_categories(rows)
    out.append(section_header("1. Dataset Overview"))
    out.append(f"  Total MiVOLO result rows: {len(rows)}")
    out.append(f"  Text categories present: {', '.join(text_cats)}")
    out.append("")
    out.append(f"  {'Side':<18} {'Rows':>8} {'No person':>10} {'Semantic match':>16}")
    out.append(f"  {separator('-', 60)}")
    for side in SIDES:
        side_rows = by_side.get(side, [])
        no_person = sum(1 for row in side_rows if row.get("image_gender_category") == "Keine Person")
        matches = sum(1 for row in side_rows if row.get("text_category") == row.get("image_gender_category"))
        out.append(f"  {SIDE_LABELS[side]:<18} {len(side_rows):>8} {no_person:>10} {matches:>9} ({pct(matches, len(side_rows))})")
    out.append("")
    return text_cats


def text_category_distribution(by_side: dict[str, list[dict]], text_cats: list[str], out: list[str]) -> None:
    out.append(section_header("2. Text Category Composition of the Image Set"))
    out.append(f"  {'Text category':<32} {'Leichte Sprache':>20}   {'Standard Dt.':>20}")
    out.append(f"  {separator('-', 76)}")
    for cat in text_cats:
        s_rows = by_side.get("simple", [])
        n_rows = by_side.get("normal", [])
        s = sum(1 for row in s_rows if row["text_category"] == cat)
        n = sum(1 for row in n_rows if row["text_category"] == cat)
        out.append(f"  {cat:<32} {s:>6} ({pct(s, len(s_rows))})   {n:>6} ({pct(n, len(n_rows))})")
    out.append("")


def stratified_comparison(rows: list[dict], text_cats: list[str], out: list[str], exclude_no_person: bool) -> None:
    out.append(section_header("3. Image Distribution Within Text Categories"))
    out.append(
        "  For each text category, compares the resulting image-category distribution\n"
        "  between Leichte Sprache and Standard German. This is the key test for a\n"
        "  language-variety effect beyond differences in prompt composition.\n"
    )

    image_cats = [cat for cat in IMAGE_CATEGORIES if not (exclude_no_person and cat == "Keine Person")]
    for text_cat in text_cats:
        out.append(f"  Text category: {text_cat}")
        out.append(f"    {'Image category':<30} {'LS':>16}   {'StD':>16}")
        out.append(f"    {separator('-', 68)}")

        s_rows = [row for row in rows if row["side"] == "simple" and row["text_category"] == text_cat]
        n_rows = [row for row in rows if row["side"] == "normal" and row["text_category"] == text_cat]
        if exclude_no_person:
            s_rows = [row for row in s_rows if row.get("image_gender_category") != "Keine Person"]
            n_rows = [row for row in n_rows if row.get("image_gender_category") != "Keine Person"]

        s_counts = Counter(row.get("image_gender_category", "") for row in s_rows)
        n_counts = Counter(row.get("image_gender_category", "") for row in n_rows)
        for img_cat in image_cats:
            s = s_counts.get(img_cat, 0)
            n = n_counts.get(img_cat, 0)
            if s == 0 and n == 0:
                continue
            out.append(f"    {img_cat:<30} {s:>5} ({pct(s, len(s_rows))})   {n:>5} ({pct(n, len(n_rows))})")

        active_cats = [cat for cat in image_cats if s_counts.get(cat, 0) + n_counts.get(cat, 0) > 0]
        if SCIPY and len(active_cats) >= 2 and s_rows and n_rows:
            table = [
                [s_counts.get(cat, 0) for cat in active_cats],
                [n_counts.get(cat, 0) for cat in active_cats],
            ]
            chi2, p, dof, _ = chi2_contingency(table)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            out.append(f"    chi2 = {chi2:.3f}, df = {dof}, p = {p:.6f} {sig}")
        out.append("")


def semantic_shift_summary(rows: list[dict], text_cats: list[str], out: list[str], exclude_no_person: bool) -> None:
    out.append(section_header("4. Text-to-Image Shift Summary"))
    out.append(f"  {'Text category':<32} {'LS match':>16}   {'StD match':>16}")
    out.append(f"  {separator('-', 72)}")
    for text_cat in text_cats:
        line = [f"  {text_cat:<32}"]
        for side in SIDES:
            side_rows = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            if exclude_no_person:
                side_rows = [row for row in side_rows if row.get("image_gender_category") != "Keine Person"]
            total = len(side_rows)
            matches = sum(1 for row in side_rows if row["text_category"] == row.get("image_gender_category"))
            line.append(f"{matches:>5}/{total:<5} ({pct(matches, total)})")
        out.append("   ".join(line))
    out.append("")

    out.append("  Most common mismatches (text -> image, pooled across both sides):")
    shifts = Counter()
    for row in rows:
        if exclude_no_person and row.get("image_gender_category") == "Keine Person":
            continue
        text_cat = row["text_category"]
        image_cat = row.get("image_gender_category", "")
        if text_cat and image_cat and text_cat != image_cat:
            shifts[(text_cat, image_cat)] += 1
    for (text_cat, image_cat), count in shifts.most_common(15):
        out.append(f"    {text_cat} -> {image_cat}: {count}")
    out.append("")


def paired_side_by_side(rows: list[dict], out: list[str], exclude_no_person: bool) -> None:
    out.append(section_header("5. Paired Side-by-Side Comparison"))
    out.append(
        "  Compares Easy Language and Standard German for the same actor and seed,\n"
        "  whenever a stable pair identifier is available in the MiVOLO results.\n"
    )

    if not rows or "pair_actor_id" not in rows[0]:
        out.append("  No pair_actor_id column found. Paired actor-level comparison is not available in this file.\n")
        return

    index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["pair_actor_id"], row["seed"])
        index[key][row["side"]] = row
    both = {key: sides for key, sides in index.items() if "simple" in sides and "normal" in sides}
    if exclude_no_person:
        both = {
            key: sides for key, sides in both.items()
            if sides["simple"].get("image_gender_category") != "Keine Person"
            and sides["normal"].get("image_gender_category") != "Keine Person"
        }

    total = len(both)
    agree = sum(1 for sides in both.values() if sides["simple"].get("image_gender_category") == sides["normal"].get("image_gender_category"))
    out.append(f"  Matched actor+seed pairs: {total}")
    out.append(f"  Same image category: {agree} ({pct(agree, total)})")
    out.append(f"  Different image category: {total - agree} ({pct(total - agree, total)})")
    out.append("")

    divergence = Counter()
    for sides in both.values():
        s_cat = sides["simple"].get("image_gender_category")
        n_cat = sides["normal"].get("image_gender_category")
        if s_cat != n_cat:
            divergence[(s_cat, n_cat)] += 1
    out.append("  Most frequent divergence patterns (LS -> StD):")
    for (s_cat, n_cat), count in divergence.most_common(12):
        out.append(f"    {s_cat} -> {n_cat}: {count}")
    if not divergence:
        out.append("    none")
    out.append("")


def write_shift_table(path: Path, rows: list[dict], text_cats: list[str]) -> None:
    image_cats = IMAGE_CATEGORIES
    table_rows = []
    for side in SIDES:
        for text_cat in text_cats:
            subset = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            total = len(subset)
            counts = Counter(row.get("image_gender_category", "") for row in subset)
            record = {"side": side, "text_category": text_cat, "total": total}
            for image_cat in image_cats:
                record[image_cat] = counts.get(image_cat, 0)
                record[f"{image_cat}_share"] = round(counts.get(image_cat, 0) / total, 6) if total else 0.0
            table_rows.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)


def plot_text_to_image_heatmaps(rows: list[dict], text_cats: list[str], out_dir: Path, exclude_no_person: bool) -> None:
    if not PLOTTING:
        return

    image_cats = [cat for cat in IMAGE_CATEGORIES if not (exclude_no_person and cat == "Keine Person")]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(text_cats) * 0.8)), constrained_layout=True)
    if len(text_cats) == 1:
        axes = [axes]

    for ax, side in zip(axes, SIDES):
        matrix = []
        for text_cat in text_cats:
            subset = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            if exclude_no_person:
                subset = [row for row in subset if row.get("image_gender_category") != "Keine Person"]
            total = len(subset) or 1
            counts = Counter(row.get("image_gender_category", "") for row in subset)
            matrix.append([counts.get(image_cat, 0) / total for image_cat in image_cats])
        mat = np.array(matrix)
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(image_cats)))
        ax.set_xticklabels(image_cats, rotation=30, ha="right")
        ax.set_yticks(range(len(text_cats)))
        ax.set_yticklabels(text_cats)
        ax.set_title(SIDE_LABELS[side])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", color="black", fontsize=8)
    fig.suptitle("Text category -> image category", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, format=mticker.PercentFormatter(1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "text_image_heatmap_by_side.png", dpi=180)
    plt.close(fig)


def plot_match_rates(rows: list[dict], text_cats: list[str], out_dir: Path, exclude_no_person: bool) -> None:
    if not PLOTTING:
        return

    y = np.arange(len(text_cats))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(text_cats) * 0.7)))
    for idx, side in enumerate(SIDES):
        vals = []
        for text_cat in text_cats:
            subset = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            if exclude_no_person:
                subset = [row for row in subset if row.get("image_gender_category") != "Keine Person"]
            total = len(subset) or 1
            matches = sum(1 for row in subset if row["text_category"] == row.get("image_gender_category"))
            vals.append(matches / total)
        bars = ax.barh(y + idx * height - height / 2, vals, height, color=COLORS[side], label=SIDE_LABELS[side])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.0%}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(text_cats)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Text-image match rate by text category", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "text_image_match_rates.png", dpi=180)
    plt.close(fig)


def run(args) -> None:
    rows = prepare_rows(load_csv(Path(args.input)))
    by_side = partition(rows)
    text_cats = dataset_overview(rows, by_side, out := [])
    text_category_distribution(by_side, text_cats, out)
    stratified_comparison(rows, text_cats, out, args.exclude_no_person)
    semantic_shift_summary(rows, text_cats, out, args.exclude_no_person)
    paired_side_by_side(rows, out, args.exclude_no_person)

    report = "TEXT-IMAGE BIAS ANALYSIS\n\n" + "\n".join(out)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")

    shift_path = TABLES_DIR / "text_image_shift_table.csv"
    write_shift_table(shift_path, rows, text_cats)
    print(f"Wrote shift table to {shift_path}")

    if PLOTTING:
        plot_text_to_image_heatmaps(rows, text_cats, PLOTS_DIR, args.exclude_no_person)
        plot_match_rates(rows, text_cats, PLOTS_DIR, args.exclude_no_person)
        print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the transformation from text categories to image categories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN, help=f"Path to t2i_mivolo_results.csv (default: {DEFAULT_IN})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"Path to text-image report (default: {DEFAULT_OUT})")
    parser.add_argument("--exclude-no-person", action="store_true", help="Exclude 'Keine Person' rows from transformation analyses.")
    run(parser.parse_args())
