#!/usr/bin/env python3
"""
analyze_image_bias.py
----------------------
Evaluates MiVOLO classification results and compares image-level gender
distributions between Standard German ('normal') and German Easy Language
('simple') prompts, as described in the paper (§ Analysis of Effects).

Analyses performed
------------------
1.  No-person rate         – share of images with zero detected persons per side
2.  Category distribution  – frequency table of image_gender_category per side
3.  Chi-Square test        – tests whether the two distributions differ significantly
4.  Semantic alignment     – accuracy of image vs text_intended_category per side
5.  Visual collapse        – for each text category, shows which image categories
                             are generated most often (reveals stereotypical defaults)
6.  Paired comparison      – same pair_id + seed: did the two sides produce the same
                             visual category? How often did they diverge and why?
7.  Dominant-gender summary– breakdown of mivolo_gender (largest face bbox) per side

Usage
-----
    python src/bias_analysis/analyze_image_bias.py
    python src/bias_analysis/analyze_image_bias.py --input data/processed/t2i_mivolo_results.csv
    python src/bias_analysis/analyze_image_bias.py --exclude-no-person
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scipy.stats import chi2_contingency
    SCIPY = True
except ImportError:
    SCIPY = False
    print("[WARN] scipy not installed – Chi-Square p-values will not be shown.")
    print("       Install with: pip install scipy\n")

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    PLOTTING = True
except ImportError:
    PLOTTING = False
    print("[WARN] matplotlib/numpy not installed – plots will not be generated.")
    print("       Install with: pip install matplotlib numpy\n")

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN   = PROJECT_ROOT / "data" / "processed" / "t2i_mivolo_results.csv"
DEFAULT_OUT  = PROJECT_ROOT / "data" / "processed" / "image_bias_analysis.txt"

# Canonical category order (matches paper)
CATEGORIES = [
    "weiblich (Singular)",
    "maskulin (Singular)",
    "Frauengruppe",
    "Männergruppe",
    "gemischte/generische Gruppe",
    "Keine Person",
]

SIDES = ["simple", "normal"]   # simple = Leichte Sprache, normal = Standard German

SIDE_LABELS = {"simple": "Leichte Sprache", "normal": "Standard Deutsch"}

# Colour palette consistent across all plots
COLORS = {
    "simple": "#5B8DB8",   # blue  – Leichte Sprache
    "normal": "#E07B54",   # orange – Standard Deutsch
    "male":   "#4A90D9",
    "female": "#E05C8A",
    "none":   "#AAAAAA",
}

PLOTS_DIR = PROJECT_ROOT / "data" / "processed" / "plots"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    """Split rows by language side."""
    result = defaultdict(list)
    for row in rows:
        result[row["side"]].append(row)
    return result


def category_counts(rows: list[dict], col="image_gender_category") -> Counter:
    return Counter(r[col] for r in rows)


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * n / total:5.1f}%"


def separator(char="─", width=72) -> str:
    return char * width


def contingency_table(
    counts_simple: Counter, counts_normal: Counter, cats: list[str]
) -> list[list[int]]:
    """Build a 2×K contingency table [simple, normal] × categories."""
    return [
        [counts_simple.get(c, 0) for c in cats],
        [counts_normal.get(c, 0) for c in cats],
    ]


# ── Analysis sections ─────────────────────────────────────────────────────────

def section_header(title: str) -> str:
    line = separator()
    return f"\n{line}\n  {title}\n{line}\n"


def no_person_rate(by_side: dict, out: list[str]):
    out.append(section_header("1. No-Person Rate"))
    out.append(f"  {'Side':<10} {'Total':>8} {'No person':>12} {'Rate':>8}")
    out.append(f"  {'':<10} {'images':>8} {'images':>12} {'':>8}")
    out.append(f"  {separator('-', 48)}")
    for side in SIDES:
        rows   = by_side.get(side, [])
        total  = len(rows)
        n_none = sum(1 for r in rows if r["image_gender_category"] == "Keine Person")
        label  = "Leichte Sprache" if side == "simple" else "Standard Dt."
        out.append(f"  {label:<16} {total:>8} {n_none:>12}   {pct(n_none, total)}")
    out.append("")


def category_distribution(by_side: dict, out: list[str]):
    out.append(section_header("2. Image Gender Category Distribution"))

    # Header
    col_w = 32
    out.append(
        f"  {'Category':<{col_w}} "
        f"{'Leichte Sprache':>20}   {'Standard Dt.':>20}"
    )
    out.append(f"  {separator('-', col_w + 46)}")

    simple_counts = category_counts(by_side.get("simple", []))
    normal_counts = category_counts(by_side.get("normal", []))
    simple_total  = len(by_side.get("simple", []))
    normal_total  = len(by_side.get("normal", []))

    for cat in CATEGORIES:
        s = simple_counts.get(cat, 0)
        n = normal_counts.get(cat, 0)
        out.append(
            f"  {cat:<{col_w}} "
            f"{s:>6} ({pct(s, simple_total)})   "
            f"{n:>6} ({pct(n, normal_total)})"
        )
    out.append("")


def chi_square_test(by_side: dict, out: list[str], exclude_no_person: bool):
    out.append(section_header("3. Chi-Square Test of Independence"))

    cats = [c for c in CATEGORIES if not (exclude_no_person and c == "Keine Person")]

    # ── 3a. Overall test (naive) ──────────────────────────────────────────────
    out.append(
        "  3a. Overall test (naive)\n"
        "  WARNING: the two sides have slightly different text-category compositions\n"
        "  (LS: 76.4% vs StD: 73.8% generic masculine; LS: 9.0% vs StD: 10.6% weiblich).\n"
        "  If LS has more male-only prompts, more male images are expected regardless\n"
        "  of language style. This test does NOT control for that -- see §3b below.\n"
        "\n  H0: the overall visual distribution is the same for both sides.\n"
    )
    simple_counts = category_counts(by_side.get("simple", []))
    normal_counts = category_counts(by_side.get("normal", []))
    table_overall = contingency_table(simple_counts, normal_counts, cats)

    out.append("  Contingency table (rows = language side, cols = category):\n")
    header = "  {:<4} " + "  ".join(f"{c[:14]:>14}" for c in cats)
    out.append(header.format(""))
    for label, row in zip(["LS", "StD"], table_overall):
        out.append("  {:<4} ".format(label) + "  ".join(f"{v:>14}" for v in row))
    out.append("")

    if SCIPY:
        chi2, p, dof, _ = chi2_contingency(table_overall)
        out.append(f"  chi2 = {chi2:.4f},  df = {dof},  p = {p:.6f}  "
                   f"{'*** (p<0.001)' if p < 0.001 else '** (p<0.01)' if p < 0.01 else '* (p<0.05)' if p < 0.05 else '(not significant)'}\n")
    else:
        out.append("  [scipy not available]\n")

    # ── 3b. Stratified test per text_intended_category (correct approach) ────
    out.append(
        "  3b. Stratified test per text-intended category  (RECOMMENDED)\n"
        "  Compares LS vs StD within each text category separately -- controls for\n"
        "  prompt composition. Research question: given the same TYPE of actor prompt,\n"
        "  does the language variant (LS vs StD) change the visual output?\n"
    )

    all_text_cats = sorted(
        set(r["text_intended_category"] for rows in by_side.values() for r in rows)
    )

    col_w = 36
    out.append(f"  {'Text category':<{col_w}} {'chi2':>8}  {'df':>3}  {'p':>10}  {'sig':>6}  {'n_LS':>6}  {'n_StD':>6}")
    out.append(f"  {separator('-', col_w + 54)}")

    for txt_cat in all_text_cats:
        s_rows = [r for r in by_side.get("simple", []) if r["text_intended_category"] == txt_cat]
        n_rows = [r for r in by_side.get("normal", []) if r["text_intended_category"] == txt_cat]
        if exclude_no_person:
            s_rows = [r for r in s_rows if r["image_gender_category"] != "Keine Person"]
            n_rows = [r for r in n_rows if r["image_gender_category"] != "Keine Person"]

        if len(s_rows) == 0 or len(n_rows) == 0:
            out.append(f"  {txt_cat:<{col_w}} {'n/a':>8}  {'':>3}  {'':>10}  {'':>6}  {len(s_rows):>6}  {len(n_rows):>6}")
            continue

        s_counts = category_counts(s_rows)
        n_counts = category_counts(n_rows)
        # Keep only columns where at least one side has observations
        active_cats = [c for c in cats if s_counts.get(c, 0) + n_counts.get(c, 0) > 0]
        filtered_table = [
            [s_counts.get(c, 0) for c in active_cats],
            [n_counts.get(c, 0) for c in active_cats],
        ]

        if SCIPY and len(active_cats) >= 2:
            try:
                chi2, p, dof, _ = chi2_contingency(filtered_table)
                sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
                out.append(f"  {txt_cat:<{col_w}} {chi2:>8.3f}  {dof:>3}  {p:>10.6f}  {sig:>6}  {len(s_rows):>6}  {len(n_rows):>6}")
            except Exception as e:
                out.append(f"  {txt_cat:<{col_w}} {'error':>8}  {str(e)[:30]}")
        else:
            out.append(f"  {txt_cat:<{col_w}} {'[scipy missing or only 1 col]':>8}")

    out.append(
        "\n  A significant p within a stratum means the language variant changes the\n"
        "  visual output distribution even when prompt type is held constant.\n"
    )

def semantic_alignment(by_side: dict, out: list[str], exclude_no_person: bool):
    out.append(section_header("4. Semantic Alignment Accuracy"))
    out.append(
        "  Measures how often the generated image matches the text's\n"
        "  intended gender category (image_gender_category == text_intended_category).\n"
    )

    col_w = 36
    out.append(
        f"  {'Text intended category':<{col_w}} "
        f"{'Leichte Sprache':>22}   {'Standard Dt.':>22}"
    )
    out.append(f"  {separator('-', col_w + 50)}")

    # Gather all text categories present in the data
    all_text_cats = sorted(
        set(r["text_intended_category"] for rows in by_side.values() for r in rows)
    )

    for txt_cat in all_text_cats:
        row_parts = [f"  {txt_cat:<{col_w}}"]
        for side in SIDES:
            rows    = [r for r in by_side.get(side, []) if r["text_intended_category"] == txt_cat]
            if exclude_no_person:
                rows = [r for r in rows if r["image_gender_category"] != "Keine Person"]
            total   = len(rows)
            matched = sum(1 for r in rows if r["semantic_match"] == "True")
            row_parts.append(f" {matched:>5}/{total:<5} ({pct(matched, total)})")
        out.append("   ".join(row_parts))

    out.append("")

    # Overall accuracy
    out.append(f"  {'Overall':<{col_w}}", )
    for side in SIDES:
        rows    = by_side.get(side, [])
        if exclude_no_person:
            rows = [r for r in rows if r["image_gender_category"] != "Keine Person"]
        total   = len(rows)
        matched = sum(1 for r in rows if r["semantic_match"] == "True")
        out.append(f"    {('Leichte Sprache' if side == 'simple' else 'Standard Dt.')}: "
                   f"{matched}/{total} ({pct(matched, total)})")
    out.append("")


def visual_collapse(by_side: dict, out: list[str], exclude_no_person: bool):
    out.append(section_header("5. Visual Collapse Analysis"))
    out.append(
        "  For each text-intended category, shows how often each visual\n"
        "  category was generated (reveals stereotypical defaults).\n"
    )

    all_text_cats = sorted(
        set(r["text_intended_category"] for rows in by_side.values() for r in rows)
    )
    vis_cats = [c for c in CATEGORIES if not (exclude_no_person and c == "Keine Person")]

    for txt_cat in all_text_cats:
        out.append(f"  Text category: '{txt_cat}'")
        col_w = 32
        out.append(
            f"    {'Visual category':<{col_w}} "
            f"{'Leichte Sprache':>20}   {'Standard Dt.':>20}"
        )
        out.append(f"    {separator('-', col_w + 46)}")

        for side in SIDES:
            label = "Leichte Sprache" if side == "simple" else "Standard Dt."
        for vis_cat in vis_cats:
            s_rows = [r for r in by_side.get("simple", []) if r["text_intended_category"] == txt_cat]
            n_rows = [r for r in by_side.get("normal", []) if r["text_intended_category"] == txt_cat]
            if exclude_no_person:
                s_rows = [r for r in s_rows if r["image_gender_category"] != "Keine Person"]
                n_rows = [r for r in n_rows if r["image_gender_category"] != "Keine Person"]
            s_cnt = sum(1 for r in s_rows if r["image_gender_category"] == vis_cat)
            n_cnt = sum(1 for r in n_rows if r["image_gender_category"] == vis_cat)
            s_tot, n_tot = len(s_rows), len(n_rows)
            if s_cnt == 0 and n_cnt == 0:
                continue
            out.append(
                f"    {vis_cat:<{col_w}} "
                f"{s_cnt:>6} ({pct(s_cnt, s_tot)})   "
                f"{n_cnt:>6} ({pct(n_cnt, n_tot)})"
            )
        out.append("")


def paired_comparison(rows: list[dict], out: list[str], exclude_no_person: bool):
    out.append(section_header("6. Paired Comparison (same pair_id + seed)"))
    out.append(
        "  For each matched pair (same pair_id, same seed), compares\n"
        "  whether Leichte Sprache and Standard German generated the\n"
        "  same visual category.\n"
    )

    # Index by (pair_id, seed)
    index: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["pair_id"], row["seed"])
        index[key][row["side"]] = row

    # Only keep pairs where both sides exist
    both = {k: v for k, v in index.items() if "simple" in v and "normal" in v}
    if exclude_no_person:
        both = {
            k: v for k, v in both.items()
            if v["simple"]["image_gender_category"] != "Keine Person"
            and v["normal"]["image_gender_category"] != "Keine Person"
        }

    total = len(both)
    agree = sum(
        1 for v in both.values()
        if v["simple"]["image_gender_category"] == v["normal"]["image_gender_category"]
    )
    disagree = total - agree

    out.append(f"  Total matched pairs : {total}")
    out.append(f"  Same visual category: {agree}  ({pct(agree, total)})")
    out.append(f"  Different           : {disagree}  ({pct(disagree, total)})\n")

    # Divergence breakdown: what did each side produce when they disagreed?
    divergence: Counter = Counter()
    for v in both.values():
        s_cat = v["simple"]["image_gender_category"]
        n_cat = v["normal"]["image_gender_category"]
        if s_cat != n_cat:
            divergence[(s_cat, n_cat)] += 1

    if divergence:
        out.append("  Most common divergence patterns (LS → StD):")
        out.append(f"  {'Leichte Sprache':<32} {'Standard Dt.':<32} {'Count':>6}")
        out.append(f"  {separator('-', 74)}")
        for (s, n), cnt in divergence.most_common(15):
            out.append(f"  {s:<32} {n:<32} {cnt:>6}")
    out.append("")


def dominant_gender_summary(by_side: dict, out: list[str]):
    out.append(section_header("7. Dominant-Gender Summary (largest face bbox)"))
    out.append(
        "  Distribution of mivolo_gender (the predicted gender of the most\n"
        "  prominent person = largest face bounding box) per language side.\n"
    )

    col_w = 12
    out.append(
        f"  {'Gender':<{col_w}} "
        f"{'Leichte Sprache':>20}   {'Standard Dt.':>20}"
    )
    out.append(f"  {separator('-', col_w + 46)}")

    for gender_val in ["male", "female", ""]:
        label = gender_val if gender_val else "(none / no person)"
        s_rows = by_side.get("simple", [])
        n_rows = by_side.get("normal", [])
        s_cnt  = sum(1 for r in s_rows if r["mivolo_gender"] == gender_val)
        n_cnt  = sum(1 for r in n_rows if r["mivolo_gender"] == gender_val)
        out.append(
            f"  {label:<{col_w}} "
            f"{s_cnt:>6} ({pct(s_cnt, len(s_rows))})   "
            f"{n_cnt:>6} ({pct(n_cnt, len(n_rows))})"
        )
    out.append("")

    # Male ratio among images that have a detected person
    out.append("  Male rate (images with ≥1 detected person):")
    for side in SIDES:
        rows_with = [r for r in by_side.get(side, []) if r["mivolo_gender"] != ""]
        male = sum(1 for r in rows_with if r["mivolo_gender"] == "male")
        label = "Leichte Sprache" if side == "simple" else "Standard Dt."
        out.append(f"    {label}: {male}/{len(rows_with)} ({pct(male, len(rows_with))})")
    out.append("")


def mixed_group_gender_skew(by_side: dict, out: list[str]):
    """
    Section 8: within images classified as gemischte/generische Gruppe,
    analyse which gender dominates (largest face bbox) and compute the
    average male-count ratio.  Tests whether LS and StD differ even among
    images where both genders were detected.
    """
    out.append(section_header("8. Gender Skew Within Mixed-Group Images"))
    out.append(
        "  Restricts to images classified as gemischte/generische Gruppe.\n"
        "  Even when both genders are present, the dominant person (largest face)\n"
        "  and the male-count ratio reveal whether one side skews masculine.\n"
    )

    col_w = 24
    out.append(
        f"  {'Metric':<{col_w}} "
        f"{'Leichte Sprache':>22}   {'Standard Dt.':>22}"
    )
    out.append(f"  {separator('-', col_w + 50)}")

    metrics = {}
    for side in SIDES:
        mixed = [r for r in by_side.get(side, [])
                 if r["image_gender_category"] == "gemischte/generische Gruppe"]
        n = len(mixed) or 1

        dom_male   = sum(1 for r in mixed if r["mivolo_gender"] == "male")
        dom_female = sum(1 for r in mixed if r["mivolo_gender"] == "female")

        ratios = []
        for r in mixed:
            tot = int(r["mivolo_total_persons"])
            mal = int(r["mivolo_males"])
            if tot > 0:
                ratios.append(mal / tot)

        avg_ratio  = sum(ratios) / len(ratios) if ratios else 0
        maj_male   = sum(1 for r in ratios if r > 0.5)
        balanced   = sum(1 for r in ratios if r == 0.5)
        maj_female = sum(1 for r in ratios if r < 0.5)

        metrics[side] = dict(n=len(mixed), dom_male=dom_male, dom_female=dom_female,
                             avg_ratio=avg_ratio, maj_male=maj_male,
                             balanced=balanced, maj_female=maj_female, ratios=ratios)

    s, n_data = metrics.get("simple", {}), metrics.get("normal", {})

    def row(label, s_val, n_val):
        out.append(f"  {label:<{col_w}} {s_val:>22}   {n_val:>22}")

    row("N (mixed-group images)",
        str(s["n"]), str(n_data["n"]))
    row("Dominant gender: male",
        f"{s['dom_male']:>5} ({pct(s['dom_male'], s['n'])})",
        f"{n_data['dom_male']:>5} ({pct(n_data['dom_male'], n_data['n'])})")
    row("Dominant gender: female",
        f"{s['dom_female']:>5} ({pct(s['dom_female'], s['n'])})",
        f"{n_data['dom_female']:>5} ({pct(n_data['dom_female'], n_data['n'])})")
    row("Avg male-count ratio",
        f"{s['avg_ratio']:.3f} ({s['avg_ratio']*100:.1f}% male)",
        f"{n_data['avg_ratio']:.3f} ({n_data['avg_ratio']*100:.1f}% male)")
    row("Majority male (>50%)",
        f"{s['maj_male']:>5} ({pct(s['maj_male'], s['n'])})",
        f"{n_data['maj_male']:>5} ({pct(n_data['maj_male'], n_data['n'])})")
    row("Balanced (50/50)",
        f"{s['balanced']:>5} ({pct(s['balanced'], s['n'])})",
        f"{n_data['balanced']:>5} ({pct(n_data['balanced'], n_data['n'])})")
    row("Majority female (<50%)",
        f"{s['maj_female']:>5} ({pct(s['maj_female'], s['n'])})",
        f"{n_data['maj_female']:>5} ({pct(n_data['maj_female'], n_data['n'])})")
    out.append("")

    # Chi-square on dominant gender (2x2: male/female x side)
    if SCIPY and s["n"] > 0 and n_data["n"] > 0:
        table_dom = [
            [s["dom_male"],   s["dom_female"]],
            [n_data["dom_male"], n_data["dom_female"]],
        ]
        chi2, p, dof, _ = chi2_contingency(table_dom)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        out.append(f"  Chi-square on dominant gender (male vs female, LS vs StD):")
        out.append(f"    chi2 = {chi2:.4f},  df = {dof},  p = {p:.6f}  {sig}")
        out.append("")

    # Mann-Whitney U on male-count ratio distributions
    if SCIPY:
        try:
            from scipy.stats import mannwhitneyu
            s_rats  = s.get("ratios", [])
            n_rats  = n_data.get("ratios", [])
            if s_rats and n_rats:
                stat, p_mw = mannwhitneyu(s_rats, n_rats, alternative="two-sided")
                sig_mw = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
                out.append(f"  Mann-Whitney U on male-count ratio (LS vs StD):")
                out.append(f"    U = {stat:.1f},  p = {p_mw:.6f}  {sig_mw}")
                out.append(f"    (Tests whether the male-count ratio distributions differ.)")
                out.append("")
        except Exception as e:
            out.append(f"  [Mann-Whitney U failed: {e}]\n")

    out.append(
        "  Interpretation: if the dominant gender chi-square is significant and\n"
        "  the male-dominant rate is higher in LS, it means that even within\n"
        "  mixed groups LS systematically foregrounds male subjects -- a\n"
        "  second-order masculine bias beyond the category level.\n"
    )



# ── Visualizations ───────────────────────────────────────────────────────────────────

def _apply_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Shared figure polish."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def plot_category_distribution(by_side: dict, excl: bool, out_dir: Path) -> Path:
    """
    Plot 1: Grouped bar chart – share of each image gender category per side.
    """
    cats = [c for c in CATEGORIES if not (excl and c == "Keine Person")]
    short = {
        "weiblich (Singular)": "weibl. Sing.",
        "maskulin (Singular)": "mask. Sing.",
        "Frauengruppe":        "Frauengr.",
        "Männergruppe":         "Männergr.",
        "gemischte/generische Gruppe": "gemischt",
        "Keine Person":        "kein Person",
    }
    labels = [short.get(c, c) for c in cats]
    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, side in enumerate(SIDES):
        rows  = by_side.get(side, [])
        total = len(rows) or 1
        counts = category_counts(rows)
        vals   = [counts.get(c, 0) / total for c in cats]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=SIDE_LABELS[side], color=COLORS[side], alpha=0.88)
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _apply_style(ax,
                 "Image Gender Category Distribution",
                 ylabel="Share of images")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "plot1_category_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_visual_collapse_heatmaps(by_side: dict, excl: bool, out_dir: Path) -> list[Path]:
    """
    Plot 2 (one per text category): heatmap showing which visual categories
    are generated for each text-intended category.
    """
    all_text_cats = sorted(
        set(r["text_intended_category"] for rows in by_side.values() for r in rows)
    )
    vis_cats = [c for c in CATEGORIES if not (excl and c == "Keine Person")]
    short_vis = {
        "weiblich (Singular)": "weibl.\nSing.",
        "maskulin (Singular)": "mask.\nSing.",
        "Frauengruppe":        "Frauen-\ngr.",
        "Männergruppe":         "Männer-\ngr.",
        "gemischte/generische Gruppe": "gemischt",
        "Keine Person":        "kein\nPerson",
    }

    saved = []
    for txt_cat in all_text_cats:
        data = []
        for side in SIDES:
            rows = [r for r in by_side.get(side, []) if r["text_intended_category"] == txt_cat]
            if excl:
                rows = [r for r in rows if r["image_gender_category"] != "Keine Person"]
            total = len(rows) or 1
            data.append([sum(1 for r in rows if r["image_gender_category"] == v) / total
                         for v in vis_cats])

        mat = np.array(data)  # shape (2, n_vis_cats)

        fig, ax = plt.subplots(figsize=(max(6, len(vis_cats) * 1.4), 2.8))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() or 0.01, aspect="auto")
        ax.set_xticks(range(len(vis_cats)))
        ax.set_xticklabels([short_vis.get(v, v) for v in vis_cats], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([SIDE_LABELS[s] for s in SIDES], fontsize=9)
        for r in range(2):
            for c_idx in range(len(vis_cats)):
                ax.text(c_idx, r, f"{mat[r, c_idx]:.0%}",
                        ha="center", va="center",
                        color="white" if mat[r, c_idx] > 0.5 * mat.max() else "#333",
                        fontsize=9)
        ax.set_title(f"Visual output for text category: ‘{txt_cat}’",
                     fontsize=11, fontweight="bold", pad=8)
        fig.colorbar(im, ax=ax, format="{x:.0%}", shrink=0.8)
        fig.tight_layout()
        safe = txt_cat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        path = out_dir / f"plot2_collapse_{safe}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_semantic_alignment(by_side: dict, excl: bool, out_dir: Path) -> Path:
    """
    Plot 3: Horizontal grouped bar chart – semantic alignment accuracy
    (% images matching text-intended category) per text category and side.
    """
    all_text_cats = sorted(
        set(r["text_intended_category"] for rows in by_side.values() for r in rows)
    )
    y = np.arange(len(all_text_cats))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, len(all_text_cats) * 0.9)))
    for i, side in enumerate(SIDES):
        accs = []
        for txt_cat in all_text_cats:
            rows = [r for r in by_side.get(side, []) if r["text_intended_category"] == txt_cat]
            if excl:
                rows = [r for r in rows if r["image_gender_category"] != "Keine Person"]
            total   = len(rows) or 1
            matched = sum(1 for r in rows if r["semantic_match"] == "True")
            accs.append(matched / total)
        bars = ax.barh(y + i * height - height / 2, accs, height,
                       label=SIDE_LABELS[side], color=COLORS[side], alpha=0.88)
        for bar, val in zip(bars, accs):
            if val > 0.02:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0%}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(all_text_cats, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Semantic Alignment Accuracy\n(image category = text-intended category)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "plot3_semantic_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dominant_gender(by_side: dict, out_dir: Path) -> Path:
    """
    Plot 4: Stacked bar – dominant gender (largest face bbox) per side.
    """
    gender_vals  = ["male", "female", ""]
    gender_labels = ["male", "female", "none/no person"]
    gender_colors = [COLORS["male"], COLORS["female"], COLORS["none"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(SIDES))
    bottoms = np.zeros(len(SIDES))

    for gval, glabel, gcol in zip(gender_vals, gender_labels, gender_colors):
        vals = []
        for side in SIDES:
            rows  = by_side.get(side, [])
            total = len(rows) or 1
            vals.append(sum(1 for r in rows if r["mivolo_gender"] == gval) / total)
        bars = ax.bar(x, vals, bottom=bottoms, label=glabel, color=gcol, alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([SIDE_LABELS[s] for s in SIDES])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Dominant Person Gender\n(largest face bounding box)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "plot4_dominant_gender.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paired_divergence(rows: list[dict], excl: bool, out_dir: Path) -> Path:
    """
    Plot 5: Heatmap of divergence patterns in matched pairs
    (rows = LS category, cols = StD category, cell = count of disagreements).
    """
    index: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        index[(row["pair_id"], row["seed"])][row["side"]] = row

    both = {k: v for k, v in index.items() if "simple" in v and "normal" in v}
    if excl:
        both = {
            k: v for k, v in both.items()
            if v["simple"]["image_gender_category"] != "Keine Person"
            and v["normal"]["image_gender_category"] != "Keine Person"
        }

    cats = CATEGORIES
    cat_idx = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((len(cats), len(cats)), dtype=int)
    for v in both.values():
        s = v["simple"]["image_gender_category"]
        n = v["normal"]["image_gender_category"]
        mat[cat_idx[s], cat_idx[n]] += 1

    short_cats = [
        "weibl.Sing.", "mask.Sing.", "Frauengr.",
        "Männergr.", "gemischt", "kein Person"
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(short_cats, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(short_cats, fontsize=8.5)
    for r in range(len(cats)):
        for c_idx in range(len(cats)):
            val = mat[r, c_idx]
            if val > 0:
                ax.text(c_idx, r, str(val), ha="center", va="center",
                        color="white" if val > mat.max() * 0.6 else "#333",
                        fontsize=8)
    ax.set_xlabel("Standard Deutsch → visual category", fontsize=9)
    ax.set_ylabel("Leichte Sprache → visual category", fontsize=9)
    ax.set_title("Paired Divergence Heatmap\n(diagonal = agreement)",
                 fontsize=12, fontweight="bold", pad=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="# matched pairs")
    fig.tight_layout()
    path = out_dir / "plot5_paired_divergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_plots(rows: list[dict], by_side: dict, excl: bool):
    if not PLOTTING:
        print("[WARN] Skipping plots (matplotlib/numpy not available).")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    print("[INFO] Generating plots...")
    paths.append(plot_category_distribution(by_side, excl, PLOTS_DIR))
    paths.extend(plot_visual_collapse_heatmaps(by_side, excl, PLOTS_DIR))
    paths.append(plot_semantic_alignment(by_side, excl, PLOTS_DIR))
    paths.append(plot_dominant_gender(by_side, PLOTS_DIR))
    paths.append(plot_paired_divergence(rows, excl, PLOTS_DIR))

    print(f"[INFO] Saved {len(paths)} plot(s) to {PLOTS_DIR}:")
    for p in paths:
        print(f"       {p.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[ERROR] Input file not found: {in_path}")

    rows   = load_csv(in_path)
    by_side = partition(rows)
    excl   = args.exclude_no_person

    print(f"[INFO] Loaded {len(rows)} rows from {in_path.name}")
    for side in SIDES:
        print(f"       {side:8s}: {len(by_side.get(side, []))} rows")
    if excl:
        print("[INFO] --exclude-no-person: 'Keine Person' rows excluded from alignment + collapse analyses")
    print()

    out: list[str] = []
    title = (
        "Image Bias Analysis: Leichte Sprache vs. Standard German\n"
        f"  Input : {in_path}\n"
        f"  Rows  : {len(rows)} total  |  "
        f"Leichte Sprache: {len(by_side.get('simple', []))}  |  "
        f"Standard Dt.: {len(by_side.get('normal', []))}\n"
        f"  Excl. no-person: {excl}"
    )
    out.append(separator("═"))
    out.append(title)
    out.append(separator("═"))

    no_person_rate(by_side, out)
    category_distribution(by_side, out)
    chi_square_test(by_side, out, excl)
    semantic_alignment(by_side, out, excl)
    visual_collapse(by_side, out, excl)
    paired_comparison(rows, out, excl)
    dominant_gender_summary(by_side, out)
    mixed_group_gender_skew(by_side, out)

    report = "\n".join(out)
    print(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[INFO] Report saved to: {out_path}")

    generate_plots(rows, by_side, excl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse MiVOLO image results and compare language variants."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_IN),
        help=f"Path to t2i_mivolo_results.csv (default: {DEFAULT_IN})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help=f"Path for the text report (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--exclude-no-person",
        action="store_true",
        help="Exclude 'Keine Person' images from alignment and collapse analyses.",
    )
    run(parser.parse_args())
