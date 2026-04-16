#!/usr/bin/env python3
"""Compute structured image-side bias results from MiVOLO outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scipy.stats import chi2_contingency, mannwhitneyu
except ImportError:  # pragma: no cover - optional dependency
    chi2_contingency = mannwhitneyu = None

try:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = np = plt = mticker = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = PROJECT_ROOT / "data" / "processed" / "t2i_mivolo_results.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "results" / "image_bias"

SIDES = ["simple", "normal"]
SIDE_LABELS = {"simple": "Easy Language", "normal": "Standard German"}
COLORS = {"simple": "#5B8DB8", "normal": "#E07B54", "male": "#4A90D9", "female": "#E05C8A", "none": "#AAAAAA"}
CATEGORIES = ["female (singular)", "male (singular)", "female group", "male group", "mixed/generic group", "no person"]
CATEGORY_FILENAME_SLUG = {
    "female (singular)": "female_singular",
    "male (singular)": "male_singular",
    "female group": "female_group",
    "male group": "male_group",
    "mixed/generic group": "mixed_generic_group",
    "no person": "no_person",
}


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


def category_counts(rows: list[dict], key: str = "image_gender_category") -> Counter:
    return Counter(row.get(key, "") for row in rows)


def side_category_count_rows(by_side: dict[str, list[dict]], key: str, categories: list[str]) -> list[dict]:
    rows = []
    for side in SIDES:
        side_rows = by_side.get(side, [])
        counts = category_counts(side_rows, key)
        total = len(side_rows)
        for category in categories:
            count = counts.get(category, 0)
            rows.append({"side": side, "category": category, "count": count, "share": round(count / total, 6) if total else 0.0})
    return rows


def chi_square_from_counts(simple_counts: Counter, normal_counts: Counter, categories: list[str]) -> dict:
    active = [category for category in categories if simple_counts.get(category, 0) + normal_counts.get(category, 0) > 0]
    table = [[simple_counts.get(category, 0) for category in active], [normal_counts.get(category, 0) for category in active]]
    result = {"categories": active, "table": table, "available": chi2_contingency is not None}
    if chi2_contingency is None or len(active) < 2:
        return result
    chi2, p, dof, expected = chi2_contingency(table)
    result.update({"chi2": float(chi2), "p": float(p), "dof": int(dof), "expected": [[float(value) for value in row] for row in expected]})
    return result


def text_categories(rows: list[dict]) -> list[str]:
    present = {row["text_intended_category"] for row in rows if row.get("text_intended_category")}
    return [category for category in CATEGORIES if category in present]


def image_category_counts(rows: list[dict]) -> list[dict]:
    by_side = partition(rows)
    return side_category_count_rows(by_side, "image_gender_category", CATEGORIES)


def no_person_rates(by_side: dict[str, list[dict]]) -> list[dict]:
    output = []
    for side in SIDES:
        rows = by_side.get(side, [])
        count = sum(1 for row in rows if row.get("image_gender_category") == "no person")
        output.append({"side": side, "total": len(rows), "no_person_count": count, "no_person_share": round(count / len(rows), 6) if rows else 0.0})
    return output


def semantic_alignment(rows: list[dict], exclude_no_person: bool) -> list[dict]:
    output = []
    for text_category in text_categories(rows):
        for side in SIDES:
            subset = [row for row in rows if row["side"] == side and row["text_intended_category"] == text_category]
            if exclude_no_person:
                subset = [row for row in subset if row["image_gender_category"] != "no person"]
            matches = sum(1 for row in subset if row["image_gender_category"] == row["text_intended_category"])
            output.append({"text_category": text_category, "side": side, "total": len(subset), "matches": matches, "match_share": round(matches / len(subset), 6) if subset else 0.0})
    for side in SIDES:
        subset = [row for row in rows if row["side"] == side]
        if exclude_no_person:
            subset = [row for row in subset if row["image_gender_category"] != "no person"]
        matches = sum(1 for row in subset if row["image_gender_category"] == row["text_intended_category"])
        output.append({"text_category": "__overall__", "side": side, "total": len(subset), "matches": matches, "match_share": round(matches / len(subset), 6) if subset else 0.0})
    return output


def visual_collapse(rows: list[dict], exclude_no_person: bool) -> list[dict]:
    image_cats = [category for category in CATEGORIES if not (exclude_no_person and category == "no person")]
    output = []
    for text_category in text_categories(rows):
        for side in SIDES:
            subset = [row for row in rows if row["side"] == side and row["text_intended_category"] == text_category]
            if exclude_no_person:
                subset = [row for row in subset if row["image_gender_category"] != "no person"]
            counts = category_counts(subset)
            for image_category in image_cats:
                count = counts.get(image_category, 0)
                output.append(
                    {
                        "text_category": text_category,
                        "side": side,
                        "image_category": image_category,
                        "count": count,
                        "share": round(count / len(subset), 6) if subset else 0.0,
                    }
                )
    return output


def stratified_chi_square(rows: list[dict], exclude_no_person: bool) -> list[dict]:
    image_cats = [category for category in CATEGORIES if not (exclude_no_person and category == "no person")]
    output = []
    for text_category in text_categories(rows):
        simple_rows = [row for row in rows if row["side"] == "simple" and row["text_intended_category"] == text_category]
        normal_rows = [row for row in rows if row["side"] == "normal" and row["text_intended_category"] == text_category]
        if exclude_no_person:
            simple_rows = [row for row in simple_rows if row["image_gender_category"] != "no person"]
            normal_rows = [row for row in normal_rows if row["image_gender_category"] != "no person"]
        result = chi_square_from_counts(category_counts(simple_rows), category_counts(normal_rows), image_cats)
        output.append(
            {
                "text_category": text_category,
                "simple_total": len(simple_rows),
                "normal_total": len(normal_rows),
                "active_categories": "|".join(result["categories"]),
                "chi2": result.get("chi2", ""),
                "dof": result.get("dof", ""),
                "p": result.get("p", ""),
            }
        )
    return output


def paired_divergence(rows: list[dict], exclude_no_person: bool) -> tuple[dict, list[dict]]:
    index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        index[(row["pair_id"], row["seed"])][row["side"]] = row
    both = {key: sides for key, sides in index.items() if "simple" in sides and "normal" in sides}
    if exclude_no_person:
        both = {
            key: sides for key, sides in both.items()
            if sides["simple"]["image_gender_category"] != "no person" and sides["normal"]["image_gender_category"] != "no person"
        }
    divergence = Counter()
    agree = 0
    for sides in both.values():
        simple = sides["simple"]["image_gender_category"]
        normal = sides["normal"]["image_gender_category"]
        if simple == normal:
            agree += 1
        else:
            divergence[(simple, normal)] += 1
    total = len(both)
    summary = {"matched_pairs": total, "same_image_category": agree, "same_image_category_share": round(agree / total, 6) if total else 0.0}
    rows_out = [{"simple_image_category": s, "normal_image_category": n, "count": count} for (s, n), count in divergence.items()]
    return summary, sorted(rows_out, key=lambda row: row["count"], reverse=True)


def dominant_gender_counts(by_side: dict[str, list[dict]]) -> list[dict]:
    output = []
    for side in SIDES:
        rows = by_side.get(side, [])
        for gender in ["male", "female", ""]:
            count = sum(1 for row in rows if row.get("mivolo_gender") == gender)
            output.append({"side": side, "dominant_gender": gender or "none/no person", "count": count, "share": round(count / len(rows), 6) if rows else 0.0})
    return output


def mixed_group_dominance(rows: list[dict]) -> tuple[dict, list[dict]]:
    output = []
    summary = {}
    for side in SIDES:
        subset = [row for row in rows if row["side"] == side and row["image_gender_category"] == "mixed/generic group"]
        ratios = []
        for row in subset:
            total_persons = int(row.get("mivolo_total_persons") or 0)
            males = int(row.get("mivolo_males") or 0)
            if total_persons > 0:
                ratios.append(males / total_persons)
        record = {
            "side": side,
            "mixed_group_images": len(subset),
            "dominant_male": sum(1 for row in subset if row.get("mivolo_gender") == "male"),
            "dominant_female": sum(1 for row in subset if row.get("mivolo_gender") == "female"),
            "avg_male_count_ratio": round(sum(ratios) / len(ratios), 6) if ratios else 0.0,
            "majority_male": sum(1 for ratio in ratios if ratio > 0.5),
            "balanced": sum(1 for ratio in ratios if ratio == 0.5),
            "majority_female": sum(1 for ratio in ratios if ratio < 0.5),
        }
        output.append(record)
        summary[side] = {**record, "ratios": ratios}
    tests = {}
    if chi2_contingency is not None and all(summary.get(side, {}).get("mixed_group_images", 0) for side in SIDES):
        table = [[summary["simple"]["dominant_male"], summary["simple"]["dominant_female"]], [summary["normal"]["dominant_male"], summary["normal"]["dominant_female"]]]
        chi2, p, dof, _ = chi2_contingency(table)
        tests["dominant_gender_chi_square"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof), "table": table}
    if mannwhitneyu is not None and summary.get("simple", {}).get("ratios") and summary.get("normal", {}).get("ratios"):
        stat, p = mannwhitneyu(summary["simple"]["ratios"], summary["normal"]["ratios"], alternative="two-sided")
        tests["male_count_ratio_mann_whitney_u"] = {"u": float(stat), "p": float(p)}
    return tests, output


def plot_category_distribution(count_rows: list[dict], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(CATEGORIES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, side in enumerate(SIDES):
        vals = [next(row["share"] for row in count_rows if row["side"] == side and row["category"] == category) for category in CATEGORIES]
        ax.bar(x + idx * width - width / 2, vals, width, label=SIDE_LABELS[side], color=COLORS[side], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(["female\nsing.", "male\nsing.", "female\ngroup", "male\ngroup", "mixed/\ngeneric", "no\nperson"], rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of images")
    ax.set_title("Image Gender Category Distribution")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "category_distribution.png", dpi=180)
    plt.close(fig)


def plot_visual_collapse(rows: list[dict], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for text_category in sorted({row["text_category"] for row in rows}):
        text_rows = [row for row in rows if row["text_category"] == text_category]
        image_cats = [row["image_category"] for row in text_rows if row["side"] == "simple"]
        matrix = []
        for side in SIDES:
            matrix.append([next(row["share"] for row in text_rows if row["side"] == side and row["image_category"] == image_cat) for image_cat in image_cats])
        mat = np.array(matrix)
        fig, ax = plt.subplots(figsize=(max(6, len(image_cats) * 1.4), 2.8))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 0.01), aspect="auto")
        ax.set_xticks(range(len(image_cats)))
        ax.set_xticklabels([cat.replace(" ", "\n", 1) for cat in image_cats], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([SIDE_LABELS[side] for side in SIDES], fontsize=9)
        for row_idx in range(mat.shape[0]):
            for col_idx in range(mat.shape[1]):
                ax.text(col_idx, row_idx, f"{mat[row_idx, col_idx]:.0%}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Visual output for text category: {text_category}")
        fig.colorbar(im, ax=ax, format="{x:.0%}", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"collapse_{CATEGORY_FILENAME_SLUG[text_category]}.png", dpi=180)
        plt.close(fig)


def plot_dominant_gender(rows: list[dict], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["male", "female", "none/no person"]
    colors = [COLORS["male"], COLORS["female"], COLORS["none"]]
    x = np.arange(len(SIDES))
    bottoms = np.zeros(len(SIDES))
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in zip(labels, colors):
        vals = [next(row["share"] for row in rows if row["side"] == side and row["dominant_gender"] == label) for side in SIDES]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.9)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([SIDE_LABELS[side] for side in SIDES])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Dominant Person Gender")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dominant_gender.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    rows = load_csv(args.input)
    by_side = partition(rows)
    exclude_no_person = args.exclude_no_person

    category_rows = image_category_counts(rows)
    no_person_rows = no_person_rates(by_side)
    alignment_rows = semantic_alignment(rows, exclude_no_person)
    collapse_rows = visual_collapse(rows, exclude_no_person)
    stratified_rows = stratified_chi_square(rows, exclude_no_person)
    paired_summary, paired_rows = paired_divergence(rows, exclude_no_person)
    dominant_rows = dominant_gender_counts(by_side)
    mixed_tests, mixed_rows = mixed_group_dominance(rows)

    overall_test = chi_square_from_counts(category_counts(by_side.get("simple", [])), category_counts(by_side.get("normal", [])), [c for c in CATEGORIES if not (exclude_no_person and c == "no person")])
    summary = {
        "input_file": str(args.input),
        "exclude_no_person": exclude_no_person,
        "rows_total": len(rows),
        "rows_by_side": {side: len(by_side.get(side, [])) for side in SIDES},
        "overall_image_category_chi_square": overall_test,
        "paired": paired_summary,
        "mixed_group_tests": mixed_tests,
    }

    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "image_category_counts.csv", category_rows)
    write_csv(out_dir / "no_person_rates.csv", no_person_rows)
    write_csv(out_dir / "semantic_alignment.csv", alignment_rows)
    write_csv(out_dir / "visual_collapse.csv", collapse_rows)
    write_csv(out_dir / "stratified_chi_square.csv", stratified_rows)
    write_csv(out_dir / "paired_divergence.csv", paired_rows)
    write_csv(out_dir / "dominant_gender_counts.csv", dominant_rows)
    write_csv(out_dir / "mixed_group_dominance.csv", mixed_rows)

    plot_category_distribution(category_rows, plots_dir)
    plot_visual_collapse(collapse_rows, plots_dir)
    plot_dominant_gender(dominant_rows, plots_dir)
    print(f"Wrote image-bias results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute structured MiVOLO image-bias results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exclude-no-person", action="store_true")
    run(parser.parse_args())
