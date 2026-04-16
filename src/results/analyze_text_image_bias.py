#!/usr/bin/env python3
"""Compute structured text-to-image transformation results."""

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
DEFAULT_IN = PROJECT_ROOT / "data" / "processed" / "t2i_mivolo_results.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "results" / "text_image_bias"

SIDES = ["simple", "normal"]
SIDE_LABELS = {"simple": "Easy Language", "normal": "Standard German"}
COLORS = {"simple": "#5B8DB8", "normal": "#E07B54"}
IMAGE_CATEGORIES = ["female (singular)", "male (singular)", "female group", "male group", "mixed/generic group", "no person"]
TEXT_CATEGORY_MAP = {"f1": "female (singular)", "m1": "male (singular)", "fg": "female group", "mg": "male group", "mx": "mixed/generic group", "ux": "mixed/generic group"}
FRAME_ORDER = ["fem", "male_specific", "gmasc_generic", "epic_individual", "epic_group", "pair"]


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


def canonical_text_category(row: dict) -> str:
    value = (row.get("text_intended_category") or row.get("final_t2i_label") or "").strip()
    return TEXT_CATEGORY_MAP.get(value, value)


def prepare_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["text_category"] = canonical_text_category(row)
        prepared.append(item)
    return prepared


def partition(rows: list[dict]) -> dict[str, list[dict]]:
    by_side: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_side[row["side"]].append(row)
    return by_side


def active_text_categories(rows: list[dict]) -> list[str]:
    present = {row["text_category"] for row in rows if row.get("text_category")}
    return [category for category in IMAGE_CATEGORIES if category in present]


def active_frames(rows: list[dict]) -> list[str]:
    present = {row.get("final_frame") for row in rows if row.get("final_frame")}
    return [frame for frame in FRAME_ORDER if frame in present]


def chi_square(simple_counts: Counter, normal_counts: Counter, categories: list[str]) -> dict:
    active = [category for category in categories if simple_counts.get(category, 0) + normal_counts.get(category, 0) > 0]
    table = [[simple_counts.get(category, 0) for category in active], [normal_counts.get(category, 0) for category in active]]
    result = {"categories": active, "table": table, "available": chi2_contingency is not None}
    if chi2_contingency is None or len(active) < 2:
        return result
    chi2, p, dof, expected = chi2_contingency(table)
    result.update({"chi2": float(chi2), "p": float(p), "dof": int(dof), "expected": [[float(value) for value in row] for row in expected]})
    return result


def dataset_overview(rows: list[dict]) -> list[dict]:
    by_side = partition(rows)
    output = []
    for side in SIDES:
        side_rows = by_side.get(side, [])
        no_person = sum(1 for row in side_rows if row.get("image_gender_category") == "no person")
        matches = sum(1 for row in side_rows if row.get("text_category") == row.get("image_gender_category"))
        output.append({"side": side, "rows": len(side_rows), "no_person": no_person, "semantic_matches": matches, "semantic_match_share": round(matches / len(side_rows), 6) if side_rows else 0.0})
    return output


def text_category_counts(rows: list[dict], text_cats: list[str]) -> list[dict]:
    output = []
    by_side = partition(rows)
    for side in SIDES:
        side_rows = by_side.get(side, [])
        counts = Counter(row["text_category"] for row in side_rows)
        for text_cat in text_cats:
            count = counts.get(text_cat, 0)
            output.append({"side": side, "text_category": text_cat, "count": count, "share": round(count / len(side_rows), 6) if side_rows else 0.0})
    return output


def text_image_shift_table(rows: list[dict], text_cats: list[str], exclude_no_person: bool) -> list[dict]:
    image_cats = [cat for cat in IMAGE_CATEGORIES if not (exclude_no_person and cat == "no person")]
    output = []
    for side in SIDES:
        for text_cat in text_cats:
            subset = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            if exclude_no_person:
                subset = [row for row in subset if row.get("image_gender_category") != "no person"]
            counts = Counter(row.get("image_gender_category", "") for row in subset)
            total = len(subset)
            record = {"side": side, "text_category": text_cat, "total": total}
            for image_cat in image_cats:
                record[image_cat] = counts.get(image_cat, 0)
                record[f"{image_cat}_share"] = round(counts.get(image_cat, 0) / total, 6) if total else 0.0
            output.append(record)
    return output


def stratified_chi_square(rows: list[dict], text_cats: list[str], exclude_no_person: bool) -> list[dict]:
    image_cats = [cat for cat in IMAGE_CATEGORIES if not (exclude_no_person and cat == "no person")]
    output = []
    for text_cat in text_cats:
        simple_rows = [row for row in rows if row["side"] == "simple" and row["text_category"] == text_cat]
        normal_rows = [row for row in rows if row["side"] == "normal" and row["text_category"] == text_cat]
        if exclude_no_person:
            simple_rows = [row for row in simple_rows if row.get("image_gender_category") != "no person"]
            normal_rows = [row for row in normal_rows if row.get("image_gender_category") != "no person"]
        result = chi_square(Counter(row["image_gender_category"] for row in simple_rows), Counter(row["image_gender_category"] for row in normal_rows), image_cats)
        output.append({"text_category": text_cat, "simple_total": len(simple_rows), "normal_total": len(normal_rows), "active_categories": "|".join(result["categories"]), "chi2": result.get("chi2", ""), "dof": result.get("dof", ""), "p": result.get("p", "")})
    return output


def semantic_match_rates(rows: list[dict], text_cats: list[str], exclude_no_person: bool) -> list[dict]:
    output = []
    for side in SIDES:
        for text_cat in text_cats:
            subset = [row for row in rows if row["side"] == side and row["text_category"] == text_cat]
            if exclude_no_person:
                subset = [row for row in subset if row.get("image_gender_category") != "no person"]
            matches = sum(1 for row in subset if row.get("text_category") == row.get("image_gender_category"))
            output.append({"side": side, "text_category": text_cat, "total": len(subset), "matches": matches, "match_share": round(matches / len(subset), 6) if subset else 0.0})
    return output


def final_frame_image_shift_table(rows: list[dict], frames: list[str]) -> list[dict]:
    output = []
    for side in SIDES:
        for frame in frames:
            subset = [row for row in rows if row["side"] == side and row.get("final_frame") == frame]
            counts = Counter(row.get("image_gender_category", "") for row in subset)
            total = len(subset)
            record = {"side": side, "final_frame": frame, "total": total}
            for image_cat in IMAGE_CATEGORIES:
                record[image_cat] = counts.get(image_cat, 0)
                record[f"{image_cat}_share"] = round(counts.get(image_cat, 0) / total, 6) if total else 0.0
            output.append(record)
    return output


def mixed_group_dominance_by_frame(rows: list[dict], frames: list[str]) -> list[dict]:
    output = []
    for side in SIDES:
        for frame in frames:
            subset = [row for row in rows if row["side"] == side and row.get("final_frame") == frame and row.get("image_gender_category") == "mixed/generic group"]
            male_ratios = []
            for row in subset:
                total_persons = int(row.get("mivolo_total_persons") or 0)
                males = int(row.get("mivolo_males") or 0)
                if total_persons > 0:
                    male_ratios.append(males / total_persons)
            output.append(
                {
                    "side": side,
                    "final_frame": frame,
                    "mixed_group_images": len(subset),
                    "dominant_male": sum(1 for row in subset if row.get("mivolo_gender") == "male"),
                    "dominant_female": sum(1 for row in subset if row.get("mivolo_gender") == "female"),
                    "avg_male_count_ratio": round(sum(male_ratios) / len(male_ratios), 6) if male_ratios else 0.0,
                }
            )
    return output


def paired_divergence(rows: list[dict], exclude_no_person: bool) -> tuple[dict, list[dict]]:
    index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        index[(row["pair_actor_id"], row["seed"])][row["side"]] = row
    both = {key: sides for key, sides in index.items() if "simple" in sides and "normal" in sides}
    if exclude_no_person:
        both = {
            key: sides for key, sides in both.items()
            if sides["simple"].get("image_gender_category") != "no person" and sides["normal"].get("image_gender_category") != "no person"
        }
    divergence = Counter()
    agree = 0
    for sides in both.values():
        simple = sides["simple"].get("image_gender_category")
        normal = sides["normal"].get("image_gender_category")
        if simple == normal:
            agree += 1
        else:
            divergence[(simple, normal)] += 1
    total = len(both)
    summary = {"matched_actor_seed_pairs": total, "same_image_category": agree, "same_image_category_share": round(agree / total, 6) if total else 0.0}
    rows_out = [{"simple_image_category": s, "normal_image_category": n, "count": count} for (s, n), count in divergence.items()]
    return summary, sorted(rows_out, key=lambda row: row["count"], reverse=True)


def plot_heatmap_from_table(table_rows: list[dict], row_key: str, out_path: Path, title: str) -> None:
    if plt is None:
        return
    row_labels = list(dict.fromkeys(row[row_key] for row in table_rows if row["side"] == "simple"))
    image_cats = [cat for cat in IMAGE_CATEGORIES if cat in table_rows[0]]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(row_labels) * 0.8)), constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        matrix = []
        for label in row_labels:
            record = next(row for row in table_rows if row["side"] == side and row[row_key] == label)
            total = record["total"] or 1
            matrix.append([record[cat] / total for cat in image_cats])
        mat = np.array(matrix)
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(image_cats)))
        ax.set_xticklabels(image_cats, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(SIDE_LABELS[side])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, format=mticker.PercentFormatter(1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_match_rates(rows: list[dict], out_path: Path) -> None:
    if plt is None:
        return
    text_cats = list(dict.fromkeys(row["text_category"] for row in rows if row["side"] == "simple"))
    y = np.arange(len(text_cats))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(text_cats) * 0.7)))
    for idx, side in enumerate(SIDES):
        vals = [next(row["match_share"] for row in rows if row["side"] == side and row["text_category"] == text_cat) for text_cat in text_cats]
        ax.barh(y + idx * height - height / 2, vals, height, color=COLORS[side], label=SIDE_LABELS[side])
    ax.set_yticks(y)
    ax.set_yticklabels(text_cats)
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Text-image match rate by text category")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mixed_group_dominance(rows: list[dict], out_path: Path) -> None:
    if plt is None:
        return
    frames = list(dict.fromkeys(row["final_frame"] for row in rows if row["side"] == "simple"))
    x = np.arange(len(frames))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, side in enumerate(SIDES):
        vals = []
        for frame in frames:
            record = next(row for row in rows if row["side"] == side and row["final_frame"] == frame)
            total = record["mixed_group_images"]
            vals.append(record["dominant_male"] / total if total else 0.0)
        ax.bar(x + idx * width - width / 2, vals, width, label=SIDE_LABELS[side], color=COLORS[side])
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylabel("Share with dominant male face")
    ax.set_title("Dominant male face in mixed-group images by final_frame")
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    rows = prepare_rows(load_csv(args.input))
    text_cats = active_text_categories(rows)
    frames = active_frames(rows)

    overview_rows = dataset_overview(rows)
    text_count_rows = text_category_counts(rows, text_cats)
    shift_rows = text_image_shift_table(rows, text_cats, args.exclude_no_person)
    stratified_rows = stratified_chi_square(rows, text_cats, args.exclude_no_person)
    match_rows = semantic_match_rates(rows, text_cats, args.exclude_no_person)
    frame_shift_rows = final_frame_image_shift_table(rows, frames)
    mixed_rows = mixed_group_dominance_by_frame(rows, frames)
    paired_summary, paired_rows = paired_divergence(rows, args.exclude_no_person)

    summary = {
        "input_file": str(args.input),
        "exclude_no_person": args.exclude_no_person,
        "rows_total": len(rows),
        "text_categories": text_cats,
        "final_frames": frames,
        "overview": overview_rows,
        "paired": paired_summary,
    }
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "dataset_overview.csv", overview_rows)
    write_csv(out_dir / "text_category_counts.csv", text_count_rows)
    write_csv(out_dir / "text_image_shift_table.csv", shift_rows)
    write_csv(out_dir / "stratified_chi_square.csv", stratified_rows)
    write_csv(out_dir / "semantic_match_rates.csv", match_rows)
    write_csv(out_dir / "final_frame_image_shift_table.csv", frame_shift_rows)
    write_csv(out_dir / "mixed_group_dominance_by_frame.csv", mixed_rows)
    write_csv(out_dir / "paired_divergence.csv", paired_rows)

    plot_heatmap_from_table(shift_rows, "text_category", plots_dir / "text_image_heatmap_by_side.png", "Text category -> image category")
    plot_match_rates(match_rows, plots_dir / "text_image_match_rates.png")
    plot_heatmap_from_table(frame_shift_rows, "final_frame", plots_dir / "final_frame_heatmap.png", "Original final_frame -> image category")
    plot_mixed_group_dominance(mixed_rows, plots_dir / "mixed_group_dominance_by_frame.png")
    print(f"Wrote text-image-bias results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute structured text-to-image transformation results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exclude-no-person", action="store_true")
    run(parser.parse_args())
