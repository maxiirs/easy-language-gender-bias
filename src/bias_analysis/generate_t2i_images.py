#!/usr/bin/env python3
"""
Generate Text-to-Image pairs for gender bias analysis.

Implements two complementary analysis strategies:
  A1 (Topic-Level): For each pair_id, generate images for ALL actors on both sides
                    using the same seeds. Enables paired statistical tests per topic.
  A3 (Global):      Aggregate all LS/NS images for overall bias comparison.

Usage:
  python src/bias_analysis/generate_t2i_images.py [--dry-run] [--n-seeds 5]
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import torch
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: 'diffusers' or 'torch' not found. Install with: pip install diffusers torch")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = list

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_FILE   = Path("data/processed/actor_prompts.csv")
IMAGE_OUT_DIR = Path("data/processed/t2i_images")
METADATA_OUT  = Path("data/processed/t2i_metadata.csv")

# ── Configuration ─────────────────────────────────────────────────────────────
# These seeds are shared between LS and NS images for reproducibility and
# matched-pair comparison. All seeds produce different latent noise patterns.
DEFAULT_SEEDS = [1000, 2000, 3000, 4000, 5000]

# Flux.1-schnell is the fastest Flux variant and natively supports German
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_actor_rows(input_file: Path) -> list[dict]:
    with input_file.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_a1_pairs(rows: list[dict]) -> dict[str, dict]:
    """
    Groups all actors by pair_id for A1 Topic-Level analysis.
    Returns a dict: pair_id -> {simple: [actor_rows], normal: [actor_rows]}
    """
    pairs = defaultdict(lambda: {"simple": [], "normal": []})
    for row in rows:
        # Only accept rows that have a usable prompt
        prompt = row.get("prompt_minimal_context", "").strip()
        if not prompt:
            continue
        pairs[row["pair_id"]][row["side"]].append(row)
    return dict(pairs)


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16


def load_pipeline(device, dtype):
    print(f"Loading Flux pipeline ({MODEL_ID}) on {device}...")
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    print("Model loaded.\n")
    return pipe


def generate_one(pipe, prompt: str, seed: int, out_path: Path, device: str):
    if out_path.exists():
        return  # Skip already generated images (resume-friendly)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        guidance_scale=0.0,       # Flux-schnell requires guidance_scale=0
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)


def run(args):
    rows = load_actor_rows(INPUT_FILE)
    if not rows:
        print("No rows found in input CSV. Run extract_actors.py first.")
        return

    pairs = build_a1_pairs(rows)
    seeds = args.seeds

    device, dtype = get_device_and_dtype()
    pipe = load_pipeline(device, dtype) if not args.dry_run and DIFFUSERS_AVAILABLE else None

    metadata_rows = []
    total_images = 0

    # Only process pairs with at least one actor on both sides (A1 requirement)
    eligible_pairs = {
        pid: sides for pid, sides in pairs.items()
        if sides.get("simple") and sides.get("normal")
    }
    skipped = len(pairs) - len(eligible_pairs)
    print(f"Found {len(pairs)} topic pairs → {len(eligible_pairs)} with actors on both sides ({skipped} skipped). Seeds: {seeds}\n")

    for pair_id, sides in tqdm(eligible_pairs.items()):
        for side, actors in sides.items():
            for actor_row in actors:
                prompt = actor_row.get("prompt_minimal_context", "").strip()
                actor_label = actor_row.get("actor_lemma") or actor_row.get("actor_label", "")
                actor_id = actor_row.get("actor_id", "")

                for i, seed in enumerate(seeds):
                    img_filename = f"{actor_id}_seed{seed:04d}.png"
                    img_path = IMAGE_OUT_DIR / pair_id / side / img_filename

                    if not args.dry_run and pipe is not None:
                        generate_one(pipe, prompt, seed, img_path, device)

                    metadata_rows.append({
                        "pair_id": pair_id,
                        "side": side,
                        "source": actor_row.get("source", ""),
                        "topic": actor_row.get("topic", ""),
                        "actor_id": actor_id,
                        "actor_label": actor_label,
                        "predicted_gender": actor_row.get("predicted_gender", ""),
                        "seed": seed,
                        "seed_iteration": i,
                        "prompt": prompt,
                        "image_path": str(img_path),
                        # MiVOLO will fill these in the analysis step:
                        "mivolo_gender": "",
                        "mivolo_age": "",
                    })
                    total_images += 1

    # Save metadata CSV (includes all planned images, even if dry run)
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    mode = "DRY-RUN planned" if args.dry_run else "Generated"
    print(f"\n{mode} {total_images} images across {len(pairs)} topic pairs.")
    print(f"Metadata saved to: {METADATA_OUT}")
    print("\nNext step: Run MiVOLO on all generated images and fill in mivolo_gender/mivolo_age columns.")


def main():
    parser = argparse.ArgumentParser(description="Generate T2I images for gender bias analysis.")
    parser.add_argument("--dry-run", action="store_true", help="Plan images but skip generation (fast test).")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds (images) per actor (default: 5).")
    args = parser.parse_args()
    args.seeds = DEFAULT_SEEDS[:args.n_seeds]
    run(args)


if __name__ == "__main__":
    main()
