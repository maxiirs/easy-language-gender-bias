#!/usr/bin/env python3
"""Generate paired T2I images from actor-level prompt tables.

The main experiment uses only actor pairs that:
- exist on both sides (`has_simple = 1`, `has_normal = 1`)
- are marked for T2I by default on both sides

For each eligible actor pair, the script generates one image per side and seed
using `*_prompt_actor_context`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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

INPUT_FILE = Path("data/processed/actor_prompts_paired.csv")
IMAGE_OUT_DIR = Path("data/processed/t2i_images")
METADATA_OUT = Path("data/processed/t2i_metadata.csv")

DEFAULT_SEEDS = [0, 1000, 2000, 3000, 4000]
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
SIDES = ("simple", "normal")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_enabled(value: str) -> bool:
    return str(value).strip() == "1"


def is_eligible_pair(row: dict) -> bool:
    return (
        is_enabled(row.get("has_simple", ""))
        and is_enabled(row.get("has_normal", ""))
        and is_enabled(row.get("simple_include_in_t2i_default", ""))
        and is_enabled(row.get("normal_include_in_t2i_default", ""))
        and bool(row.get("simple_prompt_actor_context", "").strip())
        and bool(row.get("normal_prompt_actor_context", "").strip())
    )


def iter_generation_jobs(rows: list[dict], seeds: list[int]) -> list[dict]:
    jobs: list[dict] = []
    for row in rows:
        if not is_eligible_pair(row):
            continue
        for side in SIDES:
            for seed_index, seed in enumerate(seeds):
                actor_id = row[f"{side}_actor_id"]
                prompt = row[f"{side}_prompt_actor_context"].strip()
                image_name = f"{side}_seed{seed:04d}.png"
                image_path = IMAGE_OUT_DIR / row["pair_id"] / row["actor_key"] / image_name
                jobs.append(
                    {
                        "pair_actor_id": row["pair_actor_id"],
                        "pair_id": row["pair_id"],
                        "actor_key": row["actor_key"],
                        "topic": row.get("topic", ""),
                        "source": row.get("source", ""),
                        "side": side,
                        "actor_id": actor_id,
                        "actor_label": row.get(f"{side}_actor_label", ""),
                        "final_frame": row.get(f"{side}_final_frame", ""),
                        "final_t2i_label": row.get(f"{side}_final_t2i_label", ""),
                        "is_named_person": row.get(f"{side}_is_named_person", ""),
                        "context_sentence": row.get(f"{side}_context_sentence", ""),
                        "prompt": prompt,
                        "seed": seed,
                        "seed_iteration": seed_index,
                        "image_path": str(image_path),
                        "mivolo_gender": "",
                        "mivolo_age": "",
                    }
                )
    return jobs


def get_device_and_dtype() -> tuple[str, object]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_pipeline(device: str, dtype) -> AutoPipelineForText2Image:
    print(f"Loading Flux pipeline ({MODEL_ID}) on {device}...")
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    print("Model loaded.\n")
    return pipe


def generate_one(pipe, prompt: str, seed: int, out_path: Path, device: str) -> None:
    if out_path.exists():
        return
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)


def write_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "pair_actor_id",
        "pair_id",
        "actor_key",
        "topic",
        "source",
        "side",
        "actor_id",
        "actor_label",
        "final_frame",
        "final_t2i_label",
        "is_named_person",
        "context_sentence",
        "prompt",
        "seed",
        "seed_iteration",
        "image_path",
        "mivolo_gender",
        "mivolo_age",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    rows = load_rows(INPUT_FILE)
    if not rows:
        print("No paired prompt rows found. Run build_actor_prompts.py first.")
        return

    seeds = DEFAULT_SEEDS
    eligible_rows = [row for row in rows if is_eligible_pair(row)]
    jobs = iter_generation_jobs(rows, seeds)

    print(
        f"Loaded {len(rows)} paired actor rows -> {len(eligible_rows)} eligible paired actors "
        f"({len(jobs)} planned images). Seeds: {seeds}"
    )

    pipe = None
    if not args.dry_run:
        if not DIFFUSERS_AVAILABLE:
            raise SystemExit("diffusers/torch not available. Install them or use --dry-run.")
        device, dtype = get_device_and_dtype()
        pipe = load_pipeline(device, dtype)
    else:
        device = "cpu"

    for job in tqdm(jobs):
        if args.dry_run:
            continue
        generate_one(
            pipe=pipe,
            prompt=job["prompt"],
            seed=job["seed"],
            out_path=Path(job["image_path"]),
            device=device,
        )

    write_metadata(METADATA_OUT, jobs)
    mode = "Planned" if args.dry_run else "Generated"
    print(f"{mode} {len(jobs)} images.")
    print(f"Metadata saved to: {METADATA_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired T2I images for the main LS vs normal experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Plan images but skip generation.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
