#!/usr/bin/env python3
"""
Generate Multilingual Text-to-Image (T2I) Pairs based on the MAGBIG Framework.
This script samples unbiased text pairs from the Simple Language corpus
and generates images using a multilingual text-to-image model (e.g., AltDiffusion).
"""

import os
import csv
import json
import random
from pathlib import Path

# Try to import diffusers and torch
try:
    import torch
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
    
    # Workaround für den "cudnnGetLibConfig" cuDNN Bug auf Tesla V100 Jupyter-Servern
    torch.backends.cudnn.enabled = False
    
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: 'diffusers' or 'torch' not found.")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = list

INPUT_FILE = Path("data/processed/actor_features.csv")
PROMPT_OUT_FILE = Path("data/processed/t2i_prompts_sample.json")
IMAGE_OUT_DIR = Path("data/processed/t2i_images")

SAMPLE_SIZE_PER_STRATUM = 10 

# Wir nutzen FLUX.1 [schnell] von Black Forest Labs.
# Das derzeit beste Open-Source Text-to-Image Modell mit nativ exzellentem
# Deutschverständnis durch den massiven T5-XXL Text-Encoder.
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def build_stratified_sample():
    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pairs = {}
    for row in rows:
        key = f"{row['pair_id']}_{row['actor_label']}"
        if key not in pairs:
            pairs[key] = {"simple": None, "normal": None, "gender": row["main_pronoun_class"]}
        
        preds = [p.strip() for p in row["predications"].split("||") if p.strip()]
        if preds:
            pairs[key][row["side"]] = preds[0]  

    valid_pairs = []
    for key, data in pairs.items():
        if data["simple"] and data["normal"] and len(data["simple"]) > 10 and len(data["normal"]) > 10:
            valid_pairs.append({
                "id": key,
                "gender_category": data["gender"],
                "prompt_normal": data["normal"],
                "prompt_simple": data["simple"]
            })

    strata = {"masculine": [], "feminine": [], "unknown": [], "mixed_or_unclear": []}
    for p in valid_pairs:
        cat = p["gender_category"]
        if cat in strata:
            strata[cat].append(p)
        else:
            strata["unknown"].append(p)

    random.seed(42)  
    sampled = []
    for cat, items in strata.items():
        if not items: continue
        chosen = random.sample(items, min(len(items), SAMPLE_SIZE_PER_STRATUM))
        sampled.extend(chosen)

    return sampled


def generate_images(prompts):
    if not DIFFUSERS_AVAILABLE:
        return

    try:
        from transformers import BitsAndBytesConfig, T5EncoderModel
        from diffusers import FluxTransformer2DModel
        BNB_AVAILABLE = True
    except ImportError:
        BNB_AVAILABLE = False
        print("Warning: 'bitsandbytes' not installed. Fallback to slow sequential loading.")

    print(f"\nLoading State-of-the-Art T2I Model ({MODEL_ID})...")
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    if device == "cuda" and BNB_AVAILABLE:
        print("Enable 8-bit Quantization for V100 Fast-Inference...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Lade speziell die massiven 24GB Bauteile (Transformer & T5) in gecrunchten 8-Bit (jeweils ~12GB und ~4.5GB), 
        # damit sie voll am Stück in den VRAM der V100 passen.
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, 
            subfolder="transformer", 
            quantization_config=quantization_config, 
            torch_dtype=dtype
        )
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            MODEL_ID, 
            subfolder="text_encoder_2", 
            quantization_config=quantization_config, 
            torch_dtype=dtype
        )
        
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, 
            transformer=transformer, 
            text_encoder_2=text_encoder_2,
            torch_dtype=dtype
        )
        
        # Model-Offload verschiebt die "Puzzleteile" im Ganzen (schnell), statt Layer-für-Layer (langsam).
        # Das klappt, weil der größte Block (der 12GB Transformer) nun in die 16GB der V100 passt!
        pipe.enable_model_cpu_offload()
    else:
        # Fallback (Slow Sequential)
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if device == "cuda":
            try:
                pipe.enable_sequential_cpu_offload()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception as e:
                print(f"Offload warning: {e}")
                pipe = pipe.to(device)

    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating images for {len(prompts)} pairs...")
    for item in tqdm(prompts):
        base_id = item["id"]
        
        # 1. Generate image for Standard German (Normal)
        normal_path = IMAGE_OUT_DIR / f"{base_id}_flux_normal.png"
        if not normal_path.exists():
            # FLUX.1-schnell generiert Bilder extrem schnell in exakt 4 Steps.
            image_normal = pipe(
                item["prompt_normal"], 
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256
            ).images[0]
            image_normal.save(normal_path)
            
        # 2. Generate image for Easy German (Simple)
        simple_path = IMAGE_OUT_DIR / f"{base_id}_flux_simple.png"
        if not simple_path.exists():
            image_simple = pipe(
                item["prompt_simple"], 
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256
            ).images[0]
            image_simple.save(simple_path)


def main():
    print("1. Extracting stratified random sample from actor_features.csv...")
    prompts = build_stratified_sample()
    
    print(f"   Selected {len(prompts)} valid text-pairs for the benchmark.")
    
    with PROMPT_OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"   Saved prompts reference to {PROMPT_OUT_FILE}")

    generate_images(prompts)
    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
