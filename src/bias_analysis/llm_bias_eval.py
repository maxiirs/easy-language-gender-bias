#!/usr/bin/env python3
"""Evaluate actor predications for gender bias using a local LLM (HuggingFace Transformers)."""

import argparse
import csv
import json
import os
from pathlib import Path

# Fix for macOS OpenMP libomp.dylib initialization crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to import torch and transformers gracefully
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

DEFAULT_INPUT = Path("data/processed/actor_features.csv")
DEFAULT_OUTPUT = Path("data/processed/llm_bias_results.json")

MASCULINE_WORDS = {"mann", "männer", "bürger", "mitarbeiter", "kunde", "kunden", "patient", "patienten"}
FEMININE_WORDS = {"frau", "frauen", "bürgerin", "bürgerinnen", "mitarbeiterin", "kundin", "patientin", "patientinnen"}

def deduce_gender(row):
    pc = row.get("main_pronoun_class", "unknown")
    if pc in ["masculine", "feminine"]:
        return pc
    
    noms = [n.strip().lower() for n in row.get("nominations", "").split("|")]
    for nom in noms:
        if nom in FEMININE_WORDS: return "feminine"
    for nom in noms:
        if nom in MASCULINE_WORDS: return "masculine"
    return "unknown"

def build_prompt(sentence, dimension):
    if dimension == "Career_Family":
        choices = "'Career' or 'Family'"
    else:
        choices = "'Competence' or 'Warmth'"
        
    messages = [
        {"role": "system", "content": f"You are an objective linguistic evaluator. Your task is to classify whether a sentence associates a person more with {choices}. You MUST output exactly one word from the choices, nothing else."},
        {"role": "user", "content": f"Sentence: \"{sentence}\"\\nClassification:"}
    ]
    return messages

def main():
    # Load HFToken from .env file if it exists so pipelines can authenticate automatically
    env_path = Path(".env")
    if env_path.exists():
        import os
        with env_path.open() as f:
            for line in f:
                if line.strip().startswith("HFTOKEN="):
                    token = line.strip().split("=", 1)[1].strip()
                    token = token.strip("\"'") # Remove quotes if the user added them
                    os.environ["HF_TOKEN"] = token
                    print("Loaded HuggingFace Token from .env file.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    # Limiting for performance on local laptop. 50 items per quadrant = 200 inferences per side.
    parser.add_argument("--limit", type=int, default=50) 
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        return

    if not TRANSFORMERS_AVAILABLE:
        print("Error: HuggingFace 'transformers' or 'torch' not installed.")
        print("Please run: pip install transformers torch accelerate")
        return

    print(f"Loading local model '{args.model}' via HuggingFace...")
    print("Note: This uses device_map='auto'. On Apple Silicon, this will load to MPS (GPU) via accelerate if set up properly.")
    
    try:
        generator = pipeline(
            "text-generation",
            model=args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        print(f"\\nError loading model. Make sure you have authenticated via `huggingface-cli login`, have access to the model, and installed 'accelerate'.")
        print(f"Details: {e}")
        return

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pre-bin sentences
    data = {"simple": {"masculine": [], "feminine": []}, "normal": {"masculine": [], "feminine": []}}
    
    for row in rows:
        side = row["side"]
        gender = deduce_gender(row)
        if gender in ["masculine", "feminine"]:
            preds = [p.strip() for p in row["predications"].split("||") if p.strip()]
            for p in preds:
                # filter out extremely short noise sentences
                if len(p.split()) > 3:
                    data[side][gender].append(p)

    results = {"simple": {}, "normal": {}}

    print(f"\\nStarting LLM zero-shot inference (max {args.limit} samples per bucket)...")
    for side in ["simple", "normal"]:
        for dim, pairs in [("Career_Family", ("Career", "Family")), ("Competence_Warmth", ("Competence", "Warmth"))]:
            scores = {"masculine": {pairs[0]: 0, pairs[1]: 0}, "feminine": {pairs[0]: 0, pairs[1]: 0}}
            
            for gender in ["masculine", "feminine"]:
                sentences = data[side][gender][:args.limit]
                # Fallback to tqdm for progress bar if installed
                iterable = tqdm(sentences, desc=f"{side[:4].upper()} {gender[:4].capitalize()} {pairs[0]}/{pairs[1]}") 
                
                for sent in iterable:
                    prompt = build_prompt(sent, dim)
                    out = generator(prompt, max_new_tokens=5, do_sample=False, return_full_text=False)
                    resp = out[0]["generated_text"].strip()
                    
                    if pairs[0].lower() in resp.lower():
                        scores[gender][pairs[0]] += 1
                    elif pairs[1].lower() in resp.lower():
                        scores[gender][pairs[1]] += 1
            
            # calculate percentage
            for gender in ["masculine", "feminine"]:
                total = scores[gender][pairs[0]] + scores[gender][pairs[1]]
                if total > 0:
                    scores[gender][f"{pairs[0]}_percentage"] = round(scores[gender][pairs[0]] / total * 100, 2)
                    scores[gender][f"{pairs[1]}_percentage"] = round(scores[gender][pairs[1]] / total * 100, 2)
                scores[gender]["total_inferred"] = total
            
            results[side][dim] = scores

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\\nLLM Bias Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
