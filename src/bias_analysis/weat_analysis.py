#!/usr/bin/env python3
"""Compute WEAT and ML-EAT scores on actor predications with visualizations."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import spacy

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

DEFAULT_INPUT = Path("data/processed/actor_features.csv")
DEFAULT_OUTPUT = Path("data/processed/weat_results.json")
PLOT_OUTPUT_DIR = Path("data/processed/plots")

# State-of-the-Art German Attribute Sets based on culturally adapted translations by Kurpicz-Briki (2020) and SCM
ATTRIBUTES = {
    # Karriere vs. Familie (Kurpicz-Briki 2020)
    "career": ["Karriere", "Firma", "Gehalt", "Büro", "Geschäft", "Management", "professionell", "Beruf"],
    "family": ["Zuhause", "Eltern", "Kinder", "Familie", "Cousin", "Ehe", "Hochzeit", "Verwandte"],
    # Kompetenz vs. Wärme (Standardisiertes Stereotype Content Model / WEFE Framework)
    "competence": ["kompetent", "intelligent", "fähig", "effizient", "selbstbewusst", "logisch", "objektiv", "rational", "klug"],
    "warmth": ["warmherzig", "freundlich", "aufmerksam", "fürsorglich", "mitfühlend", "emotional", "sanft", "empathisch", "lieb"]
}

MASCULINE_WORDS = {"mann", "männer", "bürger", "mitarbeiter", "kunde", "kunden", "patient", "patienten"}
FEMININE_WORDS = {"frau", "frauen", "bürgerin", "bürgerinnen", "mitarbeiterin", "kundin", "patientin", "patientinnen"}

def deduce_gender(row):
    """Fallback gender deduction if coreference resolving wasn't used to find pronouns."""
    pc = row.get("main_pronoun_class", "unknown")
    if pc in ["masculine", "feminine"]:
        return pc
    
    noms = [n.strip().lower() for n in row.get("nominations", "").split("|")]
    
    for nom in noms:
        if nom in FEMININE_WORDS:
            return "feminine"
            
    for nom in noms:
        if nom in MASCULINE_WORDS:
            return "masculine"
            
    return "unknown"


def load_vectors(nlp, words):
    vecs = []
    for w in words:
        tokens = nlp(w)
        if tokens and tokens[0].has_vector:
            vecs.append(tokens[0].vector)
    return np.array(vecs) if vecs else np.array([])


def cos_sim(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def weat_association(target_vec, attr_A_vecs, attr_B_vecs):
    # Mean cosine similarity of target to attributes A minus mean cosine similarity to attributes B
    mean_A = np.mean([cos_sim(target_vec, a) for a in attr_A_vecs]) if len(attr_A_vecs) > 0 else 0
    mean_B = np.mean([cos_sim(target_vec, b) for b in attr_B_vecs]) if len(attr_B_vecs) > 0 else 0
    return mean_A - mean_B


def weat_effect_size(masc_assoc, fem_assoc):
    # Level 1 Effect Size (Caliskan 2017)
    if not len(masc_assoc) or not len(fem_assoc):
        return 0.0
    
    mean_diff = np.mean(masc_assoc) - np.mean(fem_assoc)
    all_assoc = masc_assoc + fem_assoc
    std_dev = np.std(all_assoc, ddof=1) if len(all_assoc) > 1 else 0
    
    if std_dev == 0:
        return 0.0
    return float(mean_diff / std_dev)


def m_eat_level2(assoc_list):
    # Level 2 Differential Association of a single target with A vs B (arXiv:2408.01966)
    if not len(assoc_list):
        return {"effect_size": 0.0}
    
    mean_diff = np.mean(assoc_list)
    std_dev = np.std(assoc_list, ddof=1) if len(assoc_list) > 1 else 0
    
    if std_dev == 0:
        return {"effect_size": 0.0}
    
    return {"effect_size": float(mean_diff / std_dev)}


def plot_associations(masc_assoc, fem_assoc, A_name, B_name, side, out_dir):
    if not PLOTTING_AVAILABLE:
        return
        
    plt.figure(figsize=(8, 5))
    
    # Kernel Density Estimation Plot
    if len(masc_assoc) > 1:
        sns.kdeplot(masc_assoc, fill=True, label="Männliche Akteure", color="blue", alpha=0.5)
    if len(fem_assoc) > 1:
        sns.kdeplot(fem_assoc, fill=True, label="Weibliche Akteure", color="orange", alpha=0.5)
    
    plt.axvline(x=0, color='gray', linestyle='--')
    
    plt.title(f"Verteilung der Assoziationen [{side.upper()}]\\n(Negative: {B_name} <---> Positive: {A_name})", pad=15)
    plt.xlabel(f"Differentieller Bias Score")
    plt.ylabel("Dichte (Häufigkeit)")
    plt.legend()
    plt.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"bias_dist_{side}_{A_name.lower()}_{B_name.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def process_side(rows, side, nlp, attr_A_vecs, attr_B_vecs, A_name, B_name):
    masc_preds = []
    fem_preds = []
    
    for row in rows:
        if row["side"] != side:
            continue
        
        preds = [p.strip() for p in row["predications"].split("||") if p.strip()]
        if not preds:
            continue
            
        gender = deduce_gender(row)
            
        # extract vectors for meaning-bearing words in the predications
        for p in preds:
            doc = nlp(p)
            for token in doc:
                if token.is_alpha and not token.is_stop and token.has_vector:
                    if gender == "masculine":
                        masc_preds.append(token.vector)
                    elif gender == "feminine":
                        fem_preds.append(token.vector)
    
    # Random sampling to balance and speed up computation (limit to 800)
    random.shuffle(masc_preds)
    random.shuffle(fem_preds)
    
    limit = min(800, max(len(masc_preds), len(fem_preds)))
    masc_vecs = np.array(masc_preds[:limit])
    fem_vecs = np.array(fem_preds[:limit])
    
    # Precompute associations
    masc_assoc = [weat_association(x, attr_A_vecs, attr_B_vecs) for x in masc_vecs]
    fem_assoc = [weat_association(y, attr_A_vecs, attr_B_vecs) for y in fem_vecs]
    
    # Plotting
    plot_associations(masc_assoc, fem_assoc, A_name, B_name, side, PLOT_OUTPUT_DIR)
    
    # WEAT Level 1 (Overall Effect Size)
    es_l1 = weat_effect_size(masc_assoc, fem_assoc)
    
    # ML-EAT Level 2 (Differential Effect per Gender)
    es_masc_l2 = m_eat_level2(masc_assoc)
    es_fem_l2 = m_eat_level2(fem_assoc)
    
    return {
        "side": side,
        "test": f"Masculine vs. Feminine | {A_name} vs. {B_name}",
        "WEAT_Level_1_EffectSize": es_l1,
        "MLEAT_Level_2_EffectSize_Masc": es_masc_l2["effect_size"],
        "MLEAT_Level_2_EffectSize_Fem": es_fem_l2["effect_size"],
        "n_masc_tokens": len(masc_vecs),
        "n_fem_tokens": len(fem_vecs)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="de_core_news_lg")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return

    try:
        nlp = spacy.load(args.model)
    except OSError:
        fallback_model = "de_core_news_md"
        print(f"Warning: spaCy model '{args.model}' not found. Attempting to load '{fallback_model}'.")
        try:
            nlp = spacy.load(fallback_model)
        except OSError:
            print(f"Error: Neither '{args.model}' nor '{fallback_model}' are installed.")
            print(f"Please run: python -m spacy download {args.model}")
            return

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} actor features.")
    print("Computing attribute vector representations...")
    career_vecs = load_vectors(nlp, ATTRIBUTES["career"])
    family_vecs = load_vectors(nlp, ATTRIBUTES["family"])
    comp_vecs = load_vectors(nlp, ATTRIBUTES["competence"])
    warmth_vecs = load_vectors(nlp, ATTRIBUTES["warmth"])

    results = {}
    
    for side in ["simple", "normal"]:
        print(f"Running WEAT tests for side: {side.upper()}")
        res_car_fam = process_side(rows, side, nlp, career_vecs, family_vecs, "Career", "Family")
        res_comp_warm = process_side(rows, side, nlp, comp_vecs, warmth_vecs, "Competence", "Warmth")
        
        results[side] = [res_car_fam, res_comp_warm]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\\nAnalysis completed. Results saved to {args.output}")
    if PLOTTING_AVAILABLE:
        print(f"Visualizations saved to {PLOT_OUTPUT_DIR}")
    else:
        print("Note: Install 'matplotlib' and 'seaborn' to automatically generate distribution plots.")
        
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
