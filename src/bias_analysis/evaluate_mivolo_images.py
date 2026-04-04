#!/usr/bin/env python3
"""
Evaluate generated images using MiVOLO and map to discrete gender categories.

This script takes the generated images listed in `t2i_metadata.csv`, runs them 
through the MiVOLO age/gender predictor, and maps the detections back to the 
six identical text categories (e.g., 'weiblich (Singular)', 'Frauengruppe') 
for direct alignment and Chi-Square statistical testing.

Prerequisites on your JupyterHub:
1. Since MiVOLO's pip installer is currently broken in modern Python versions, clone the repo into the project root: 
   `git clone https://github.com/WildChlamydia/MiVOLO.git`
2. Ensure you have the weights downloaded (yolov8x_person_face.pt and mivolo_imb_weight.pth.tar).
"""

import argparse
import csv
import sys
import os
from pathlib import Path
import cv2

# Inject local MiVOLO into python path so we don't have to cleanly pip install it
sys.path.append(os.path.abspath("MiVOLO"))

# ── YOU MAY NEED TO ADJUST THE IMPORT BASED ON YOUR LOCAL MIVOLO INSTALL ──
try:
    from mivolo.predictor import Predictor
    import mivolo.structures as mivolo_structs

except ImportError:
    print("WARNING: MiVOLO not found in python path.")
    print("Please ensure you are running this from the MiVOLO directory")
    print("or that MiVOLO is installed in your conda environment.")
    MIVOLO_AVAILABLE = False
else:
    MIVOLO_AVAILABLE = True

try:
    from tqdm import tqdm
except ImportError:
    tqdm = list

METADATA_IN = Path("data/processed/t2i_metadata.csv")
RESULTS_OUT = Path("data/processed/t2i_mivolo_results.csv")


def determine_category(males: int, females: int) -> str:
    """
    Maps the physical detected counts to our precise textual categories.
    """
    total = males + females
    
    if total == 0:
        return "Keine Person"
        
    if total == 1:
        if females == 1:
            return "weiblich (Singular)"
        else:
            return "maskulin (Singular)"
    
    # Plural cases (total > 1)
    if females > 0 and males == 0:
        return "Frauengruppe"
    elif males > 0 and females == 0:
        return "Männergruppe"
    else:
        return "gemischte/generische Gruppe"


def run(args):
    if not METADATA_IN.exists():
        print(f"Input file {METADATA_IN} not found. Generate images first.")
        return

    # Load existing metadata
    with METADATA_IN.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if MIVOLO_AVAILABLE and not args.dry_run:
        # Initialize MiVOLO predictor
        # Note: You might need to supply a dummy config object depending on the MiVOLO version
        class DummyConfig:
            def __init__(self):
                self.yolo_model = args.yolo_weights
                self.model = args.mivolo_weights
                self.device = args.device
                self.half = True
                self.with_persons = True
                self.disable_faces = False
                self.draw = False
        
        config = DummyConfig()
        print(f"Loading MiVOLO models on {args.device}...")
        try:
            predictor = Predictor(config)
        except Exception as e:
            print(f"Could not load MiVOLO predictor. Error: {e}")
            sys.exit(1)
    else:
        print("Running in DRY RUN mode (MiVOLO missing or --dry-run specified).")

    updated_rows = []
    
    print(f"Processing {len(rows)} images...")
    for row in tqdm(rows):
        img_path = Path(row["image_path"])
        
        m_count = 0
        f_count = 0
        
        if not args.dry_run and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                try:
                    detected_objects, _ = predictor.recognize(img)
                    
                    # MiVOLO detected objects usually act as a list of Person objects
                    # or an object possessing vectors of ages/genders
                    # (Implementation might vary slightly per MiVOLO github commit)
                    if hasattr(detected_objects, 'genders'):
                        # Batch return format
                        genders = detected_objects.genders
                        for g in genders:
                            if g.lower() == "m" or g == 0: # Note: Adjust based on MiVOLO output format
                                m_count += 1
                            else:
                                f_count += 1
                    else:
                        # Iterable Persons format
                        for person in detected_objects.get_results():
                            if person.gender is not None:
                                if person.gender == "M":
                                    m_count += 1
                                elif person.gender == "F":
                                    f_count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Determine alignment category
        img_cat = determine_category(m_count, f_count)
        
        # Determine specific text intended category from the text prediction
        txt_pred = row.get("predicted_gender", "").strip()
        
        # Did the generated image perfectly match the text expectation?
        # (Assuming the text string from the previous extraction step hasn't changed)
        alignment_match = (img_cat == txt_pred)
        
        # Populate our output dictionaries
        out_row = dict(row)
        out_row["mivolo_males"] = m_count
        out_row["mivolo_females"] = f_count
        out_row["mivolo_total_persons"] = m_count + f_count
        out_row["image_gender_category"] = img_cat
        out_row["text_intended_category"] = txt_pred
        out_row["semantic_match"] = alignment_match
        
        updated_rows.append(out_row)

    # Save ready-to-test CSV
    fieldnames = list(updated_rows[0].keys())
    
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\nCompleted. Final dataset for Chi-Square tests saved to: {RESULTS_OUT}")
    
    # Quick Summary Print
    cat_counts = {}
    for r in updated_rows:
        c = r["image_gender_category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        
    print("\nGenerated Image Categories Summary:")
    for k, v in cat_counts.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run MiVOLO on generated dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run the loop without model execution")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--yolo-weights", default="yolov8x_person_face.pt", help="Path to YOLO weights")
    parser.add_argument("--mivolo-weights", default="mivolo_imb_weight.pth.tar", help="Path to MiVOLO weights")
    
    args = parser.parse_args()
    run(args)
