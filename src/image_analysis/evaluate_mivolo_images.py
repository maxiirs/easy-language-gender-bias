#!/usr/bin/env python3
"""
Evaluate generated images using MiVOLO and map to discrete gender categories.

This script takes the generated images listed in `t2i_metadata.csv`, runs them 
through the MiVOLO age/gender predictor, and maps the detections back to the 
six identical text categories (e.g., 'female (singular)', 'female group') 
for direct alignment and Chi-Square statistical testing.

Prerequisites on your JupyterHub:
1. Since MiVOLO's pip installer is currently broken in modern Python versions, clone the repo into the project root: 
   `git clone https://github.com/WildChlamydia/MiVOLO.git`
2. Ensure you have the weights downloaded (yolov8x_person_face.pt and model_imdb_cross_person_4.22_99.46.pth.tar).
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

T2I_LABEL_TO_CATEGORY = {
    "f1": "female (singular)",
    "m1": "male (singular)",
    "fg": "female group",
    "mg": "male group",
    "mx": "mixed/generic group",
    "ux": "mixed/generic group",
}


def determine_category(males: int, females: int) -> str:
    """
    Maps the physical detected counts to our precise textual categories.
    """
    total = males + females
    
    if total == 0:
        return "no person"
        
    if total == 1:
        if females == 1:
            return "female (singular)"
        else:
            return "male (singular)"
    
    # Plural cases (total > 1)
    if females > 0 and males == 0:
        return "female group"
    elif males > 0 and females == 0:
        return "male group"
    else:
        return "mixed/generic group"


def get_text_intended_category(row: dict) -> str:
    """Map current metadata fields back to canonical text-side image categories."""
    label = (row.get("final_t2i_label") or "").strip()
    if label:
        return T2I_LABEL_TO_CATEGORY.get(label, label)

    # Backward compatibility with older metadata schema.
    legacy = (row.get("predicted_gender") or "").strip()
    if legacy:
        return legacy
    return ""


def get_dominant_person(detected_objects):
    """
    Find the most prominent person and return its predicted gender and age.

    Strategy: compare ALL face bounding boxes by area (largest = closest to
    camera = most prominent).  Face boxes are all the same type of detection
    so their areas are directly comparable - no scale-mixing issues.

    Why face boxes and not person boxes?
        - Every MiVOLO-classifiable person must have a face detection
          (gender is predicted from the face crop, with optional body context).
        - Face boxes are uniformly sized relative to subject distance, so a
          larger face bbox reliably means that person is closer / more prominent.
        - Body boxes vary wildly in aspect ratio and are not comparable to face
          boxes, so mixing them would let a distant person with a full-body box
          beat a foreground person detected only by face.

    Returns:
        dominant_gender (str | None):  "male" / "female" / None
        dominant_age    (float | None): predicted age or None
    """
    face_inds = detected_objects.get_bboxes_inds("face")

    if not face_inds:
        return None, None

    best_idx, best_area = None, -1
    for idx in face_inds:
        try:
            x1, y1, x2, y2 = (int(v) for v in
                               detected_objects.get_bbox_by_ind(idx).cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
        except Exception:
            continue
        if area > best_area:
            best_area = area
            best_idx  = idx

    if best_idx is None:
        return None, None

    return detected_objects.genders[best_idx], detected_objects.ages[best_idx]

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
                self.detector_weights = args.yolo_weights
                self.checkpoint = args.mivolo_weights
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

    # ── Optional row limit for quick smoke-tests ──────────────────────────────
    if args.limit:
        rows = rows[:args.limit]
        print(f"[--limit] Processing only the first {len(rows)} rows.")

    print(f"Processing {len(rows)} images...")
    succ_opened = 0
    skipped = 0

    for row in tqdm(rows):
        img_path = Path(row["image_path"])
        
        m_count = 0
        f_count = 0
        
        dom_gender, dom_age = None, None   # populated when MiVOLO runs

        if not args.dry_run and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                succ_opened += 1
                try:
                    detected_objects, _ = predictor.recognize(img)

                    if hasattr(detected_objects, 'genders'):
                        # MiVOLO assigns gender to BOTH the person-box index AND the
                        # face-box index for the same physical person (see structures.py).
                        # Iterating detected_objects.genders directly would double-count.
                        # Fix: count only person-category boxes; for detections where no
                        # body was found, also count the unmatched face-only boxes.
                        person_inds = detected_objects.get_bboxes_inds("person")
                        face_inds   = detected_objects.get_bboxes_inds("face")
                        unmatched_face_inds = [
                            fi for fi in face_inds
                            if detected_objects.face_to_person_map.get(fi) is None
                        ]
                        for idx in list(person_inds) + unmatched_face_inds:
                            g = detected_objects.genders[idx]
                            if g is not None and g.lower() == "male":
                                m_count += 1
                            elif g is not None:
                                f_count += 1

                        # Dominant person: largest bbox → fills mivolo_gender / mivolo_age
                        dom_gender, dom_age = get_dominant_person(detected_objects)
                    else:
                        # Iterable Persons format (older MiVOLO versions)
                        for person in detected_objects.get_results():
                            if person.gender is not None:
                                if person.gender.lower() == "male":
                                    m_count += 1
                                else:
                                    f_count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                skipped += 1
        else:
            skipped += 1

        # Determine alignment category
        img_cat = determine_category(m_count, f_count)
        
        # Determine specific text intended category from the text metadata
        txt_pred = get_text_intended_category(row)
        
        # Did the generated image perfectly match the text expectation?
        # (Assuming the text string from the previous extraction step hasn't changed)
        alignment_match = (img_cat == txt_pred)
        
        # Populate our output dictionaries
        out_row = dict(row)
        out_row["mivolo_gender"] = dom_gender if dom_gender is not None else ""  # dominant face bbox
        out_row["mivolo_age"]    = round(dom_age, 1) if dom_age is not None else ""  # dominant age
        out_row["mivolo_males"] = m_count
        out_row["mivolo_females"] = f_count
        out_row["mivolo_total_persons"] = m_count + f_count
        out_row["image_gender_category"] = img_cat
        out_row["text_intended_category"] = txt_pred
        out_row["semantic_match"] = alignment_match
        
        updated_rows.append(out_row)

    # Save ready-to-test CSV
    # Always opened with "w" → overwrites any previous run.
    # When --limit is active a separate _test file is written so the real
    # results are never corrupted during a smoke-test.
    fieldnames = list(updated_rows[0].keys())

    out_path = RESULTS_OUT
    if args.limit:
        out_path = RESULTS_OUT.with_stem(RESULTS_OUT.stem + "_test")
        print(f"[--limit] Writing test output to: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\nCompleted. Dataset saved to: {out_path}")
    
    # Quick Summary Print
    cat_counts = {}
    for r in updated_rows:
        c = r["image_gender_category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        
    print(f"\nDebug Stats:")
    print(f" - Successfully Read by OpenCV: {succ_opened}")
    print(f" - Skipped (File NOT Found or Read Error): {skipped}")
    
    print("\nGenerated Image Categories Summary:")
    for k, v in cat_counts.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run MiVOLO on generated dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run the loop without model execution")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--yolo-weights", default="yolov8x_person_face.pt", help="Path to YOLO weights")
    parser.add_argument("--mivolo-weights", default="model_imdb_cross_person_4.22_99.46.pth.tar", help="Path to MiVOLO weights")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (writes to *_test.csv, safe for smoke-testing).")

    args = parser.parse_args()
    run(args)
