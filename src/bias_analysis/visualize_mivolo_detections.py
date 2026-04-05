#!/usr/bin/env python3
"""
visualize_mivolo_detections.py
-------------------------------
Picks N random images from the t2i image pool, runs them through MiVOLO,
and draws person/face bounding boxes annotated with predicted gender and age.

Usage (from the project root, inside the MiVOLO-compatible conda env):
    python src/bias_analysis/visualize_mivolo_detections.py

    # Optional overrides:
    python src/bias_analysis/visualize_mivolo_detections.py --n 5 --seed 7 --device cuda

Weight file defaults (same as evaluate_mivolo_images.py):
    YOLO detector : yolov8x_person_face.pt
    MiVOLO model  : model_imdb_cross_person_4.22_99.46.pth.tar
Both files are expected to be in the project root (same convention).

Prerequisites
-------------
* Clone MiVOLO into the project root:
      git clone https://github.com/WildChlamydia/MiVOLO.git
* Activate the conda env that has torch, ultralytics, timm, opencv-python, etc.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Inject local MiVOLO repo ──────────────────────────────────────────────────
MIVOLO_REPO = Path(__file__).resolve().parents[2] / "MiVOLO"
if MIVOLO_REPO.exists():
    sys.path.insert(0, str(MIVOLO_REPO))
else:
    # Fallback: try CWD/MiVOLO
    sys.path.insert(0, str(Path.cwd() / "MiVOLO"))

try:
    from mivolo.predictor import Predictor
    from mivolo.structures import PersonAndFaceResult
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import MiVOLO: {e}\n"
        "Make sure MiVOLO is cloned into the project root and you are running "
        "inside the correct conda environment."
    )

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
IMAGES_DIR     = PROJECT_ROOT / "data" / "processed" / "t2i_images"
DEFAULT_OUT    = PROJECT_ROOT / "data" / "processed" / "plots" / "mivolo_viz.png"

# Weight file defaults – mirrors evaluate_mivolo_images.py
DEFAULT_YOLO_WEIGHTS   = "yolov8x_person_face.pt"
DEFAULT_MIVOLO_WEIGHTS = "model_imdb_cross_person_4.22_99.46.pth.tar"

# Gender → box colour (BGR)
GENDER_COLORS = {
    "male":   (255, 100,  50),   # blue-ish
    "female": ( 60, 180, 255),   # warm orange
    "unknown":(180, 180, 180),
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_random_images(n: int, seed: int) -> list[Path]:
    """Return *n* random .png paths from the t2i images folder."""
    all_imgs = list(IMAGES_DIR.rglob("*.png"))
    if not all_imgs:
        sys.exit(f"[ERROR] No images found under {IMAGES_DIR}")
    random.seed(seed)
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"[INFO] Selected {len(chosen)} image(s):")
    for p in chosen:
        print(f"       {p.relative_to(PROJECT_ROOT)}")
    return chosen


def build_predictor(args) -> Predictor:
    """Instantiate the MiVOLO Predictor via a DummyConfig object."""
    yolo_weights   = args.yolo_weights
    mivolo_weights = args.mivolo_weights

    # Resolve relative paths against the project root (same as evaluate script)
    if not Path(yolo_weights).is_absolute():
        yolo_weights = str(PROJECT_ROOT / yolo_weights)
    if not Path(mivolo_weights).is_absolute():
        mivolo_weights = str(PROJECT_ROOT / mivolo_weights)

    print(f"[INFO] YOLO weights  : {yolo_weights}")
    print(f"[INFO] MiVOLO weights: {mivolo_weights}")

    class DummyConfig:
        detector_weights = yolo_weights
        checkpoint       = mivolo_weights
        device           = args.device
        half             = False          # safer on CPU / MPS
        with_persons     = True
        disable_faces    = False
        draw             = False          # we do our own drawing

    print(f"[INFO] Loading MiVOLO on '{args.device}' …")
    return Predictor(DummyConfig())


def gender_label(detected_objects: "PersonAndFaceResult", idx: int) -> str:
    """
    Extract the gender string for detection index *idx*.
    MiVOLO's PersonAndFaceResult stores results in per-person tensors/lists;
    'gender' is either a string ('male'/'female') or None.
    """
    try:
        g = detected_objects.gender[idx]
        if g is None:
            return "unknown"
        return str(g).lower()
    except (IndexError, AttributeError, TypeError):
        return "unknown"


def age_label(detected_objects: "PersonAndFaceResult", idx: int) -> str:
    """Extract the age value for detection index *idx*."""
    try:
        a = detected_objects.ages[idx]
        if a is None:
            return "?"
        return f"{int(round(float(a)))}"
    except (IndexError, AttributeError, TypeError):
        return "?"


def draw_boxes(img_bgr: np.ndarray, detected_objects: "PersonAndFaceResult") -> np.ndarray:
    """
    Draw bounding boxes + gender/age labels onto *img_bgr* (in-place copy).

    MiVOLO's PersonAndFaceResult exposes:
        .yolo_results         – raw ultralytics Results object
        .persons_bboxes       – list of xyxy tensors for body boxes
        .faces_bboxes         – list of xyxy tensors for face boxes
        .gender               – list / 1-D tensor of gender strings or ints
        .ages                 – list / 1-D tensor of age floats
    The indices align between persons_bboxes, faces_bboxes, gender, ages.
    """
    canvas = img_bgr.copy()
    n_persons = 0

    # ── Try persons_bboxes first, fall back to faces_bboxes ──────────────────
    bboxes = getattr(detected_objects, "persons_bboxes", None) or []
    source_label = "person"
    if len(bboxes) == 0:
        bboxes = getattr(detected_objects, "faces_bboxes", None) or []
        source_label = "face"

    for i, bbox in enumerate(bboxes):
        # bbox is a tensor or array-like: [x1, y1, x2, y2]
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        except Exception:
            continue

        gender = gender_label(detected_objects, i)
        age    = age_label(detected_objects, i)
        color  = GENDER_COLORS.get(gender, GENDER_COLORS["unknown"])
        label  = f"{gender}, {age}y"

        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas, label,
            (x1 + 2, y1 - baseline - 2),
            FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA,
        )
        n_persons += 1

    print(f"       → drew {n_persons} {source_label} box(es)")
    return canvas


def add_title_bar(canvas: np.ndarray, title: str, height: int = 36) -> np.ndarray:
    """Prepend a dark title bar above *canvas*."""
    bar = np.zeros((height, canvas.shape[1], 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.putText(bar, title, (8, height - 9), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def build_collage(annotated: list[np.ndarray], titles: list[str]) -> np.ndarray:
    """
    Tile the annotated images side-by-side, each padded to the same height.
    Returns a single BGR image.
    """
    target_h = max(img.shape[0] for img in annotated)

    panels = []
    for img, title in zip(annotated, titles):
        # Pad vertically if necessary
        h, w = img.shape[:2]
        if h < target_h:
            pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        img = add_title_bar(img, title)
        # Separator line
        sep = np.full((img.shape[0], 4, 3), 50, dtype=np.uint8)
        panels.extend([img, sep])

    return np.hstack(panels[:-1])   # drop trailing separator


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    img_paths = load_random_images(args.n, args.seed)
    predictor = build_predictor(args)

    annotated = []
    titles    = []

    for img_path in img_paths:
        print(f"\n[INFO] Processing: {img_path.name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"       [WARN] Could not read image, skipping.")
            continue

        try:
            detected_objects, _ = predictor.recognize(img_bgr)
        except Exception as e:
            print(f"       [WARN] MiVOLO inference failed: {e}")
            detected_objects = None

        if detected_objects is not None:
            canvas = draw_boxes(img_bgr, detected_objects)
        else:
            canvas = img_bgr.copy()
            cv2.putText(
                canvas, "inference failed", (10, 30),
                FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

        annotated.append(canvas)
        # Short title: relative path two levels deep (pair/side/filename)
        parts = img_path.parts
        short = "/".join(parts[-3:])
        titles.append(short)

    if not annotated:
        sys.exit("[ERROR] No images could be processed.")

    collage = build_collage(annotated, titles)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), collage)
    print(f"\n[INFO] Saved collage → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualise MiVOLO gender/age bounding boxes on random images.\n"
            "Weight files default to the same values as evaluate_mivolo_images.py "
            "and are resolved relative to the project root."
        )
    )
    # Weight paths – hidden from the user, defaults match evaluate_mivolo_images.py
    parser.add_argument(
        "--yolo-weights",
        default=DEFAULT_YOLO_WEIGHTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mivolo-weights",
        default=DEFAULT_MIVOLO_WEIGHTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of random images to visualise (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection (default: 42).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output collage path (default: {DEFAULT_OUT}).",
    )
    run(parser.parse_args())
