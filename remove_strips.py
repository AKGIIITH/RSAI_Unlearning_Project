"""
zebra_destripe.py
-----------------
Removes black stripes from zebra images and replaces them with a neutral grey.

Usage
-----
python zebra_destripe.py /path/to/zebra_images/
python zebra_destripe.py /path/to/zebra_images/ --output_folder /path/to/output/
python zebra_destripe.py /path/to/zebra_images/ --debug
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# Fixed neutral grey replacement colour (BGR) — tweak if you want warmer/cooler
REPLACE_BGR = np.array([230, 230, 225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: isolate the zebra body
# ─────────────────────────────────────────────────────────────────────────────

def get_zebra_body_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    low_sat_mask = (sat < 60).astype(np.uint8) * 255

    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    lap_smooth = cv2.GaussianBlur(np.abs(lap), (51, 51), 0)
    lap_norm = cv2.normalize(lap_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, high_contrast_mask = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_and(low_sat_mask, high_contrast_mask)

    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_big, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, k_big, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_med, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    body_mask = np.zeros_like(combined)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= (h * w) * 0.01:
            body_mask[labels == lbl] = 255

    body_mask = cv2.dilate(body_mask, k_med, iterations=2)
    return body_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: within the body, find the black stripes
# ─────────────────────────────────────────────────────────────────────────────

def get_black_stripe_mask(img_bgr: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    body_pixels = L[body_mask > 0]
    if len(body_pixels) == 0:
        return np.zeros_like(L)

    otsu_val, _ = cv2.threshold(body_pixels.reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_val = int(np.clip(int(otsu_val * 0.85), 40, 120))

    _, black_mask = cv2.threshold(L, threshold_val, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.bitwise_and(black_mask, body_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, k, iterations=2)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,  k, iterations=1)
    return black_mask


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def remove_black_stripes(image_path: str, output_path: str, debug: bool = False) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[SKIP] Could not read: {image_path}")
        return

    stem = Path(image_path).stem

    body_mask  = get_zebra_body_mask(img)
    black_mask = get_black_stripe_mask(img, body_mask)

    # Feathered blend — replace black stripes with fixed neutral grey
    feather_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    black_mask_dilated = cv2.dilate(black_mask, feather_k, iterations=1)
    soft_mask = cv2.GaussianBlur(black_mask_dilated.astype(np.float32), (15, 15), 0) / 255.0
    soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

    fill = np.ones_like(img, dtype=np.float32) * REPLACE_BGR
    result = (fill * soft_mask_3ch +
              img.astype(np.float32) * (1.0 - soft_mask_3ch)).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"[DONE] {Path(image_path).name}")

    if debug:
        dbg_dir = Path(output_path).parent / "debug"
        dbg_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(dbg_dir / f"{stem}_body_mask.png"),  body_mask)
        cv2.imwrite(str(dbg_dir / f"{stem}_black_mask.png"), black_mask)


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(input_folder: str, output_folder: str, debug: bool = False) -> None:
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)

    if not files:
        print(f"No supported images found in: {input_folder}")
        return

    print(f"Found {len(files)} image(s). Processing...\n")
    for f in files:
        remove_black_stripes(str(f), str(output_path / f.name), debug=debug)
    print(f"\nAll done!  Results saved to: {output_folder}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove black stripes from zebra images, replace with neutral grey."
    )
    parser.add_argument("input_folder", type=str)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    out = args.output_folder or os.path.join(args.input_folder, "output")
    process_folder(args.input_folder, out, debug=args.debug)