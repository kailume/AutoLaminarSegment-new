#!/usr/bin/env python
"""
Large image nucleus point detection using StarDist probability maps.

This script follows the same tiling strategy as segment_large_image_fast.py:
the image is split into non-overlapping valid regions, each valid region is
surrounded by an overlap margin, and only points whose centers fall inside the
valid region are kept. Unlike Cellpose instance segmentation, this script does
not reconstruct masks. It runs StarDist only far enough to obtain the object
probability map, then extracts centers with local peak detection.

Outputs
-------
  {stem}_cell_centroids.csv   - cell_id, centroid_x, centroid_y, score
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from csbdeep.utils import normalize
from stardist.models import StarDist2D


# Tunable defaults. CLI arguments override these values.
TILE_SIZE = 4096
OVERLAP = 256
DOWNSAMPLE_RATE = 1.0
MODEL_NAME = "2D_versatile_fluo"
PROB_THRESHOLD = 0.35
PEAK_MIN_DISTANCE = 10
PROB_SMOOTH_SIGMA = 1.0
NORMALIZE_PMIN = 1.0
NORMALIZE_PMAX = 99.8

STRIDE = TILE_SIZE - 2 * OVERLAP


def load_stardist_model(model_name: str) -> StarDist2D:
    """
    Load a StarDist model.

    On Windows, StarDist/CSBDeep may try to create a symlink for downloaded
    pretrained models and fail with WinError 1314 when Developer Mode/admin
    privileges are unavailable. In that case, load the extracted model folder
    directly from the Keras cache.
    """
    try:
        return StarDist2D.from_pretrained(model_name)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise

        cache_root = Path.home() / ".keras" / "models" / "StarDist2D" / model_name
        candidates = [
            cache_root / f"{model_name}_extracted",
            cache_root / model_name / f"{model_name}_extracted",
        ]
        candidates.extend(cache_root.glob("*_extracted"))

        for model_dir in candidates:
            if model_dir.exists() and (model_dir / "config.json").exists():
                print(
                    "StarDist pretrained-model symlink failed on Windows; "
                    f"loading extracted model directly: {model_dir}"
                )
                return StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))

        raise RuntimeError(
            "StarDist failed to create a pretrained-model symlink on Windows, "
            f"and no extracted model folder was found under {cache_root}. "
            "Workarounds: enable Windows Developer Mode, run once as Administrator, "
            "or manually extract/copy the pretrained model folder."
        ) from exc


def load_image(image_path: Path) -> np.ndarray:
    """Load grayscale / multichannel image as uint8, preserving contrast."""
    try:
        img = tifffile.imread(str(image_path))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    if img.ndim == 3:
        img = img[..., 0] if img.shape[2] > 3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(cv2.GaussianBlur(img, (3, 3), 0))

    print(f"Image loaded: {img.shape[1]}w x {img.shape[0]}h px")
    return img


def load_region_mask(image_path: Path, H: int, W: int, mask_path: Path | None = None) -> np.ndarray | None:
    """Load optional binary ROI mask. Auto-detects mask.png/tif next to image."""
    if mask_path is not None:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise ValueError(f"Mask path does not exist: {mask_path}")
    else:
        for ext in (".png", ".tif", ".tiff"):
            p = image_path.with_name(f"mask{ext}")
            if p.exists():
                mask_path = p
                break

    if mask_path is None:
        return None

    orig_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    finally:
        Image.MAX_IMAGE_PIXELS = orig_limit

    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {mask.shape} != image shape {(H, W)}")

    fg = int((mask > 0).sum())
    print(f"Region mask loaded: {mask_path} ({fg} foreground px, {fg / (H * W) * 100:.1f}% of image)")
    return mask > 0


def downsample_image_and_mask(image: np.ndarray, region_mask: np.ndarray | None, rate: float):
    """Resize image/mask for faster inference while restoring output coordinates."""
    if rate <= 0 or rate > 1:
        raise ValueError(f"downsample_rate must be in (0, 1], got {rate}")
    if rate == 1:
        return image, region_mask

    H, W = image.shape
    small_w = max(1, int(round(W * rate)))
    small_h = max(1, int(round(H * rate)))
    image_small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    mask_small = None
    if region_mask is not None:
        mask_small = cv2.resize(
            region_mask.astype(np.uint8),
            (small_w, small_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    print(f"Downsample for inference: {W}x{H} -> {small_w}x{small_h} (rate={rate})")
    return image_small, mask_small


def valid_region_grid(H: int, W: int):
    row_starts = list(range(0, H, STRIDE))
    col_starts = list(range(0, W, STRIDE))
    regions = []
    for i, vr1 in enumerate(row_starts):
        vr2 = row_starts[i + 1] if i + 1 < len(row_starts) else H
        for j, vc1 in enumerate(col_starts):
            vc2 = col_starts[j + 1] if j + 1 < len(col_starts) else W
            regions.append((vr1, vr2, vc1, vc2))
    return regions


def extract_tile(image: np.ndarray, vr1: int, vr2: int, vc1: int, vc2: int):
    H, W = image.shape
    tr1 = max(0, vr1 - OVERLAP)
    tr2 = min(H, vr2 + OVERLAP)
    tc1 = max(0, vc1 - OVERLAP)
    tc2 = min(W, vc2 + OVERLAP)

    patch = image[tr1:tr2, tc1:tc2]
    ph, pw = patch.shape

    pad_b = max(0, TILE_SIZE - ph)
    pad_r = max(0, TILE_SIZE - pw)
    if pad_b or pad_r:
        patch = np.pad(patch, ((0, pad_b), (0, pad_r)), mode="reflect")

    return patch, tr1, tc1, ph, pw


def predict_probability(model: StarDist2D, tile: np.ndarray) -> np.ndarray:
    """Run StarDist and return a probability map aligned to the input tile."""
    tile_norm = normalize(tile, NORMALIZE_PMIN, NORMALIZE_PMAX, axis=(0, 1))
    prob, _dist = model.predict(tile_norm, axes="YX")
    prob = np.asarray(prob, dtype=np.float32)

    if prob.shape != tile.shape:
        print(f"    resizing StarDist prob {prob.shape} -> {tile.shape}")
        prob = cv2.resize(prob, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob


def detect_peaks(prob: np.ndarray) -> np.ndarray:
    """Return peak coordinates as an array of (row, col)."""
    if PROB_SMOOTH_SIGMA and PROB_SMOOTH_SIGMA > 0:
        prob = gaussian_filter(prob, sigma=PROB_SMOOTH_SIGMA)

    peaks = peak_local_max(
        prob,
        min_distance=PEAK_MIN_DISTANCE,
        threshold_abs=PROB_THRESHOLD,
        exclude_border=False,
    )
    if peaks.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    scores = prob[peaks[:, 0], peaks[:, 1]]
    return np.column_stack([peaks[:, 0], peaks[:, 1], scores]).astype(np.float32)


def detect_large_image_points(
    image: np.ndarray,
    region_mask: np.ndarray | None = None,
    downsample_rate: float | None = None,
    model_name: str | None = None,
) -> list[dict]:
    if downsample_rate is None:
        downsample_rate = DOWNSAMPLE_RATE
    if model_name is None:
        model_name = MODEL_NAME

    H, W = image.shape
    regions = valid_region_grid(H, W)
    n_tiles = len(regions)
    coord_scale = 1.0 / downsample_rate

    print(f"Loading StarDist model: {model_name}")
    model = load_stardist_model(model_name)

    print(
        f"Tile size: {TILE_SIZE}px | overlap: {OVERLAP}px | stride: {STRIDE}px | tiles: {n_tiles}"
    )
    print(
        f"prob_threshold={PROB_THRESHOLD} min_distance={PEAK_MIN_DISTANCE} "
        f"sigma={PROB_SMOOTH_SIGMA} normalize=({NORMALIZE_PMIN}, {NORMALIZE_PMAX})\n"
    )

    records = []
    global_id = 0
    n_skipped = 0
    t_start = time.time()

    for idx, (vr1, vr2, vc1, vc2) in enumerate(regions):
        if region_mask is not None:
            vr_mask = region_mask[vr1:vr2, vc1:vc2]
            if not vr_mask.any():
                n_skipped += 1
                print(f"  [{idx + 1:>4}/{n_tiles}] rows {vr1}-{vr2}, cols {vc1}-{vc2} -> skipped (no mask)")
                continue

        tile, tr1, tc1, ph, pw = extract_tile(image, vr1, vr2, vc1, vc2)

        tb = time.time()
        prob = predict_probability(model, tile)
        prob = prob[:ph, :pw]
        peaks = detect_peaks(prob)
        ti = time.time() - tb

        n_kept = 0
        for row, col, score in peaks:
            y_global = float(row) + tr1
            x_global = float(col) + tc1

            if not (vr1 <= y_global < vr2 and vc1 <= x_global < vc2):
                continue
            if region_mask is not None and not region_mask[int(y_global), int(x_global)]:
                continue

            global_id += 1
            records.append(
                {
                    "cell_id": global_id,
                    "centroid_x": round(x_global * coord_scale, 2),
                    "centroid_y": round(y_global * coord_scale, 2),
                    "score": round(float(score), 6),
                }
            )
            n_kept += 1

        print(
            f"  [{idx + 1:>4}/{n_tiles}] rows {vr1}-{vr2}, cols {vc1}-{vc2} "
            f"-> {len(peaks)} peaks, {n_kept} kept ({ti:.2f}s)"
        )

    elapsed = time.time() - t_start
    processed = n_tiles - n_skipped
    print(
        f"\nPoint detection completed in {elapsed:.1f}s "
        f"({elapsed / max(processed, 1):.2f}s/tile, {global_id / max(elapsed, 0.01):.0f} cells/s)"
    )
    if n_skipped:
        print(f"  ({n_skipped}/{n_tiles} tiles skipped outside region mask)")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect nuclei centers from a large image with StarDist prob maps.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same folder as input image)")
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"StarDist pretrained model name (default: {MODEL_NAME})")
    parser.add_argument("--tile-size", type=int, default=None, help=f"Tile size in pixels (default: {TILE_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="Overlap margin in pixels")
    parser.add_argument("--mask-path", default=None, help="Optional ROI mask path. Defaults to mask.png/tif next to image.")
    parser.add_argument(
        "--downsample-rate",
        type=float,
        default=None,
        help=f"Inference downsample rate in (0, 1]. Default: {DOWNSAMPLE_RATE}",
    )
    parser.add_argument("--prob-threshold", type=float, default=PROB_THRESHOLD)
    parser.add_argument("--peak-min-distance", type=int, default=PEAK_MIN_DISTANCE)
    parser.add_argument("--prob-smooth-sigma", type=float, default=PROB_SMOOTH_SIGMA)
    parser.add_argument("--normalize-pmin", type=float, default=NORMALIZE_PMIN)
    parser.add_argument("--normalize-pmax", type=float, default=NORMALIZE_PMAX)
    return parser.parse_args()


def main() -> None:
    global TILE_SIZE, OVERLAP, STRIDE
    global PROB_THRESHOLD, PEAK_MIN_DISTANCE, PROB_SMOOTH_SIGMA
    global NORMALIZE_PMIN, NORMALIZE_PMAX

    args = parse_args()

    TILE_SIZE = args.tile_size if args.tile_size is not None else TILE_SIZE
    OVERLAP = args.overlap
    STRIDE = TILE_SIZE - 2 * OVERLAP
    if STRIDE <= 0:
        raise ValueError(f"overlap ({OVERLAP}) must be less than tile_size/2 ({TILE_SIZE / 2})")

    PROB_THRESHOLD = args.prob_threshold
    PEAK_MIN_DISTANCE = args.peak_min_distance
    PROB_SMOOTH_SIGMA = args.prob_smooth_sigma
    NORMALIZE_PMIN = args.normalize_pmin
    NORMALIZE_PMAX = args.normalize_pmax

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_full = load_image(image_path)
    region_mask_full = load_region_mask(
        image_path,
        *image_full.shape,
        mask_path=Path(args.mask_path) if args.mask_path else None,
    )
    downsample_rate = args.downsample_rate if args.downsample_rate is not None else DOWNSAMPLE_RATE
    image, region_mask = downsample_image_and_mask(image_full, region_mask_full, downsample_rate)

    print("\n--- StarDist Probability-Map Point Detection ---")
    records = detect_large_image_points(
        image,
        region_mask=region_mask,
        downsample_rate=downsample_rate,
        model_name=args.model_name,
    )

    csv_path = output_dir / f"{image_path.stem}_cell_centroids.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nTotal points: {len(records)}")
    print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
