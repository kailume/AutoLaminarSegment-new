#!/usr/bin/env python
"""
Large image cell segmentation using Cellpose with configurable tiling.

Strategy
--------
The image is partitioned into non-overlapping "valid regions" of size STRIDE x STRIDE.
Each valid region is surrounded by an OVERLAP-pixel margin to form a TILE_SIZE × TILE_SIZE
tile that is fed to Cellpose.  After segmentation, only cells whose centroid falls inside the
valid region are kept — this guarantees every cell is counted exactly once even when
its mask straddles a tile boundary.

Outputs
-------
  {stem}_full_mask.tif        — uint32 label image, same size as input
  {stem}_cell_centroids.csv   — cell_id, centroid_x, centroid_y, area_px, area_um2
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from skimage.measure import regionprops
from PIL import Image

# ── tuneable constants ──────────────────────────────────────────────────────────
# TILE_SIZE          = 2048        # larger → more internal 256×256 patches → better GPU util
TILE_SIZE          = 4096        # larger → more internal 256×256 patches → better GPU util
OVERLAP            = 256          # pixels of context on each side
DOWNSAMPLE_RATE    = 1.0         # inference-only resize; output centroids are restored to full-res coordinates
CELLPROB_THRESHOLD = 0
FLOW_THRESHOLD     = 0.4
# DIAMETER           = 30          # pixels
DIAMETER           = 40          # pixels
PIXEL_SIZE_UM      = 0.1625        # µm per pixel
BATCH_SIZE         = 64          # Cellpose internal patch batch size (num 256×256 patches per GPU pass)
USE_TTA            = False        # test-time augmentation (multi-scale resampling)
# ───────────────────────────────────────────────────────────────────────────────

STRIDE = TILE_SIZE - 2 * OVERLAP   # non-overlapping step


# ── image loading ───────────────────────────────────────────────────────────────

def load_image(image_path: Path) -> np.ndarray:
    """Load any grayscale / multichannel image as uint8, preserving contrast."""
    # tifffile handles 16-bit TIFFs correctly
    try:
        img = tifffile.imread(str(image_path))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    # Collapse to single channel
    if img.ndim == 3:
        img = img[..., 0] if img.shape[2] > 3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stretch to full uint8 range
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    # Global CLAHE for uniform contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(cv2.GaussianBlur(img, (3, 3), 0))

    print(f"Image loaded: {img.shape[1]}w × {img.shape[0]}h px")
    return img


def load_region_mask(image_path: Path, H: int, W: int, mask_path: Path | None = None) -> np.ndarray | None:
    """
    Auto-detect ``mask.png`` / ``mask.tif`` / ``mask.tiff`` next to the input
    image and load it as a binary region-of-interest mask.

    Returns ``None`` when no mask file exists — caller skips mask logic entirely.
    Raises ``ValueError`` if a mask file is found but cannot be read or has wrong shape.
    """
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

    # Bypass PIL's decompression-bomb check for large whole-slide masks
    orig_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    finally:
        Image.MAX_IMAGE_PIXELS = orig_limit

    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {mask.shape} != image shape {(H, W)}")

    fg = int((mask > 0).sum())
    print(f"Region mask loaded: {mask_path}  ({fg} foreground px, "
          f"{fg / (H * W) * 100:.1f}% of image)")
    return mask > 0


def downsample_image_and_mask(image: np.ndarray, region_mask: np.ndarray | None, rate: float):
    """Resize image/mask for faster inference while keeping original coordinates for outputs."""
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


def restore_label_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Upsample label mask back to original image shape with nearest-neighbor interpolation."""
    out_h, out_w = output_shape
    if mask.shape == (out_h, out_w):
        return mask
    restored = cv2.resize(mask.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return restored.astype(np.uint32)


# ── tiling helpers ──────────────────────────────────────────────────────────────

def valid_region_grid(H: int, W: int):
    """
    Partition (H, W) into non-overlapping valid regions of size ~STRIDE.
    Returns list of (vr1, vr2, vc1, vc2).
    """
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
    """
    Extract a tile with OVERLAP margin.  Returns:
        tile        — uint8 array, padded to TILE_SIZE × TILE_SIZE with reflect
        tr1, tc1    — top-left corner of the tile in global image coordinates
        ph, pw      — unpadded height and width
    """
    H, W = image.shape
    tr1 = max(0, vr1 - OVERLAP)
    tr2 = min(H, vr2 + OVERLAP)
    tc1 = max(0, vc1 - OVERLAP)
    tc2 = min(W, vc2 + OVERLAP)

    patch = image[tr1:tr2, tc1:tc2]
    ph, pw = patch.shape

    # Pad to TILE_SIZE with reflect if smaller than tile
    pad_b = max(0, TILE_SIZE - ph)
    pad_r = max(0, TILE_SIZE - pw)
    if pad_b or pad_r:
        patch = np.pad(patch, ((0, pad_b), (0, pad_r)), mode="reflect")

    return patch, tr1, tc1, ph, pw   # ph/pw = unpadded dimensions


# ── main segmentation routine ────────────────────────────────────────────────────

def segment_large_image(image: np.ndarray, use_gpu: bool = True,
                        batch_size: int = None, use_tta: bool = None,
                        region_mask: np.ndarray = None,
                        downsample_rate: float = None):
    """
    Tile the image, run Cellpose on each tile, stitch into a global label mask.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (H x W).
    use_gpu : bool
        Whether to use GPU for inference.
    batch_size : int, optional
        Cellpose internal patch batch size — how many 256×256 patches are
        processed per GPU forward pass.  Larger = better GPU utilization.
        Defaults to BATCH_SIZE module constant.
    use_tta : bool, optional
        Enable test-time augmentation (multi-scale resampling).
        Defaults to USE_TTA.
    region_mask : np.ndarray, optional
        Binary mask (H x W, bool).  When provided, only tiles whose valid
        region overlaps the mask are processed, and only cells whose centroid
        falls inside the mask are kept.

    Returns
    -------
    global_mask   : np.ndarray, uint32, shape == image.shape
    cell_records  : list[dict]  — one entry per cell
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if use_tta is None:
        use_tta = USE_TTA
    if downsample_rate is None:
        downsample_rate = DOWNSAMPLE_RATE

    H, W = image.shape
    global_mask = np.zeros((H, W), dtype=np.uint32)

    model = models.CellposeModel(gpu=use_gpu, model_type="nuclei")
    regions = valid_region_grid(H, W)
    n_tiles = len(regions)

    global_id = 0
    cell_records = []

    print(f"Tile size: {TILE_SIZE}px  |  overlap: {OVERLAP}px  |  "
          f"stride: {STRIDE}px  |  tiles: {n_tiles}")
    effective_diameter = max(1, int(round(DIAMETER * downsample_rate)))
    coord_scale = 1.0 / downsample_rate
    area_scale = coord_scale ** 2

    print(f"cellprob_threshold={CELLPROB_THRESHOLD}  flow_threshold={FLOW_THRESHOLD}  "
          f"diameter={effective_diameter}px (base={DIAMETER}px, rate={downsample_rate})")
    print(f"batch_size={batch_size}  TTA={use_tta}\n")

    t_start = time.time()

    n_skipped = 0
    for idx, (vr1, vr2, vc1, vc2) in enumerate(regions):
        # Skip tiles with no overlap with the region mask
        if region_mask is not None:
            vr_mask = region_mask[vr1:vr2, vc1:vc2]
            if not vr_mask.any():
                n_skipped += 1
                print(f"  [{idx+1:>4}/{n_tiles}] rows {vr1}-{vr2}, cols {vc1}-{vc2} "
                      f"→ skipped (no mask)")
                continue

        tile, tr1, tc1, ph, pw = extract_tile(image, vr1, vr2, vc1, vc2)

        tb = time.time()
        masks_out, _, _ = model.eval(
            [tile],
            diameter=effective_diameter,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            resample=use_tta,
            batch_size=batch_size,
        )
        ti = time.time() - tb

        tile_mask = masks_out[0]
        # Cellpose may return masks in its internally rescaled coordinate system
        # when diameter differs from the model default. Restore to the tile's
        # input size before converting tile-local coordinates back to global.
        if tile_mask.shape != tile.shape:
            print(
                f"    resizing Cellpose mask {tile_mask.shape} -> {tile.shape} "
                f"(diameter={effective_diameter})"
            )
            tile_mask = cv2.resize(
                tile_mask.astype(np.float32),
                (tile.shape[1], tile.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint32)
        tile_mask = tile_mask[:ph, :pw]   # strip padding

        # Use regionprops for vectorized per-cell analysis (avoids repeated np.where)
        n_kept = 0
        n_detected = 0
        for region in regionprops(tile_mask):
            n_detected += 1
            cy_tile, cx_tile = region.centroid
            cy_global = cy_tile + tr1
            cx_global = cx_tile + tc1

            # Centroid must fall inside this tile's exclusive valid region
            if not (vr1 <= cy_global < vr2 and vc1 <= cx_global < vc2):
                continue

            # Centroid must fall inside the region mask (if provided)
            if region_mask is not None and not region_mask[int(cy_global), int(cx_global)]:
                continue

            # Pixel coordinates from regionprops (avoids np.where per cell)
            coords = region.coords   # (N, 2) array of (row, col)
            gy = coords[:, 0].astype(np.int64) + tr1
            gx = coords[:, 1].astype(np.int64) + tc1

            # Clip to image bounds (safety for edge tiles)
            in_bounds = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
            gy, gx = gy[in_bounds], gx[in_bounds]

            # Write to global mask — skip pixels already claimed (overlap artefacts)
            free = global_mask[gy, gx] == 0
            if not free.any():
                continue

            global_id += 1
            global_mask[gy[free], gx[free]] = global_id

            area_px  = int(round(float(free.sum()) * area_scale))
            area_um2 = round(area_px * PIXEL_SIZE_UM ** 2, 4)
            cell_records.append({
                "cell_id":    global_id,
                "centroid_x": round(float(cx_global) * coord_scale, 2),
                "centroid_y": round(float(cy_global) * coord_scale, 2),
                "area_px":    area_px,
                "area_um2":   area_um2,
            })
            n_kept += 1

        print(f"  [{idx+1:>4}/{n_tiles}] rows {vr1}-{vr2}, cols {vc1}-{vc2} "
              f"→ {n_detected} detected, {n_kept} kept  ({ti:.2f}s)")

    elapsed = time.time() - t_start
    processed = n_tiles - n_skipped
    print(f"\nSegmentation completed in {elapsed:.1f}s "
          f"({elapsed / max(processed, 1):.2f}s/tile, "
          f"{global_id / max(elapsed, 0.01):.0f} cells/s)")
    if n_skipped:
        print(f"  ({n_skipped}/{n_tiles} tiles skipped — outside region mask)")

    return global_mask, cell_records


def save_random_color_mask(mask: np.ndarray, path: Path) -> None:
    """
    Render label mask as a random-colour RGB image.
    Each cell gets a distinct random colour; background is black.
    Uses a fixed seed so the output is reproducible.
    """
    rng = np.random.default_rng(seed=42)
    n_labels = int(mask.max())

    # Build colour lookup table: index 0 = black (background)
    lut = np.zeros((n_labels + 1, 3), dtype=np.uint8)
    if n_labels > 0:
        lut[1:] = rng.integers(60, 256, size=(n_labels, 3), dtype=np.uint8)

    rgb = lut[mask]   # H × W × 3
    Image.fromarray(rgb).save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path,
                 alpha: float = 0.45) -> None:
    """
    Blend the grayscale image with a random-colour mask overlay.
    alpha controls mask opacity (0 = invisible, 1 = fully opaque).
    """
    rng = np.random.default_rng(seed=42)
    n_labels = int(mask.max())

    lut = np.zeros((n_labels + 1, 3), dtype=np.uint8)
    if n_labels > 0:
        lut[1:] = rng.integers(60, 256, size=(n_labels, 3), dtype=np.uint8)

    # Grayscale → RGB
    base = np.stack([image] * 3, axis=-1).astype(np.float32)

    # Colour mask
    colour = lut[mask].astype(np.float32)

    # Blend only where mask > 0
    cell_px = mask > 0
    blended = base.copy()
    blended[cell_px] = (1 - alpha) * base[cell_px] + alpha * colour[cell_px]

    Image.fromarray(blended.astype(np.uint8)).save(path)


def main():
    global TILE_SIZE, OVERLAP, STRIDE   # must precede any reference to these names
    parser = argparse.ArgumentParser(
        description="Segment a large image with Cellpose tiling."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: same folder as input image)"
    )
    parser.add_argument("--no_gpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Cellpose internal batch size (256×256 patches per GPU pass, "
                             f"default: file BATCH_SIZE={BATCH_SIZE})")
    parser.add_argument("--tile-size", type=int, default=None,
                        help=f"Tile size in pixels (default: file TILE_SIZE={TILE_SIZE}). "
                             "Larger tiles → more internal patches → better GPU utilization.")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help="Overlap margin in pixels (default: %(default)s). "
                             "Must be < tile_size/2.")
    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument("--no-tta", dest="use_tta", action="store_false", default=None,
                           help=f"Disable test-time augmentation (default: file USE_TTA={USE_TTA})")
    tta_group.add_argument("--tta", dest="use_tta", action="store_true",
                           help="Enable test-time augmentation")
    parser.add_argument("--mask-path", default=None,
                        help="Optional explicit region mask path. Defaults to mask.png/tif next to image.")
    parser.add_argument("--downsample-rate", type=float, default=None,
                        help="Inference downsample rate in (0, 1]. "
                             "1 = full resolution, 0.5 = half width/height, 0.25 = quarter width/height. "
                             f"Default: file DOWNSAMPLE_RATE={DOWNSAMPLE_RATE}.")
    args = parser.parse_args()

    # Update tiling parameters so all helper functions pick up the new values
    TILE_SIZE = args.tile_size if args.tile_size is not None else TILE_SIZE
    OVERLAP   = args.overlap
    STRIDE    = TILE_SIZE - 2 * OVERLAP
    if STRIDE <= 0:
        parser.error(f"overlap ({OVERLAP}) must be less than tile_size/2 ({TILE_SIZE / 2})")

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_full = load_image(image_path)

    region_mask_full = load_region_mask(
        image_path, *image_full.shape, mask_path=Path(args.mask_path) if args.mask_path else None
    )
    downsample_rate = args.downsample_rate if args.downsample_rate is not None else DOWNSAMPLE_RATE
    image, region_mask = downsample_image_and_mask(image_full, region_mask_full, downsample_rate)

    print("\n── Segmentation ─────────────────────────────────────────────────────")
    global_mask, cell_records = segment_large_image(
        image,
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        region_mask=region_mask,
        downsample_rate=downsample_rate,
    )

    n_cells = len(cell_records)
    print(f"\nTotal cells: {n_cells}")

    df = pd.DataFrame(cell_records)
    csv_path = output_dir / f"{image_path.stem}_cell_centroids.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV   saved → {csv_path}")

    # Save full mask as compressed uint16 TIFF (lossless, ~90% smaller than uncompressed uint32)
    full_res_mask = restore_label_mask(global_mask, image_full.shape)
    mask_path = output_dir / f"{image_path.stem}_full_mask.tif"
    tifffile.imwrite(str(mask_path), full_res_mask.astype(np.uint32), compression="zlib")
    print(f"Mask  saved → {mask_path}")

    # Save random-colour pseudocolor mask (8-bit PNG)
    color_path = output_dir / f"{image_path.stem}_mask_color.png"
    save_random_color_mask(full_res_mask, color_path)
    print(f"Color saved → {color_path}")

    # Save grayscale + mask overlay (8-bit PNG)
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    save_overlay(image_full, full_res_mask, overlay_path)
    print(f"Overlay saved → {overlay_path}")


if __name__ == "__main__":
    main()
