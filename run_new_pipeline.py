#!/usr/bin/env python
"""
New laminar segmentation pipeline — operates directly on preprocessed data.

No 4x images, no scan metadata, no cell segmentation visualisations.

Inputs (per sample under <input_root>/<sample>/):
  - dapi.png / dapi.tif      : DAPI / 40x image for cell segmentation
  - pia.csv                   : gray matter outer boundary (x,y)  — alias for GM
  - white.csv                 : white matter boundary (x,y)       — alias for WM
  - graymask.png              : binary gray matter mask (0 / 255)

Outputs (per sample under <output_root>/<sample>/):
  - cell_centroid.csv         : cell centroid point cloud
  - segmented_depth.csv       : layer depth boundaries (renamed from segmented_layers.csv)
  - all_boundaries.csv        : all boundary coordinates in one table (column "boundary" tags each row)
  - pia_boundary.csv          : pia boundary points (copied from input)
  - white_boundary.csv        : white boundary points (copied from input)
  - boundary_L1_2.csv         : L1/L2 contour coordinates
  - boundary_L3_4.csv         : L3/L4 contour coordinates
  - boundary_L4_5.csv         : L4/L5 contour coordinates
  - layers_color_mask.png     : multi-colour per-pixel layer mask
  - layer_lines.png           : all boundary lines drawn in white thick dashes
  - depth_density_layers_peak_based.png  : algorithm diagnostic plot

Usage:
  python run_new_pipeline.py
  python run_new_pipeline.py --samples 1 2 3
  python run_new_pipeline.py --input-root dataset/inputs/preprocessed --output-root dataset/outputs
  python run_new_pipeline.py --no-gpu --separate-l23
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import src.analyseDensity as density
import src.layerVisualize as layer_visual

# ── defaults ───────────────────────────────────────────────────────────────────────
AUTO_TILE_SIZE = 4096
AUTO_BATCH_SIZE = 64
AUTO_DOWNSAMPLE_RATE = 0.8
AUTO_NO_TTA = True
AUTO_DEPTH_METHOD = "legacy"
AUTO_COMPACT_RATE = 0.1
AUTO_BOUNDARY_DOWNSAMPLE_RATE = 0.2
MERGE_LAYER23 = True
KDE_BANDWIDTH = "scott"
PIXEL_SIZE_UM = 0.1625

IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)

# ── helpers ─────────────────────────────────────────────────────────────────────────


def _resolve_image_path(directory: Path, stem: str) -> Path:
    """Resolve *stem* in *directory* trying common image extensions."""
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    # fallback — first match with any extension
    for p in sorted(directory.iterdir()):
        if p.stem.lower() == stem.lower() and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return directory / f"{stem}.png"  # nominal; will fail downstream


def _load_mask(mask_path: Path) -> np.ndarray | None:
    """Load a binary mask (0/255 uint8) via PIL (bypasses OpenCV large-image limit)."""
    try:
        from PIL import Image as _PIL
        import PIL
        saved = PIL.Image.MAX_IMAGE_PIXELS
        PIL.Image.MAX_IMAGE_PIXELS = None
        m = np.array(_PIL.open(str(mask_path)).convert("L"))
        PIL.Image.MAX_IMAGE_PIXELS = saved
        return (m > 0).astype(np.uint8) * 255
    except Exception:
        return None


def _filter_cells_by_mask(cells: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Keep only cells whose (X, Y) centroid falls inside *mask*."""
    coords = cells[["X", "Y"]].to_numpy(dtype=int)
    h, w = mask.shape[:2]
    in_bounds = (
        (coords[:, 1] >= 0) & (coords[:, 1] < h) &
        (coords[:, 0] >= 0) & (coords[:, 0] < w)
    )
    keep = np.zeros(len(cells), dtype=bool)
    keep[in_bounds] = mask[coords[in_bounds, 1], coords[in_bounds, 0]] > 0
    return cells.loc[keep].reset_index(drop=True)


def _load_cells(csv_path: Path) -> pd.DataFrame:
    """Load centroid CSV and normalise coordinate columns to X / Y."""
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for src_name in ("centroid_x", "x", "X"):
        if src_name in cols_lower and "X" not in df.columns:
            rename_map[cols_lower[src_name]] = "X"
            break
    for src_name in ("centroid_y", "y", "Y"):
        if src_name in cols_lower and "Y" not in df.columns:
            rename_map[cols_lower[src_name]] = "Y"
            break
    if rename_map:
        df = df.rename(columns=rename_map)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"Cannot find centroid columns in {csv_path}: {list(df.columns)}")
    return df[["X", "Y"]].copy()


def _load_boundary_csv(csv_path: Path) -> pd.DataFrame:
    """Load boundary CSV (x,y columns) and return as DataFrame."""
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x")
    y_col = cols_lower.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"Boundary CSV must have x/y columns, got {list(df.columns)}")
    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    return out


def _clip_boundary_to_image(pts: np.ndarray, h: int, w: int, name: str) -> np.ndarray:
    """Remove boundary points outside the image canvas."""
    pts = np.asarray(pts, dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    in_bounds = finite & (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    removed = int(len(pts) - int(in_bounds.sum()))
    if removed:
        print(f"    [{name}] clipped {removed}/{len(pts)} points outside image")
    clipped = np.rint(pts[in_bounds]).astype(np.int32)
    if len(clipped) == 0:
        raise ValueError(f"{name} has no points inside image ({w}x{h})")
    return clipped


def _choose_param(cli_value, auto_value):
    """CLI overrides auto default; None delegates to downstream module default."""
    return cli_value if cli_value is not None else auto_value


def _resolve_cellpose_python(value: str | None) -> Path | None:
    """Return a separate cellpose Python executable when current env lacks cellpose."""
    if value:
        return Path(value)
    if importlib.util.find_spec("cellpose") is not None:
        return None
    candidates = [
        Path("venv-cellpose") / "Scripts" / "python.exe",
        Path(".venv-cellpose") / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        "Cellpose not installed and no venv-cellpose found. "
        "Use --cellpose-python <path>."
    )


# ── cell segmentation (no extra visualisations) ────────────────────────────────────


def segment_cells(
    dapi_path: Path,
    graymask_path: Path,
    output_dir: Path,
    use_gpu: bool,
    batch_size: int | None,
    tile_size: int | None,
    overlap: int,
    use_tta: bool,
    downsample_rate: float | None,
    cellpose_python: Path | None = None,
) -> Path:
    """Segment nuclei on DAPI using tiled Cellpose.

    Returns path to ``cell_centroid.csv``.
    Does **not** save full-res label mask, colour overlay, or any cell
    segmentation visualisation — only the centroid CSV and (if in-process) a
    lossless compressed label TIFF for possible later re-use.
    """
    print(f"\n  ── Cell segmentation ──")
    print(f"  DAPI: {dapi_path.name}")

    if cellpose_python is not None:
        # subprocess path — suppresses visualisations via CLI flags?
        # The CLI always saves them; we can just ignore / delete them.
        cmd = [
            str(cellpose_python),
            str(Path(__file__).parent / "segment_large_image_fast.py"),
            str(dapi_path),
            "--output_dir", str(output_dir),
            "--overlap", str(overlap),
            "--mask-path", str(graymask_path),
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if tile_size is not None:
            cmd.extend(["--tile-size", str(tile_size)])
        if downsample_rate is not None:
            cmd.extend(["--downsample-rate", str(downsample_rate)])
        if not use_gpu:
            cmd.append("--no_gpu")
        if use_tta:
            cmd.append("--tta")
        else:
            cmd.append("--no-tta")

        print(f"  Running Cellpose (subprocess) ...")
        subprocess.run(cmd, check=True)

        stem_csv = output_dir / f"{dapi_path.stem}_cell_centroids.csv"
        cell_csv = output_dir / "cell_centroids.csv"
        if not stem_csv.exists():
            raise FileNotFoundError(f"Expected centroid CSV not found: {stem_csv}")
        pd.read_csv(stem_csv).to_csv(cell_csv, index=False)
        print(f"  -> {cell_csv}")

        # clean up cell-segmentation visualisations
        for pattern in ("*_mask_color.png", "*_overlay.png", "combined_visualization.png",
                        "layers_overlay.png"):
            for f in output_dir.glob(pattern):
                f.unlink(missing_ok=True)
        return cell_csv

    # ── in-process path ──
    import tifffile
    import segment_large_image_fast as cellseg

    if tile_size is not None:
        cellseg.TILE_SIZE = tile_size
    tile_size = cellseg.TILE_SIZE
    cellseg.OVERLAP = overlap
    cellseg.STRIDE = tile_size - 2 * overlap
    if cellseg.STRIDE <= 0:
        raise ValueError(f"overlap ({overlap}) must be < tile_size/2 ({tile_size / 2})")

    # load image
    image_full = cellseg.load_image(dapi_path)
    H, W = image_full.shape

    # load region mask
    region_mask_full = cellseg.load_region_mask(
        dapi_path, H, W, mask_path=graymask_path
    )

    if downsample_rate is None:
        downsample_rate = cellseg.DOWNSAMPLE_RATE
    image, region_mask = cellseg.downsample_image_and_mask(
        image_full, region_mask_full, downsample_rate
    )

    global_mask, cell_records = cellseg.segment_large_image(
        image,
        use_gpu=use_gpu,
        batch_size=batch_size,
        use_tta=use_tta,
        region_mask=region_mask,
        downsample_rate=downsample_rate,
    )

    # save centroid CSV (the only output we keep)
    df = pd.DataFrame(cell_records)
    cell_csv = output_dir / "cell_centroids.csv"
    df.to_csv(cell_csv, index=False)
    print(f"  -> {cell_csv}  ({len(df)} cells)")

    # save full-res label mask as compressed TIFF (internal; not a visualisation)
    full_res_mask = cellseg.restore_label_mask(global_mask, image_full.shape)
    tifffile.imwrite(
        str(output_dir / f"{dapi_path.stem}_full_mask.tif"),
        full_res_mask.astype(np.uint32),
        compression="zlib",
    )

    return cell_csv


# ── layer boundary extraction from depth map ───────────────────────────────────────


def _compute_depth_map(h: int, w: int, pia_pts: np.ndarray, white_pts: np.ndarray) -> np.ndarray:
    """Per-pixel normalised depth via Euclidean distance transform.
    pia (GM) -> 0, white (WM) -> 1.
    """
    gm_mask = np.zeros((h, w), dtype=np.uint8)
    wm_mask = np.zeros((h, w), dtype=np.uint8)

    for pt in pia_pts:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < w and 0 <= py < h:
            gm_mask[py, px] = 255
    for pt in white_pts:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < w and 0 <= py < h:
            wm_mask[py, px] = 255

    dist_gm = cv2.distanceTransform(255 - gm_mask, cv2.DIST_L2, 5)
    dist_wm = cv2.distanceTransform(255 - wm_mask, cv2.DIST_L2, 5)
    total = dist_gm + dist_wm + 1e-8
    return dist_gm / total


def _extract_contour_points(depth_map: np.ndarray, threshold: float) -> np.ndarray | None:
    """Extract the longest contour at *threshold* from the depth map.

    Returns Nx2 array of (x, y) pixel coordinates, or None if no contour found.
    """
    binary = (depth_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # pick the longest contour
    longest = max(contours, key=lambda c: cv2.arcLength(c, False))
    pts = longest.squeeze()  # (N, 1, 2) -> (N, 2)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    return pts.astype(np.float32)


def _save_boundary_csv(pts: np.ndarray | None, path: Path, name: str):
    """Save (x, y) points as CSV.  Saves empty file with header-only if *pts* is None."""
    if pts is not None and len(pts) > 0:
        df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
        df.to_csv(path, index=False)
        print(f"    -> {name}: {path.name}  ({len(df)} points)")
    else:
        pd.DataFrame({"x": [], "y": []}).to_csv(path, index=False)
        print(f"    -> {name}: {path.name}  (empty: contour not found)")


def save_layer_boundaries(
    pia_csv: Path,
    white_csv: Path,
    layers_csv: Path,
    dapi_path: Path,
    graymask_path: Path,
    output_dir: Path,
    compact_rate: float | None = None,
):
    """Compute depth map and save all boundary coordinate CSVs + layer_lines.png.

    Saves:
      - pia_boundary.csv         (copy of input pia.csv)
      - white_boundary.csv       (copy of input white.csv)
      - boundary_L1_2.csv        (depth contour at first internal boundary)
      - boundary_L3_4.csv        (depth contour at second internal boundary)
      - boundary_L4_5.csv        (depth contour at third internal boundary)
      - all_boundaries.csv       (aggregate of all 5 boundaries with a "boundary" tag column)
      - layer_lines.png          (all 5 boundaries as white thick dashed lines)
      - layers_color_mask.png    (multi-colour per-pixel layer mask)
    """
    print(f"\n  ── Layer boundary extraction ──")

    # 1. load image for canvas size
    orig = layer_visual.load_visualization_image(str(dapi_path))
    if orig is None:
        raise ValueError(f"Cannot load DAPI image for visualisation: {dapi_path}")
    if orig.ndim == 2:
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    elif orig.shape[2] == 4:
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)
    else:
        orig_bgr = orig.copy()
    h, w = orig_bgr.shape[:2]

    # load graymask if available
    mask = None
    if graymask_path.exists():
        mask = _load_mask(graymask_path)
        if mask is not None and mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. load boundary points
    pia_df = _load_boundary_csv(pia_csv)
    white_df = _load_boundary_csv(white_csv)
    pia_pts = _clip_boundary_to_image(pia_df[["x", "y"]].to_numpy(), h, w, "pia")
    white_pts = _clip_boundary_to_image(white_df[["x", "y"]].to_numpy(), h, w, "white")

    # 3. compact scaling if requested
    _compact_rate = AUTO_COMPACT_RATE if compact_rate is None else compact_rate
    use_compact = _compact_rate is not None and 0 < _compact_rate < 1
    if use_compact:
        scale = float(_compact_rate)
        small_w = max(int(round(w * scale)), 1)
        small_h = max(int(round(h * scale)), 1)
        orig_bgr = cv2.resize(orig_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        h, w = small_h, small_w
        pia_pts = np.rint(pia_pts * scale).astype(np.int32)
        white_pts = np.rint(white_pts * scale).astype(np.int32)
        pia_pts[:, 0] = np.clip(pia_pts[:, 0], 0, w - 1)
        pia_pts[:, 1] = np.clip(pia_pts[:, 1], 0, h - 1)
        white_pts[:, 0] = np.clip(white_pts[:, 0], 0, w - 1)
        white_pts[:, 1] = np.clip(white_pts[:, 1], 0, h - 1)
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        print(f"  Compact mode: rate={_compact_rate}, resized to {w}x{h}")

    # 4. depth map
    print("  Computing depth map ...")
    depth_map = _compute_depth_map(h, w, pia_pts, white_pts)

    # 5. load layer definitions
    layers_df = pd.read_csv(layers_csv)
    layers = layers_df.to_dict("records")

    # 6. map internal boundary depths by layer name
    #    Layers are ordered GM->WM (depth 0->1).  The internal boundary after a
    #    given layer is that layer's "end" value.  We name them according to
    #    the conventions the user asked for: L1/2, L3/4, L4/5.
    boundary_by_name: dict[str, float] = {}
    for i, layer in enumerate(layers):
        lname = str(layer["layer"]).strip()
        end = float(layer["end"])
        if i < len(layers) - 1 and 0.0 < end < 1.0:
            boundary_by_name[lname] = end

    # resolve the three requested boundaries irrespective of merge setting
    boundary_map = {}
    # L1/2 = end of first layer (always named "1")
    if "1" in boundary_by_name:
        boundary_map["L1_2"] = boundary_by_name["1"]
    # L3/4 = end of "2/3" (merged) or "3" (separate)
    for key in ("2/3", "3"):
        if key in boundary_by_name:
            boundary_map["L3_4"] = boundary_by_name[key]
            break
    # L4/5 = end of "4"
    if "4" in boundary_by_name:
        boundary_map["L4_5"] = boundary_by_name["4"]

    internal_names = ["L1_2", "L3_4", "L4_5"]
    internal_depths = [boundary_map.get(n) for n in internal_names]
    print(f"  Internal boundaries: "
          f"{', '.join(f'{n}={d:.4f}' for n, d in zip(internal_names, internal_depths) if d is not None)}")

    # 7. aggregate all boundaries into one CSV with a "boundary" column
    # (individual boundary CSVs temporarily disabled — all_boundaries.csv has them all)
    # shutil.copy2(pia_csv, output_dir / "pia_boundary.csv")
    # shutil.copy2(white_csv, output_dir / "white_boundary.csv")
    # for name, depth in zip(internal_names, internal_depths):
    #     if depth is None:
    #         _save_boundary_csv(None, output_dir / f"boundary_{name}.csv", name)
    #     else:
    #         pts = _extract_contour_points(depth_map, depth)
    #         _save_boundary_csv(pts, output_dir / f"boundary_{name}.csv", name)

    # build all_boundaries.csv from in-memory data
    all_dfs = []
    pia_tagged = pia_df[["x", "y"]].copy()
    pia_tagged["boundary"] = "pia"
    all_dfs.append(pia_tagged)
    white_tagged = white_df[["x", "y"]].copy()
    white_tagged["boundary"] = "white"
    all_dfs.append(white_tagged)
    for name, depth in zip(internal_names, internal_depths):
        if depth is not None:
            pts = _extract_contour_points(depth_map, depth)
            if pts is not None and len(pts) > 0:
                dfb = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "boundary": name})
                all_dfs.append(dfb)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / "all_boundaries.csv", index=False)
    print(f"    -> all_boundaries.csv  ({len(combined)} points across {len(all_dfs)} boundaries)")

    # collect all boundary depths for drawing (internal; skip 0/1)
    all_depths_for_drawing = [d for d in internal_depths if d is not None]

    # generate layer_lines.png — all boundaries in white thick dashed lines
    print("  Rendering layer_lines.png ...")
    line_thickness = 10
    dash_len = 20
    gap_len = 15

    # Internal layer boundaries: draw with mask filtering to suppress
    # findContours edge artefacts.
    internal_mask = np.ones((h, w), dtype=bool)
    margin = max(line_thickness + 2, 1)
    internal_mask[:margin, :] = False
    internal_mask[-margin:, :] = False
    internal_mask[:, :margin] = False
    internal_mask[:, -margin:] = False
    if mask is not None:
        ksize = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        internal_mask &= cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1) > 0

    line_canvas = np.zeros_like(orig_bgr)
    white_color = (255, 255, 255)

    for depth in all_depths_for_drawing:
        binary = (depth_map >= depth).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 50:
                continue
            _draw_dashed_contour(line_canvas, cnt, white_color, line_thickness, dash_len, gap_len, valid_mask=internal_mask)

    layer_lines_img = orig_bgr.copy()
    if mask is not None:
        line_mask = (line_canvas > 0) & (np.tile(mask > 0, (3, 1, 1)).transpose(1, 2, 0))
    else:
        line_mask = line_canvas > 0
    layer_lines_img[line_mask] = line_canvas[line_mask]

    # pia + white: draw directly on top, no mask filtering (they sit at the
    # graymask edge and would be clipped otherwise).
    _draw_dashed_points(layer_lines_img, pia_pts, white_color, line_thickness, dash_len, gap_len)
    _draw_dashed_points(layer_lines_img, white_pts, white_color, line_thickness, dash_len, gap_len)
    cv2.imwrite(str(output_dir / "layer_lines.png"), layer_lines_img)
    n_internal = len([d for d in all_depths_for_drawing if d is not None])
    print(f"    -> layer_lines.png  ({n_internal} internal + pia + white boundaries)")

    # generate layers_color_mask.png (multi-colour per-pixel layer mask)
    print("  Rendering layers_color_mask.png ...")
    layer_colors = [
        (255, 100, 100),   # L1  — light red
        (100, 255, 100),   # L2  — light green
        (100, 100, 255),   # L3  — light blue
        (255, 255, 100),   # L4  — yellow
        (255, 100, 255),   # L5  — pink
        (100, 255, 255),   # L6  — cyan
    ]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, layer in enumerate(layers):
        s = float(layer["start"])
        e = float(layer["end"])
        color = layer_colors[i % len(layer_colors)]
        region = (depth_map >= s) & (depth_map < e)
        color_mask[region] = color

    if mask is not None:
        color_mask[mask == 0] = 0
    cv2.imwrite(str(output_dir / "layers_color_mask.png"), color_mask)
    filled = int(np.count_nonzero(np.any(color_mask > 0, axis=2)))
    print(f"    -> layers_color_mask.png  ({filled} / {h * w} px filled)")


def _draw_dashed_points(img, pts, color, thickness, dash_len, gap_len, valid_mask=None):
    """Draw dashed polyline through an Nx2 point array."""
    n = pts.shape[0]
    i = 0
    while i < n:
        end = min(i + dash_len, n)
        if end - i >= 2:
            seg = pts[i:end].reshape((-1, 1, 2)).astype(np.int32)
            if valid_mask is not None:
                # check if segment midpoint is inside valid area
                mid = pts[(i + end) // 2] if (i + end) // 2 < n else pts[i]
                mx, my = int(mid[0]), int(mid[1])
                if 0 <= mx < valid_mask.shape[1] and 0 <= my < valid_mask.shape[0] and valid_mask[my, mx]:
                    cv2.polylines(img, [seg], False, color, thickness)
            else:
                cv2.polylines(img, [seg], False, color, thickness)
        i += dash_len + gap_len


def _draw_dashed_contour(img, contour, color, thickness, dash_len, gap_len, valid_mask=None):
    """Draw dashed contour by iterating its points."""
    pts = contour.squeeze()
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
        return
    _draw_dashed_points(img, pts, color, thickness, dash_len, gap_len, valid_mask)


# ── layer analysis (density + depth) ───────────────────────────────────────────────


def run_layer_analysis(
    pia_csv: Path,
    white_csv: Path,
    cell_csv: Path,
    dapi_path: Path,
    graymask_path: Path,
    output_dir: Path,
    depth_method: str | None = None,
    merge_layer23: bool = True,
    compact_rate: float | None = None,
) -> Path:
    """Run peak-based layer segmentation.

    Returns path to ``segmented_depth.csv``.
    """
    print(f"\n  ── Layer segmentation ──")

    # set module-level output dirs
    density.output_dir = str(output_dir)
    layer_visual.output_dir = str(output_dir)
    layer_visual.input_dir = str(output_dir)

    # load data
    wm_df = _load_boundary_csv(white_csv)
    gm_df = _load_boundary_csv(pia_csv)
    cells = _load_cells(cell_csv)

    # filter cells by graymask
    mask = None
    if graymask_path.exists():
        mask = _load_mask(graymask_path)
    if mask is not None:
        cells_before = len(cells)
        cells = _filter_cells_by_mask(cells, mask)
        print(f"  Cells inside graymask: {len(cells)} / {cells_before}")
    else:
        print(f"  Total cells: {len(cells)} (no graymask filter)")

    # depth + density
    depth_arr, dens_arr = density.analyze(
        wm_df, gm_df, cells,
        kde_bandwidth=KDE_BANDWIDTH,
        depth_method=_choose_param(depth_method, AUTO_DEPTH_METHOD),
    )
    avg_density, bin_centers = density.computeAverage(depth_arr, dens_arr, mode="average")

    # peak-based segmentation
    layers = density.segmentLayer_peak_based(
        avg_density, bin_centers,
        sigma=2,
        merge_layer23=merge_layer23,
        issave=True,
    )

    # save segmented_depth.csv (renamed from segmented_layers.csv)
    depth_csv = output_dir / "segmented_depth.csv"
    pd.DataFrame(layers).to_csv(depth_csv, index=False)
    print(f"  -> {depth_csv}")
    for L in layers:
        print(f"    Layer {L['layer']:>4s}: depth=[{L['start']:.3f}, {L['end']:.3f}]  "
              f"mean_density={L['mean_density']:.2f}")

    return depth_csv


# ── main ────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="New laminar segmentation pipeline — direct preprocessed inputs."
    )
    parser.add_argument(
        "--input-root", default="dataset/inputs/preprocessed",
        help="Root directory containing sample subfolders (default: dataset/inputs/preprocessed)"
    )
    parser.add_argument(
        "--output-root", default="dataset/outputs",
        help="Root directory for per-sample outputs (default: dataset/outputs)"
    )
    parser.add_argument(
        "--samples", nargs="*", default=None,
        help="Specific sample names to process (default: all subfolders in input-root)"
    )
    # cellpose options
    parser.add_argument("--no-gpu", action="store_true", help="Run Cellpose on CPU")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=64)
    tta = parser.add_mutually_exclusive_group()
    tta.add_argument("--no-tta", dest="no_tta", action="store_true", default=None)
    tta.add_argument("--tta", dest="no_tta", action="store_false")
    parser.add_argument("--downsample-rate", type=float, default=None)
    parser.add_argument("--cellpose-python", default=None)
    # layer options
    parser.add_argument("--depth-method", choices=["legacy", "harmonic"], default=None)
    parser.add_argument("--separate-l23", action="store_true", help="Do not merge L2/L3")
    parser.add_argument("--compact", action="store_true", help="Shortcut for --compact-rate 0.1")
    parser.add_argument("--compact-rate", type=float, default=None)
    parser.add_argument("--skip-cellpose", action="store_true",
                        help="Skip cell segmentation (use existing cell_centroids.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # determine sample folders
    if args.samples:
        sample_dirs = [input_root / s for s in args.samples]
    else:
        sample_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    if not sample_dirs:
        print(f"No sample directories found under {input_root}")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} sample(s): {[d.name for d in sample_dirs]}")
    print(f"Output root: {output_root.resolve()}")

    for sample_idx, sample_dir in enumerate(sample_dirs):
        sample_name = sample_dir.name
        out_dir = output_root / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Sample {sample_idx + 1}/{len(sample_dirs)}: {sample_name}")
        print(f"  Input:  {sample_dir}")
        print(f"  Output: {out_dir}")

        # ── resolve input files ──────────────────────────────────────────────
        dapi_path = _resolve_image_path(sample_dir, "dapi")
        pia_csv = sample_dir / "pia.csv"
        white_csv = sample_dir / "white.csv"
        graymask_path = sample_dir / "graymask.png"

        for p, desc in [(dapi_path, "DAPI image"), (pia_csv, "pia.csv"),
                        (white_csv, "white.csv"), (graymask_path, "graymask.png")]:
            if not p.exists():
                print(f"  [ERROR] Missing {desc}: {p}")
                print(f"  Skipping sample {sample_name}")
                break
        else:
            # all files exist -> proceed

            # ── Step 1: Cell segmentation ────────────────────────────────────
            if not args.skip_cellpose:
                cell_csv = segment_cells(
                    dapi_path=dapi_path,
                    graymask_path=graymask_path,
                    output_dir=out_dir,
                    use_gpu=not args.no_gpu,
                    batch_size=_choose_param(args.batch_size, AUTO_BATCH_SIZE),
                    tile_size=_choose_param(args.tile_size, AUTO_TILE_SIZE),
                    overlap=args.overlap,
                    use_tta=not _choose_param(args.no_tta, AUTO_NO_TTA),
                    downsample_rate=_choose_param(args.downsample_rate, AUTO_DOWNSAMPLE_RATE),
                    cellpose_python=_resolve_cellpose_python(args.cellpose_python),
                )
            else:
                cell_csv = out_dir / "cell_centroids.csv"
                if not cell_csv.exists():
                    print(f"  [ERROR] --skip-cellpose but no existing {cell_csv}")
                    continue
                print(f"  Skipping Cellpose (--skip-cellpose), using existing {cell_csv}")

            # ── Step 2: Layer analysis ───────────────────────────────────────
            merge_l23 = not args.separate_l23
            compact_rate = _choose_param(args.compact_rate, AUTO_COMPACT_RATE)
            if args.compact:
                compact_rate = 0.1

            depth_csv = run_layer_analysis(
                pia_csv=pia_csv,
                white_csv=white_csv,
                cell_csv=cell_csv,
                dapi_path=dapi_path,
                graymask_path=graymask_path,
                output_dir=out_dir,
                depth_method=args.depth_method,
                merge_layer23=merge_l23,
                compact_rate=compact_rate,
            )

            # ── Step 3: Boundary extraction + visualisations ─────────────────
            # (layer_lines.png, layers_color_mask.png, boundary CSVs)
            save_layer_boundaries(
                pia_csv=pia_csv,
                white_csv=white_csv,
                layers_csv=depth_csv,
                dapi_path=dapi_path,
                graymask_path=graymask_path,
                output_dir=out_dir,
                compact_rate=compact_rate,
            )

            # ── Clean up unwanted files from downstream modules ─────────────
            for unwanted in ("layers_overlay.png", "combined_visualization.png"):
                p = out_dir / unwanted
                if p.exists():
                    p.unlink()

            print(f"\n  [OK] Sample {sample_name} complete -> {out_dir}")

    print(f"\n{'=' * 70}")
    print(f"All done. Results in {output_root.resolve()}")


if __name__ == "__main__":
    main()
