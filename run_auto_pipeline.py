#!/usr/bin/env python
"""
Fully automatic laminar segmentation pipeline.

Inputs:
  - 4x image + 4x scan JSON
  - 40x/DAPI image + 40x scan JSON

Steps:
  1. Segment GM/WM boundaries and gray matter mask from the 4x image.
  2. Map GM/WM and gray mask to the 40x/DAPI mosaic coordinate system.
  3. Segment nuclei on the 40x/DAPI image with tiled Cellpose inference.
  4. Run peak-based laminar segmentation.
  5. Render final colored layer mask, overlay, and layer-line visualization.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import segment_boundaries as boundary
import src.analyseDensity as density
import src.layerVisualize as layer_visual
from src.layerVisualize import assign_layers_to_mask


# ── Cellpose parameter defaults for the automatic pipeline ──────────────────────
# Priority:
#   1. CLI arguments passed to run_auto_pipeline.py
#   2. Values below
#   3. Defaults inside the downstream script/module
#
# Set a value to None here to delegate that parameter to the downstream default.
AUTO_TILE_SIZE = 4096
AUTO_BATCH_SIZE = 64
AUTO_DOWNSAMPLE_RATE = 0.8
AUTO_NO_TTA = True             # True = Cellpose TTA off by default; False = TTA on
AUTO_DEPTH_METHOD = "legacy"       # "legacy" | "harmonic" | None -> src/analyseDensity.py default
AUTO_COMPACT_RATE = 0.1       # 1.0 = no compression, 0.1 = old 10:1 compact, None -> layerVisualize default
AUTO_BOUNDARY_DOWNSAMPLE_RATE = 0.2  # 4x GM/WM segmentation scale; None -> segment_boundaries.py default
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)


def _load_cells(csv_path: Path) -> pd.DataFrame:
    """Load centroid CSV and normalize coordinate columns to X/Y."""
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "centroid_x" in cols_lower and "X" not in df.columns:
        rename_map[cols_lower["centroid_x"]] = "X"
    if "centroid_y" in cols_lower and "Y" not in df.columns:
        rename_map[cols_lower["centroid_y"]] = "Y"
    if "x" in cols_lower and "X" not in df.columns:
        rename_map[cols_lower["x"]] = "X"
    if "y" in cols_lower and "Y" not in df.columns:
        rename_map[cols_lower["y"]] = "Y"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"Cannot find centroid coordinate columns in {csv_path}: {list(df.columns)}")
    return df[["X", "Y"]].copy()


def _load_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} != expected image shape {shape}")
    return (mask > 0).astype(np.uint8) * 255


def _filter_cells_by_mask(cells: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    coords = cells[["X", "Y"]].to_numpy(dtype=int)
    in_bounds = (
        (coords[:, 1] >= 0)
        & (coords[:, 1] < mask.shape[0])
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < mask.shape[1])
    )
    keep = np.zeros(len(cells), dtype=bool)
    keep[in_bounds] = mask[coords[in_bounds, 1], coords[in_bounds, 0]] > 0
    return cells.loc[keep].reset_index(drop=True)


def segment_boundaries_and_mask(
    image_4x_path: Path,
    json_4x_path: Path,
    image_40x_path: Path,
    json_40x_path: Path,
    output_dir: Path,
    boundary_downsample_rate: float | None = None,
    objective_offset_path: Path | None = None,
) -> dict[str, Path]:
    """Run 4x tissue segmentation and map boundaries/mask into 40x coordinates."""
    image_4x = boundary.read_gray_image(image_4x_path)
    image_40x = boundary.read_color_image(image_40x_path)
    h4, w4 = image_4x.shape[:2]
    h40, w40 = image_40x.shape[:2]

    print("\n[1/4] Segmenting GM/WM boundaries and gray mask from 4x image")
    print(f"  4x image: {w4} x {h4}")
    print(f"  40x/DAPI image: {w40} x {h40}")

    seg = boundary.segment_4x_image(image_4x, downsample_rate=boundary_downsample_rate)
    tissue_mask = seg["tissue_mask"]
    white_mask = seg["white_mask"]
    gray_mask = seg["gray_mask"]
    gm_contours = seg["gm_contours"]
    wm_contours = seg["wm_contours"]
    tissue_image = cv2.bitwise_and(image_4x, image_4x, mask=tissue_mask)

    gm_4x = boundary.contours_to_points(gm_contours)
    wm_4x = boundary.contours_to_points(wm_contours)
    print(f"  4x boundary points: GM={len(gm_4x)}, WM={len(wm_4x)}")

    cv2.imwrite(str(output_dir / "tissueMask.png"), tissue_mask)
    cv2.imwrite(str(output_dir / "whiteMask.png"), white_mask)
    cv2.imwrite(str(output_dir / "grayMask.png"), gray_mask)
    cv2.imwrite(str(output_dir / "grayImage.png"), cv2.bitwise_and(image_4x, image_4x, mask=gray_mask))
    boundary.save_points_csv(gm_4x, output_dir / "GM_4x.csv")
    boundary.save_points_csv(wm_4x, output_dir / "WM_4x.csv")
    boundary.draw_boundary_visualization(
        image_4x,
        gm_contours,
        wm_contours,
        output_dir / "OuterInnerPoints_4x.png",
        thickness=max(6, min(h4, w4) // 350),
    )

    meta_4x = boundary.read_scan_metadata(json_4x_path, image_4x.shape[:2])
    meta_40x = boundary.read_scan_metadata(json_40x_path, image_40x.shape[:2])
    objective_offset = boundary.read_objective_offset(objective_offset_path) if objective_offset_path else None
    matrix = boundary.affine_4x_to_40x(meta_4x, meta_40x, objective_offset=objective_offset)
    print(f"  4x -> 40x affine:\n{matrix}")

    gm_40x = boundary.transform_points(gm_4x, matrix)
    wm_40x = boundary.transform_points(wm_4x, matrix)
    gm_40x_clipped = boundary.clip_points(gm_40x, w40, h40)
    wm_40x_clipped = boundary.clip_points(wm_40x, w40, h40)
    print(f"  40x clipped points: GM={len(gm_40x_clipped)}, WM={len(wm_40x_clipped)}")

    boundary.save_points_csv(gm_40x, output_dir / "GM_40x.csv")
    boundary.save_points_csv(wm_40x, output_dir / "WM_40x.csv")
    boundary.save_points_csv(gm_40x_clipped, output_dir / "GM_40x_clipped.csv")
    boundary.save_points_csv(wm_40x_clipped, output_dir / "WM_40x_clipped.csv")
    boundary.save_points_csv(gm_40x_clipped, output_dir / "GM.csv")
    boundary.save_points_csv(wm_40x_clipped, output_dir / "WM.csv")

    gray_mask_40x = boundary.transform_mask(gray_mask, matrix, image_40x.shape[:2])
    cv2.imwrite(str(output_dir / "grayMask_40x.png"), gray_mask_40x)
    cv2.imwrite(str(output_dir / "mask.png"), gray_mask_40x)

    gm_segments = boundary.split_in_bounds_segments(gm_40x, w40, h40)
    wm_segments = boundary.split_in_bounds_segments(wm_40x, w40, h40)

    # Save 4x segmentation parameters for reproducibility.
    params = boundary.collect_parameters(
        argparse.Namespace(boundary_downsample_rate=boundary_downsample_rate),
        image_4x.shape[:2],
    )
    params_path = output_dir / "boundary_parameters.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    boundary.draw_boundary_visualization(
        image_40x,
        gm_segments,
        wm_segments,
        output_dir / "OuterInnerPoints_40x_clipped.png",
        thickness=max(10, min(h40, w40) // 500),
    )

    return {
        "gm_csv": output_dir / "GM.csv",
        "wm_csv": output_dir / "WM.csv",
        "mask": output_dir / "mask.png",
    }


def segment_cells_40x(
    image_40x_path: Path,
    mask_path: Path,
    output_dir: Path,
    use_gpu: bool,
    batch_size: int | None,
    tile_size: int | None,
    overlap: int,
    use_tta: bool,
    downsample_rate: float | None,
    cellpose_python: Path | None = None,
) -> Path:
    """Run tiled Cellpose nuclei segmentation and save cell_centroids.csv."""
    print("\n[2/4] Segmenting nuclei on 40x/DAPI image")

    if cellpose_python is not None:
        cmd = [
            str(cellpose_python),
            str(Path(__file__).with_name("segment_large_image_fast.py")),
            str(image_40x_path),
            "--output_dir",
            str(output_dir),
            "--overlap",
            str(overlap),
            "--mask-path",
            str(mask_path),
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

        print(f"  Running Cellpose with: {cellpose_python}")
        subprocess.run(cmd, check=True)

        stem_csv = output_dir / f"{image_40x_path.stem}_cell_centroids.csv"
        cell_csv = output_dir / "cell_centroids.csv"
        if not stem_csv.exists():
            raise FileNotFoundError(f"Expected Cellpose centroid CSV not found: {stem_csv}")
        pd.read_csv(stem_csv).to_csv(cell_csv, index=False)
        print(f"  Cell centroids saved: {cell_csv}")
        return cell_csv

    # Import lazily so --help and boundary-only diagnostics do not require Cellpose import.
    import tifffile
    import segment_large_image_fast as cellseg

    if tile_size is not None:
        cellseg.TILE_SIZE = tile_size
    tile_size = cellseg.TILE_SIZE
    cellseg.OVERLAP = overlap
    cellseg.STRIDE = tile_size - 2 * overlap
    if cellseg.STRIDE <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size/2 ({tile_size / 2})")

    image = cellseg.load_image(image_40x_path)
    region_mask_full = _load_mask(mask_path, image.shape) > 0
    image_full = image
    if downsample_rate is None:
        downsample_rate = cellseg.DOWNSAMPLE_RATE
    image, region_mask = cellseg.downsample_image_and_mask(image_full, region_mask_full, downsample_rate)
    global_mask, cell_records = cellseg.segment_large_image(
        image,
        use_gpu=use_gpu,
        batch_size=batch_size,
        use_tta=use_tta,
        region_mask=region_mask,
        downsample_rate=downsample_rate,
    )

    stem = image_40x_path.stem
    stem_csv = output_dir / f"{stem}_cell_centroids.csv"
    cell_csv = output_dir / "cell_centroids.csv"
    df = pd.DataFrame(cell_records)
    df.to_csv(stem_csv, index=False)
    df.to_csv(cell_csv, index=False)
    print(f"  Cell centroids saved: {cell_csv}")

    full_res_mask = cellseg.restore_label_mask(global_mask, image_full.shape)
    full_mask_path = output_dir / f"{stem}_full_mask.tif"
    tifffile.imwrite(str(full_mask_path), full_res_mask.astype(np.uint32), compression="zlib")
    cellseg.save_random_color_mask(full_res_mask, output_dir / f"{stem}_mask_color.png")
    cellseg.save_overlay(image_full, full_res_mask, output_dir / f"{stem}_overlay.png")
    return cell_csv


def run_layer_pipeline(
    gm_csv: Path,
    wm_csv: Path,
    cell_csv: Path,
    image_40x_path: Path,
    mask_path: Path,
    output_dir: Path,
    kde_bandwidth: str,
    depth_method: str | None,
    merge_layer23: bool,
    compact_rate: float | None,
) -> None:
    """Run peak-based layer segmentation and final DAPI visualizations."""
    print("\n[3/4] Running peak-based layer segmentation")
    density.output_dir = str(output_dir)
    layer_visual.output_dir = str(output_dir)
    layer_visual.input_dir = str(output_dir)

    wm_df = pd.read_csv(wm_csv)
    gm_df = pd.read_csv(gm_csv)
    cells = _load_cells(cell_csv)
    mask = _load_mask(mask_path, cv2.imread(str(image_40x_path), cv2.IMREAD_GRAYSCALE).shape[:2])
    cells = _filter_cells_by_mask(cells, mask)
    print(f"  Cells inside gray mask: {len(cells)}")

    depth, dens = density.analyze(
        wm_df,
        gm_df,
        cells,
        kde_bandwidth=kde_bandwidth,
        depth_method=depth_method,
    )
    avg_density, bin_centers = density.computeAverage(depth, dens, mode="average")
    layers = density.segmentLayer_peak_based(
        avg_density,
        bin_centers,
        sigma=2,
        merge_layer23=merge_layer23,
        issave=True,
    )

    layers_csv = output_dir / "segmented_layers.csv"
    pd.DataFrame(layers).to_csv(layers_csv, index=False)
    print(f"  Segmented layers saved: {layers_csv}")

    print("\n[4/4] Rendering final layer visualizations")
    assign_layers_to_mask(
        str(wm_csv),
        str(gm_csv),
        str(layers_csv),
        str(image_40x_path),
        issave=True,
        save_dir=str(output_dir),
        mask_img=mask,
        compact_rate=compact_rate,
    )


def run_result_analysis(
    output_dir: Path,
    input_dir: Path,
    groundtruth_path: Path,
) -> None:
    """Run resultanalysis.py metrics when a groundtruth mask is available."""
    print("\n[5/5] Running result analysis")
    if not groundtruth_path.exists():
        print(f"  Ground truth not found, skip result analysis: {groundtruth_path}")
        return

    from resultanalysis import LayerResultAnalyzer

    analyzer = LayerResultAnalyzer(
        output_dir=str(output_dir),
        input_dir=str(input_dir),
        groundtruth_path=str(groundtruth_path),
    )
    analyzer.run_full_analysis()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full automatic laminar segmentation from raw 4x/40x inputs.")
    parser.add_argument("--input-dir", default="input", help="Directory containing default input files")
    parser.add_argument("--output-dir", default="output/auto_pipeline", help="Directory for all generated outputs")
    parser.add_argument("--image-4x", default="4x.png", help="4x image path or filename under input-dir")
    parser.add_argument("--json-4x", default="4x.json", help="4x scan JSON path or filename under input-dir")
    parser.add_argument("--image-40x", default="dapi.png", help="40x/DAPI image path or filename under input-dir")
    parser.add_argument("--json-40x", default="40x.json", help="40x scan JSON path or filename under input-dir")
    parser.add_argument("--groundtruth", default="groundtruth.png",
                        help="Ground truth mask path or filename under input-dir for final resultanalysis.py metrics")
    parser.add_argument("--skip-result-analysis", action="store_true",
                        help="Skip final resultanalysis.py evaluation even if groundtruth exists")
    parser.add_argument("--4x-downsample-rate", "--boundary-downsample-rate",
                        dest="boundary_downsample_rate", type=float, default=None,
                        help=f"4x GM/WM segmentation scale. 1=no downsample, 0.5=half, 0.25=quarter. Default: run_auto AUTO_BOUNDARY_DOWNSAMPLE_RATE={AUTO_BOUNDARY_DOWNSAMPLE_RATE}; then segment_boundaries default.")
    parser.add_argument("--no-gpu", action="store_true", help="Run Cellpose on CPU")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Cellpose batch size. Default: run_auto AUTO_BATCH_SIZE={AUTO_BATCH_SIZE}; then segment_large_image_fast default.")
    parser.add_argument("--tile-size", type=int, default=None,
                        help=f"Cellpose tile size. Default: run_auto AUTO_TILE_SIZE={AUTO_TILE_SIZE}; then segment_large_image_fast default.")
    parser.add_argument("--overlap", type=int, default=64, help="Cellpose tile overlap")
    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument("--no-tta", dest="no_tta", action="store_true", default=None,
                           help=f"Disable Cellpose test-time augmentation. Default: run_auto AUTO_NO_TTA={AUTO_NO_TTA}; then segment_large_image_fast default.")
    tta_group.add_argument("--tta", dest="no_tta", action="store_false",
                           help="Enable Cellpose test-time augmentation")
    parser.add_argument("--objective-offset-config", default="ObjectiveOffsetConfig.json",
                        help="Objective offset calibration JSON (relative to input-dir or absolute path)")
    parser.add_argument("--downsample-rate", type=float, default=None,
                        help=f"Downsample 40x image for Cellpose inference only. Default: run_auto AUTO_DOWNSAMPLE_RATE={AUTO_DOWNSAMPLE_RATE}; then segment_large_image_fast default.")
    parser.add_argument("--cellpose-python", default=None,
                        help="Python executable with Cellpose installed. Auto-detects venv-cellpose when needed.")
    parser.add_argument("--kde-bandwidth", default="scott", help="KDE bandwidth for density estimation")
    parser.add_argument("--depth-method", choices=["legacy", "harmonic"], default=None,
                        help=f"Depth calculation method. Default: run_auto AUTO_DEPTH_METHOD={AUTO_DEPTH_METHOD}; then analyseDensity default.")
    parser.add_argument("--separate-l23", action="store_true", help="Do not merge L2/L3")
    parser.add_argument("--compact", action="store_true", help="Backward-compatible shortcut for --compact-rate 0.1")
    parser.add_argument("--compact-rate", type=float, default=None,
                        help=f"Final visualization scale. 1=no compression, 0.1=old 10:1 compact. Default: run_auto AUTO_COMPACT_RATE={AUTO_COMPACT_RATE}; then layerVisualize default.")
    return parser.parse_args()


def _resolve_path(input_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_dir / path


def _resolve_image_path(input_dir: Path, value: str) -> Path:
    """Resolve an image by exact path first, then same-stem common image formats."""
    base = _resolve_path(input_dir, value)
    if base.exists():
        return base

    stem = base.stem if base.suffix else base.name
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = base.parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    matches = [p for p in base.parent.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        priority = {ext: idx for idx, ext in enumerate(IMAGE_EXTENSIONS)}
        return sorted(matches, key=lambda p: priority.get(p.suffix.lower(), 999))[0]

    return base


def _resolve_cellpose_python(value: str | None) -> Path | None:
    """Return a separate Cellpose Python executable when current Python lacks Cellpose."""
    if value:
        return Path(value)
    if importlib.util.find_spec("cellpose") is not None:
        return None

    candidates = [
        Path("venv-cellpose") / "Scripts" / "python.exe",
        Path(".venv-cellpose") / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Cellpose is not installed in the current Python and no venv-cellpose Python was found. "
        "Run with --cellpose-python <path-to-python-with-cellpose>."
    )


def _choose_param(cli_value, auto_value):
    """CLI overrides run_auto default; None delegates to the downstream module default."""
    return cli_value if cli_value is not None else auto_value


def _choose_compact_rate(cli_rate, cli_compact, auto_rate):
    """CLI compact-rate overrides --compact shortcut, then run_auto default, then layerVisualize default."""
    if cli_rate is not None:
        return cli_rate
    if cli_compact:
        return 0.1
    return auto_rate


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_4x_path = _resolve_image_path(input_dir, args.image_4x)
    json_4x_path = _resolve_path(input_dir, args.json_4x)
    image_40x_path = _resolve_image_path(input_dir, args.image_40x)
    json_40x_path = _resolve_path(input_dir, args.json_40x)
    groundtruth_path = _resolve_image_path(input_dir, args.groundtruth)

    objective_offset_path = _resolve_path(input_dir, args.objective_offset_config)
    boundary_paths = segment_boundaries_and_mask(
        image_4x_path=image_4x_path,
        json_4x_path=json_4x_path,
        image_40x_path=image_40x_path,
        json_40x_path=json_40x_path,
        output_dir=output_dir,
        boundary_downsample_rate=_choose_param(args.boundary_downsample_rate, AUTO_BOUNDARY_DOWNSAMPLE_RATE),
        objective_offset_path=objective_offset_path,
    )

    cell_csv = segment_cells_40x(
        image_40x_path=image_40x_path,
        mask_path=boundary_paths["mask"],
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
        batch_size=_choose_param(args.batch_size, AUTO_BATCH_SIZE),
        tile_size=_choose_param(args.tile_size, AUTO_TILE_SIZE),
        overlap=args.overlap,
        use_tta=not _choose_param(args.no_tta, AUTO_NO_TTA),
        downsample_rate=_choose_param(args.downsample_rate, AUTO_DOWNSAMPLE_RATE),
        cellpose_python=_resolve_cellpose_python(args.cellpose_python),
    )

    run_layer_pipeline(
        gm_csv=boundary_paths["gm_csv"],
        wm_csv=boundary_paths["wm_csv"],
        cell_csv=cell_csv,
        image_40x_path=image_40x_path,
        mask_path=boundary_paths["mask"],
        output_dir=output_dir,
        kde_bandwidth=args.kde_bandwidth,
        depth_method=_choose_param(args.depth_method, AUTO_DEPTH_METHOD),
        merge_layer23=not args.separate_l23,
        compact_rate=_choose_compact_rate(args.compact_rate, args.compact, AUTO_COMPACT_RATE),
    )

    if not args.skip_result_analysis:
        run_result_analysis(
            output_dir=output_dir,
            input_dir=input_dir,
            groundtruth_path=groundtruth_path,
        )

    print(f"\nAuto pipeline complete. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
