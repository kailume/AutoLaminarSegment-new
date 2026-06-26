"""
简化版密度分层Pipeline

跳过细胞分割步骤，直接使用已有的输入文件进行分层：
  input/WM.csv            - 白质边界坐标 (x, y)
  input/GM.csv            - 灰质边界坐标 (x, y)
  input/cell_centroids.csv - 细胞核坐标 (centroid_x, centroid_y, ...)

输出到 output/ 目录：
  segmented_layers.csv    - 分层结果
  depth_density_layers_peak_based.png - 分层可视化曲线图
  layers_color_mask.png   - 分层颜色填充Mask（可选）
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
INPUT_DIR  = Path("input")
OUTPUT_DIR = Path("output")      # 可在命令行用 --output-dir 覆盖

# ── 分层参数（可修改）────────────────────────────────────
MERGE_23 = True              # True → 合并L2/3，输出4层；False → 输出5层
KDE_BANDWIDTH = "scott"      # KDE带宽: 'scott' | 'silverman' | float
PIPELINE_DEPTH_METHOD = None     # "legacy" | "harmonic" | None -> src/analyseDensity.py default
PIPELINE_COMPACT_RATE = None     # 1.0 = no compression, 0.1 = old 10:1 compact, None -> layerVisualize default
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)



# ─────────────────────────────────────────────────────────
# 将 src/analyseDensity.py 的 output_dir 指向我们的输出目录
# （该模块使用模块级变量控制保存路径）
import src.analyseDensity as _ad
_ad.output_dir = str(OUTPUT_DIR)

import src.layerVisualize as _lv
_lv.output_dir = str(OUTPUT_DIR)
_lv.input_dir = str(INPUT_DIR)

from src.analyseDensity import (
    analyze, computeAverage,
    segmentLayer_peak_based,
)
from src.layerVisualize import assign_layers_to_mask


def load_cells(csv_path: Path) -> pd.DataFrame:
    """读取细胞坐标，统一映射为 X / Y 列名。"""
    df = pd.read_csv(csv_path)
    # 兼容多种列名约定
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
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
        raise ValueError(
            f"无法在 {csv_path} 中找到坐标列，当前列名: {list(df.columns)}"
        )
    return df[["X", "Y"]]


def resolve_image_path(input_dir: Path, value: str | Path) -> Path:
    """
    Resolve an image path by exact filename first, then by same-stem image formats.

    If value is "dapi.png" but only "dapi.tif" exists, this returns "dapi.tif".
    If value is "dapi", it searches dapi.tif/dapi.png/dapi.jpg/etc.
    """
    raw = Path(value)
    base = raw if raw.is_absolute() else input_dir / raw
    if base.exists():
        return base

    parent = base.parent
    stem = base.stem if base.suffix else base.name
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    matches = [p for p in parent.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        priority = {ext: idx for idx, ext in enumerate(IMAGE_EXTENSIONS)}
        return sorted(matches, key=lambda p: priority.get(p.suffix.lower(), 999))[0]

    return base


def load_image_shape(image_path: Path):
    """Return (height, width) for the visualization image without changing pixel data."""
    if not image_path.exists():
        return None

    try:
        import cv2 as _cv2
        img = _cv2.imread(str(image_path), _cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.shape[:2]
    except Exception:
        pass

    try:
        from PIL import Image as _PIL
        import PIL
        _saved_max = PIL.Image.MAX_IMAGE_PIXELS
        PIL.Image.MAX_IMAGE_PIXELS = None
        with _PIL.open(str(image_path)) as img:
            w, h = img.size
        PIL.Image.MAX_IMAGE_PIXELS = _saved_max
        return h, w
    except Exception:
        return None


def filter_boundary_to_image(df: pd.DataFrame, image_shape, name: str) -> pd.DataFrame:
    """Drop GM/WM points outside the DAPI image before depth analysis."""
    if image_shape is None:
        return df

    h, w = image_shape
    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x")
    y_col = cols_lower.get("y")
    if x_col is None or y_col is None:
        return df

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    in_bounds = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    removed = int(len(df) - int(in_bounds.sum()))
    if removed:
        print(f"  [边界裁剪] {name}: 移除图外/无效点 {removed}/{len(df)}，DAPI尺寸={w}x{h}")

    filtered = df.loc[in_bounds].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"{name} 边界在 dapi.png 范围内没有可用点")
    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Run laminar segmentation pipeline.")
    parser.add_argument("input_dir", nargs="?", default="input",
                        help="输入文件夹路径（包含 WM.csv, GM.csv, cell_centroids.csv）(default: input)")
    parser.add_argument("--output-dir",
                        help="输出文件夹路径（默认与输入文件夹相同）")
    parser.add_argument("--image", default="dapi.png",
                        help="DAPI/40x image filename or path. If missing, search the same stem across tif/png/jpg/etc.")
    parser.add_argument("--mask", default="mask.png",
                        help="Mask filename or path. If missing, search the same stem across tif/png/jpg/etc.")
    parser.add_argument("--compact", action="store_true",
                        help="兼容旧参数；等同于 --compact-rate 0.1")
    parser.add_argument("--compact-rate", type=float, default=None,
                        help=f"最终可视化缩放比例，1=不压缩，0.1=原10:1压缩。默认: PIPELINE_COMPACT_RATE={PIPELINE_COMPACT_RATE}; then layerVisualize default.")
    parser.add_argument("--depth-method", choices=["legacy", "harmonic"], default=None,
                        help=f"深度计算方式。默认: PIPELINE_DEPTH_METHOD={PIPELINE_DEPTH_METHOD}; then analyseDensity default.")
    parser.add_argument("--refine", action="store_true",
                        help="启用边界精炼模型（基于 depth-density 曲线回归精炼层边界）")
    parser.add_argument("--refine-model", default="dataset/boundary_model.pkl",
                        help="边界精炼模型路径 (default: dataset/boundary_model.pkl)")
    return parser.parse_args()


def _choose_param(cli_value, pipeline_value):
    """CLI overrides run_pipeline default; None delegates to downstream module default."""
    return cli_value if cli_value is not None else pipeline_value


def _choose_compact_rate(cli_rate, cli_compact, pipeline_rate):
    """CLI compact-rate overrides --compact shortcut, then pipeline default, then layerVisualize default."""
    if cli_rate is not None:
        return cli_rate
    if cli_compact:
        return 0.1
    return pipeline_rate


def main():
    args = parse_args()
    depth_method = _choose_param(args.depth_method, PIPELINE_DEPTH_METHOD)
    compact_rate = _choose_compact_rate(args.compact_rate, args.compact, PIPELINE_COMPACT_RATE)
    INPUT_DIR  = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else INPUT_DIR
    IMG_PATH   = resolve_image_path(INPUT_DIR, args.image)
    MSK_PATH   = resolve_image_path(INPUT_DIR, args.mask)

    # 更新模块级路径变量
    _ad.output_dir = str(OUTPUT_DIR)
    _lv.input_dir = str(INPUT_DIR)
    _lv.output_dir = str(OUTPUT_DIR)

    print("=" * 60)
    print("  密度分层Pipeline")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    if compact_rate is not None and compact_rate < 1:
        print(f"  压缩模式: rate={compact_rate}（仅用于最终图像可视化）")
    print("=" * 60)

    # ── 准备输出目录 ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visualization_shape = load_image_shape(IMG_PATH)
    if visualization_shape is not None:
        vh, vw = visualization_shape
        print(f"  DAPI图像尺寸: {vw} x {vh}")

    # ── 验证输入文件 ──────────────────────────────────────
    wm_path   = INPUT_DIR / "WM.csv"
    gm_path   = INPUT_DIR / "GM.csv"
    cell_path = INPUT_DIR / "cell_centroids.csv"

    for p in [wm_path, gm_path, cell_path]:
        if not p.exists():
            print(f"[错误] 缺少输入文件: {p}")
            sys.exit(1)
    print(f"[✓] 输入文件验证通过")

    # ── Step 1: 读取数据 ──────────────────────────────────
    print("\n[步骤1] 读取边界与细胞坐标...")
    wm_df   = pd.read_csv(wm_path)
    gm_df   = pd.read_csv(gm_path)
    cells   = load_cells(cell_path)

    print(f"  WM边界点数: {len(wm_df)}")
    print(f"  GM边界点数: {len(gm_df)}")
    print(f"  细胞数:     {len(cells)}")
    wm_df = filter_boundary_to_image(wm_df, visualization_shape, "WM")
    gm_df = filter_boundary_to_image(gm_df, visualization_shape, "GM")
    print(f"  图内边界点数: WM={len(wm_df)}, GM={len(gm_df)}")

    # ── Step 1.5: 按 mask 过滤细胞（可选）──────────────
    mask_img = None
    if MSK_PATH.exists():
        print(f"\n[步骤1.5] 按 mask 过滤细胞: {MSK_PATH}")
        # 尝试多种方式加载大图 mask（cv2 → skimage → PIL）
        try:
            import cv2 as _cv2
            mask_img = _cv2.imread(str(MSK_PATH), _cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask_img = None
        if mask_img is None:
            try:
                from skimage import io as _io
                mask_img = _io.imread(str(MSK_PATH))
                if mask_img.ndim == 3:
                    mask_img = mask_img[:, :, 0]
                mask_img = (mask_img > 0).astype(np.uint8) * 255
                print(f"  (使用 skimage 加载掩码)")
            except Exception as e:
                print(f"  skimage 失败: {e}")
                mask_img = None
        if mask_img is None:
            try:
                from PIL import Image as _PIL
                import PIL
                _saved_max = PIL.Image.MAX_IMAGE_PIXELS
                PIL.Image.MAX_IMAGE_PIXELS = None  # 关闭解压炸弹保护
                _pil_img = _PIL.open(str(MSK_PATH))
                mask_img = np.array(_pil_img.convert("L"))
                PIL.Image.MAX_IMAGE_PIXELS = _saved_max
                mask_img = (mask_img > 0).astype(np.uint8) * 255
                print(f"  (使用 PIL 加载掩码)")
            except Exception as e:
                print(f"  PIL 失败: {e}")
                mask_img = None
        if mask_img is not None:
            print(f"  掩码尺寸: {mask_img.shape},  dtype={mask_img.dtype}")
            cells_before = len(cells)
            coords = cells[['X', 'Y']].values.astype(int)
            in_mask = (coords[:, 1] >= 0) & (coords[:, 1] < mask_img.shape[0]) & \
                      (coords[:, 0] >= 0) & (coords[:, 0] < mask_img.shape[1])
            in_mask[in_mask] = mask_img[coords[in_mask, 1], coords[in_mask, 0]] > 0
            cells = cells[in_mask].reset_index(drop=True)
            print(f"  过滤前细胞数: {cells_before}")
            print(f"  过滤后细胞数: {len(cells)}")
            print(f"  过滤掉的细胞: {cells_before - len(cells)}")
        else:
            print(f"  [警告] mask 读取失败: {MSK_PATH}")
            mask_img = None
    elif MSK_PATH:
        print(f"\n[步骤1.6] mask 文件不存在，跳过: {MSK_PATH}")

    # ── Step 2: 计算深度与密度 ────────────────────────────
    print("\n[步骤2] 计算细胞深度与局部密度...")
    depth, density = analyze(
        wm_df, gm_df, cells,
        kde_bandwidth=KDE_BANDWIDTH,
        depth_method=depth_method,
    )

    # ── Step 3: 深度-密度平均曲线 ─────────────────────────
    print("\n[步骤3] 计算深度-平均密度曲线...")
    avg_density, bin_centers = computeAverage(
        depth, density,
        mode="average",
    )

    # ── Step 4: 分层 ──────────────────────────────────────
    print("\n[步骤4] 使用 peak_based 方法进行分层...")
    layers = segmentLayer_peak_based(
        avg_density, bin_centers,
        sigma=2,
        merge_layer23=MERGE_23,
        issave=True,
    )

    # ── 保存分层结果 ──────────────────────────────────────
    layers_csv = OUTPUT_DIR / "segmented_layers.csv"
    pd.DataFrame(layers).to_csv(layers_csv, index=False)
    print(f"\n[✓] 分层结果已保存: {layers_csv}")

    print("\n  分层结果摘要:")
    for L in layers:
        print(f"    Layer {L['layer']:>4s}: depth=[{L['start']:.3f}, {L['end']:.3f}]"
              f"  mean_density={L['mean_density']:.2f}")

    # ── Step 4.5: 边界精炼（可选）─────────────────────────
    if args.refine:
        print("\n[步骤4.5] 边界精炼 (curve→boundary regression)...")
        refine_model_path = Path(args.refine_model)
        if not refine_model_path.exists():
            print(f"  [警告] 精炼模型不存在: {refine_model_path}，跳过")
        else:
            try:
                from src.densityRefinement import DensityDrivenRefiner

                refiner = DensityDrivenRefiner()
                refiner.load(str(refine_model_path))

                # 从密度剖面预测精炼边界
                boundaries, _ = refiner.predict(
                    os.path.basename(str(OUTPUT_DIR)) if os.path.basename(str(OUTPUT_DIR)).isdigit() else os.path.basename(str(INPUT_DIR)),
                    every_n=5,
                )
                refined_layers = []
                for i, name in enumerate(["1", "2/3", "4", "5/6"]):
                    s = 0.0 if i == 0 else boundaries[i-1]
                    e = boundaries[i] if i < 3 else 1.0
                    refined_layers.append({"layer": name, "start": float(s), "end": float(e), "mean_density": 0.0})

                # 输出精炼结果
                refined_csv = OUTPUT_DIR / "segmented_layers_refined.csv"
                pd.DataFrame(refined_layers).to_csv(refined_csv, index=False)
                print(f"  [✓] 精炼分层结果已保存: {refined_csv}")

                print("\n  精炼分层结果摘要:")
                for L in refined_layers:
                    print(f"    Layer L{L['layer']:>4s}: depth=[{L['start']:.3f}, {L['end']:.3f}]")

                # 生成精炼结果的图像可视化（独立子目录，不覆盖 coarse 可视化）
                if IMG_PATH.exists():
                    refined_vis_dir = OUTPUT_DIR / "refined_visualization"
                    refined_vis_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n  生成精炼结果可视化 -> {refined_vis_dir}")
                    try:
                        assign_layers_to_mask(
                            str(wm_path), str(gm_path),
                            str(refined_csv),
                            str(IMG_PATH),
                            issave=True,
                            save_dir=str(refined_vis_dir),
                            mask_img=mask_img,
                            compact_rate=compact_rate,
                        )
                        # 也在主目录生成简化的边界线对比图
                        print(f"  [✓] 精炼可视化已保存: {refined_vis_dir}/")
                    except Exception as ve:
                        print(f"  [警告] 精炼可视化失败: {ve}")

            except ImportError as e:
                print(f"  [警告] 加载边界精炼模块失败: {e}")
            except Exception as e:
                print(f"  [警告] 边界精炼失败: {e}")
                import traceback
                traceback.print_exc()

    # ── Step 5: 图像可视化（可选）────────────────────────
    if IMG_PATH.exists():
        print(f"\n[步骤5] 在图像上绘制分层结果: {IMG_PATH}")
        assign_layers_to_mask(
            str(wm_path), str(gm_path),
            str(layers_csv),
            str(IMG_PATH),
            issave=True,
            save_dir=str(OUTPUT_DIR),
            mask_img=mask_img,
            compact_rate=compact_rate,
        )
        print(f"[✓] 可视化图像已保存到 {OUTPUT_DIR}")
    else:
        print(f"\n[警告] 图像文件不存在: {IMG_PATH}，跳过图像可视化步骤")

    print("\n" + "=" * 60)
    print("  Pipeline完成！输出目录:", OUTPUT_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
