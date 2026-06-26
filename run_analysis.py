#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意: 请使用 venv-cellpose 环境运行:
  .\\venv-cellpose\\Scripts\\python run_analysis.py

默认的 Anaconda Python 存在 numpy 版本冲突 (numpy 2.x 与旧编译模块不兼容)。
"""
自动分析 dataset/outputs 中所有样本的分层结果指标。

对每个样本:
  1. 读取算法输出的 layers_color_mask.png
  2. 读取 Ground Truth 的 label_mask.png
  3. 计算 IoU / Dice / Precision / Recall / 边界误差 / 厚度误差
  4. 保存到 dataset/analysis/<sample>/

汇总:
  5. 计算所有样本的汇总统计 → dataset/analysis/summary/stats_summary.csv
  6. 绘制可视化图像 → dataset/analysis/summary/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# ── 中文字体 ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ──
ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "dataset" / "outputs"
LABEL_DIR = ROOT / "dataset" / "inputs" / "label"
ANALYSIS_DIR = ROOT / "dataset" / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"

# ── 固定颜色映射 (BGR) ──
# 算法输出颜色 (来自 run_new_pipeline.py)
ALGO_LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "L1": (255, 100, 100),
    "L2/3": (100, 255, 100),
    "L4": (100, 100, 255),
    "L5/6": (255, 255, 100),
}

# Ground Truth 候选颜色 (来自 resultanalysis.py)
GT_LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "L1": (255, 100, 100),
    "L2/3": (100, 255, 100),
    "L4": (100, 100, 255),
    "L5/6": (255, 100, 255),
}

# 替代 GT 颜色 (可能不同的标注工具使用了不同调色板)
ALT_GT_COLORS: dict[str, list[tuple[int, int, int]]] = {
    "L1": [(25, 28, 252), (255, 100, 100)],
    "L2/3": [(18, 126, 126), (100, 255, 100)],
    "L4": [(255, 255, 76), (100, 100, 255)],
    "L5/6": [(127, 127, 248), (255, 100, 255)],
}

STANDARD_LAYERS = ["L1", "L2/3", "L4", "L5/6"]

# 像素转微米
PX_TO_UM = 0.1625

FONT_SIZES = {"title": 14, "label": 11, "tick": 9, "legend": 9, "annotation": 8}

# ── 颜色容差 ──
COLOR_TOLERANCE = 15


# ── 工具函数 ──


def _extract_color_mask(image: np.ndarray, target_color: tuple[int, int, int],
                        tolerance: int = COLOR_TOLERANCE) -> np.ndarray:
    """从 BGR 图像中提取指定颜色的二值掩膜。"""
    target = np.array(target_color, dtype=np.uint8)
    diff = np.abs(image.astype(np.int16) - target)
    return np.all(diff <= tolerance, axis=2).astype(np.uint8) * 255


def _load_image_pil(path: Path) -> np.ndarray:
    """用 PIL 加载任意大图像 (BGR 格式返回)。"""
    PILImage.MAX_IMAGE_PIXELS = None
    arr = np.array(PILImage.open(str(path)).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _detect_gt_colors(gt_mask: np.ndarray) -> dict[str, tuple[int, int, int]]:
    """自动检测 GT 图像中实际使用的颜色，映射到标准层名。

    根据每层颜色的平均 y 坐标从上到下排序: L1 → L2/3 → L4 → L5/6
    """
    # 1. 找出所有非黑色唯一颜色
    reshaped = gt_mask.reshape(-1, 3)
    # 只保留彩色像素 (排除黑色/背景)
    colored_mask_full = np.any(reshaped > 10, axis=1)
    colored_pixels = reshaped[colored_mask_full]

    if len(colored_pixels) == 0:
        raise ValueError("GT 图像中未找到任何彩色像素")

    # 用四舍五入归并相近颜色 (除以10取整)
    buckets: dict[tuple[int, ...], list[np.ndarray]] = {}
    for px in colored_pixels:
        key = tuple((px // 20).tolist())
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(px)

    # 取每个桶的平均色作为代表色 (最多取5种颜色)
    unique_colors = []
    for bucket_pixels in buckets.values():
        avg_color = np.mean(bucket_pixels, axis=0).astype(np.uint8)
        unique_colors.append(tuple(avg_color.tolist()))

    unique_colors.sort()
    # 经常会有(0,0,0)混入，重新过滤
    unique_colors = [c for c in unique_colors if not all(v < 20 for v in c)]

    # 2. 计算每层颜色的平均 y 坐标 (行号越大越深)
    color_y_centroids = []
    for color in unique_colors:
        mask = _extract_color_mask(gt_mask, color, tolerance=15)
        y_coords = np.where(mask > 0)[0]
        if len(y_coords) > 0:
            color_y_centroids.append((color, y_coords.mean()))

    # 按 y 坐标排序 (从上到下)
    color_y_centroids.sort(key=lambda x: x[1])

    # 应该正好有 4 层
    if len(color_y_centroids) != 4:
        print(f"  [警告] 检测到 {len(color_y_centroids)} 种颜色, 期望 4 层")
        print(f"  颜色: {color_y_centroids}")
        # 尝试用标准 GT_LAYER_COLORS 匹配
        matched = _try_match_standard_colors(gt_mask)
        if matched:
            return matched
        # 如果不够4层，按已有映射
        result = {}
        for i, (color, _) in enumerate(color_y_centroids):
            if i < len(STANDARD_LAYERS):
                result[STANDARD_LAYERS[i]] = color
            else:
                break
        # 补全缺失的层 (用标准颜色 + 容差)
        for layer in STANDARD_LAYERS:
            if layer not in result:
                result[layer] = GT_LAYER_COLORS[layer]
        return result

    result = {}
    for i, (color, _) in enumerate(color_y_centroids):
        if i < len(STANDARD_LAYERS):
            result[STANDARD_LAYERS[i]] = color
    return result


def _try_match_standard_colors(gt_mask: np.ndarray) -> dict[str, tuple[int, int, int]] | None:
    """尝试用标准 GT_LAYER_COLORS 和 ALT_GT_COLORS 匹配每一层。"""
    result = {}
    all_ok = True
    for layer in STANDARD_LAYERS:
        found = False
        candidate_colors = [GT_LAYER_COLORS[layer]] + ALT_GT_COLORS.get(layer, [])
        for color in candidate_colors:
            mask = _extract_color_mask(gt_mask, color, tolerance=15)
            if np.sum(mask > 0) > 100:  # 至少100个像素
                result[layer] = color
                found = True
                break
        if not found:
            all_ok = False
            break
    return result if all_ok else None


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return intersection / union if union > 0 else 0.0


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    sum_pixels = (mask1 > 0).sum() + (mask2 > 0).sum()
    return 2 * intersection / sum_pixels if sum_pixels > 0 else 0.0


def calculate_precision_recall(algo_mask: np.ndarray, gt_mask: np.ndarray):
    tp = np.logical_and(algo_mask > 0, gt_mask > 0).sum()
    fp = np.logical_and(algo_mask > 0, gt_mask == 0).sum()
    fn = np.logical_and(algo_mask == 0, gt_mask > 0).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def extract_layer_boundaries(layers_dict: dict[str, np.ndarray]):
    """从分层 mask 中提取层边界线。"""
    boundaries = {}
    # 获取尺寸
    valid_masks = [m for m in layers_dict.values() if m is not None]
    if not valid_masks:
        return boundaries
    h, w = valid_masks[0].shape[:2]

    for layer_name, mask in layers_dict.items():
        if mask is None:
            continue

        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        top_boundary = []
        bottom_boundary = []
        for x in range(w):
            col = gradient[:, x]
            y_coords = np.where(col > 0)[0]
            if len(y_coords) > 0:
                top_boundary.append((x, int(y_coords.min())))
                bottom_boundary.append((x, int(y_coords.max())))

        boundaries[layer_name] = {
            "top": np.array(top_boundary) if top_boundary else None,
            "bottom": np.array(bottom_boundary) if bottom_boundary else None,
        }

    return boundaries


def calculate_boundary_distance(boundary1: np.ndarray | None,
                                boundary2: np.ndarray | None) -> float | None:
    if boundary1 is None or boundary2 is None or len(boundary1) == 0 or len(boundary2) == 0:
        return None
    b1_dict = {p[0]: p[1] for p in boundary1}
    b2_dict = {p[0]: p[1] for p in boundary2}
    common_x = set(b1_dict.keys()) & set(b2_dict.keys())
    if not common_x:
        return None
    distances = [abs(b1_dict[x] - b2_dict[x]) for x in common_x]
    return float(np.mean(distances))


def calculate_thickness(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape[:2]
    thicknesses = []
    for x in range(w):
        col = mask[:, x]
        y_coords = np.where(col > 0)[0]
        if len(y_coords) > 0:
            thickness = int(y_coords.max() - y_coords.min() + 1)
            thicknesses.append(thickness)
    return np.array(thicknesses) if thicknesses else None


# ── 核心分析函数 ──


def analyze_sample(sample: str) -> pd.DataFrame | None:
    """分析单个样本，返回指标 DataFrame。"""
    sample_name = sample
    out_dir = OUTPUTS_DIR / sample
    label_dir = LABEL_DIR / sample
    analysis_sample_dir = ANALYSIS_DIR / sample
    analysis_sample_dir.mkdir(parents=True, exist_ok=True)

    # ── 验证文件存在 ──
    algo_path = out_dir / "layers_color_mask.png"
    gt_path = label_dir / "label_mask.png"

    if not algo_path.exists():
        print(f"  [跳过] 未找到算法输出: {algo_path}")
        return None
    if not gt_path.exists():
        print(f"  [跳过] 未找到 GT: {gt_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"分析样本: {sample_name}")
    print(f"{'=' * 60}")

    # ── 加载图像 ──
    print("  加载图像...")
    algo_mask = _load_image_pil(algo_path)
    gt_mask = _load_image_pil(gt_path)

    print(f"  算法输出尺寸: {algo_mask.shape}")
    print(f"  GT 尺寸: {gt_mask.shape}")

    # 调整尺寸一致
    if algo_mask.shape != gt_mask.shape:
        print(f"  警告: 尺寸不一致, 调整 GT 尺寸")
        gt_mask = cv2.resize(gt_mask, (algo_mask.shape[1], algo_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # ── 解析颜色 ──
    print("  解析颜色映射...")

    # 算法层
    algo_layers: dict[str, np.ndarray] = {}
    for layer_name, color_bgr in ALGO_LAYER_COLORS.items():
        mask = _extract_color_mask(algo_mask, color_bgr, COLOR_TOLERANCE)
        pixel_count = np.sum(mask > 0)
        algo_layers[layer_name] = mask
        print(f"    算法 {layer_name}: BGR{color_bgr} → {pixel_count} 像素")

    # GT 层 - 自动检测颜色
    gt_layer_colors = _detect_gt_colors(gt_mask)
    gt_layers: dict[str, np.ndarray] = {}
    print(f"    检测到 GT 颜色: {gt_layer_colors}")
    for layer_name, color_bgr in gt_layer_colors.items():
        mask = _extract_color_mask(gt_mask, color_bgr, COLOR_TOLERANCE)
        pixel_count = np.sum(mask > 0)
        gt_layers[layer_name] = mask
        print(f"    GT {layer_name}: BGR{color_bgr} → {pixel_count} 像素")

    # ── 计算重叠指标 ──
    print("  计算重叠指标...")
    overlap_results = []
    for layer_name in STANDARD_LAYERS:
        a_mask = algo_layers.get(layer_name)
        g_mask = gt_layers.get(layer_name)

        if a_mask is None:
            overlap_results.append({
                "layer": layer_name, "iou": None, "dice": None,
                "precision": None, "recall": None, "status": "algo_missing"
            })
            continue
        if g_mask is None:
            overlap_results.append({
                "layer": layer_name, "iou": None, "dice": None,
                "precision": None, "recall": None, "status": "gt_missing"
            })
            continue

        iou = calculate_iou(a_mask, g_mask)
        dice = calculate_dice(a_mask, g_mask)
        precision, recall = calculate_precision_recall(a_mask, g_mask)

        overlap_results.append({
            "layer": layer_name, "iou": iou, "dice": dice,
            "precision": precision, "recall": recall, "status": "ok"
        })
        print(f"    {layer_name}: IoU={iou:.4f}, Dice={dice:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}")

    # ── 计算边界误差 ──
    print("  计算边界误差...")
    algo_boundaries = extract_layer_boundaries(algo_layers)
    gt_boundaries = extract_layer_boundaries(gt_layers)

    boundary_results = []
    for layer_name in STANDARD_LAYERS:
        a_b = algo_boundaries.get(layer_name, {})
        g_b = gt_boundaries.get(layer_name, {})

        top_dist = calculate_boundary_distance(a_b.get("top"), g_b.get("top"))
        bottom_dist = calculate_boundary_distance(a_b.get("bottom"), g_b.get("bottom"))

        avg_dist = None
        if top_dist is not None and bottom_dist is not None:
            avg_dist = (top_dist + bottom_dist) / 2
        elif top_dist is not None:
            avg_dist = top_dist
        elif bottom_dist is not None:
            avg_dist = bottom_dist

        boundary_results.append({
            "layer": layer_name,
            "top_boundary_error_px": top_dist,
            "bottom_boundary_error_px": bottom_dist,
            "avg_boundary_error_px": avg_dist,
        })
        if avg_dist is not None:
            print(f"    {layer_name}: 边界误差={avg_dist:.2f}px")

    # ── 计算厚度误差 ──
    print("  计算厚度误差...")
    thickness_results = []
    for layer_name in STANDARD_LAYERS:
        a_mask = algo_layers.get(layer_name)
        g_mask = gt_layers.get(layer_name)

        if a_mask is None or g_mask is None:
            thickness_results.append({
                "layer": layer_name,
                "algo_mean_thickness_px": None,
                "gt_mean_thickness_px": None,
                "thickness_error_px": None,
                "thickness_error_percent": None,
            })
            continue

        algo_t = calculate_thickness(a_mask)
        gt_t = calculate_thickness(g_mask)

        if algo_t is None or gt_t is None or len(algo_t) == 0 or len(gt_t) == 0:
            continue

        algo_mean = float(np.mean(algo_t))
        gt_mean = float(np.mean(gt_t))
        error = abs(algo_mean - gt_mean)
        error_pct = (error / gt_mean * 100) if gt_mean > 0 else 0.0

        thickness_results.append({
            "layer": layer_name,
            "algo_mean_thickness_px": round(algo_mean, 2),
            "gt_mean_thickness_px": round(gt_mean, 2),
            "thickness_error_px": round(error, 2),
            "thickness_error_percent": round(error_pct, 2),
        })
        print(f"    {layer_name}: 算法厚度={algo_mean:.1f}px, GT厚度={gt_mean:.1f}px, "
              f"误差={error:.1f}px ({error_pct:.1f}%)")

    # ── 合并结果 ──
    overlap_df = pd.DataFrame(overlap_results)
    boundary_df = pd.DataFrame(boundary_results)
    thickness_df = pd.DataFrame(thickness_results)

    results_df = overlap_df.merge(boundary_df, on="layer", how="outer")
    results_df = results_df.merge(thickness_df, on="layer", how="outer")

    # 添加微米制单位
    for col in ["top_boundary_error_px", "bottom_boundary_error_px", "avg_boundary_error_px",
                "algo_mean_thickness_px", "gt_mean_thickness_px", "thickness_error_px"]:
        if col in results_df.columns:
            um_col = col.replace("_px", "_um")
            if results_df[col].notna().any():
                results_df[um_col] = results_df[col].apply(
                    lambda x: round(x * PX_TO_UM, 3) if pd.notna(x) else None)

    # 添加样本信息列
    results_df.insert(0, "sample", sample_name)

    # 保存
    csv_path = analysis_sample_dir / "analysis_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  结果已保存: {csv_path}")

    # ── 可视化对比 ──
    _visualize_comparison(algo_mask, gt_mask, algo_layers, gt_layers,
                          results_df, analysis_sample_dir, sample_name)

    return results_df


def _visualize_comparison(algo_mask: np.ndarray, gt_mask: np.ndarray,
                          algo_layers: dict[str, np.ndarray],
                          gt_layers: dict[str, np.ndarray],
                          metrics_df: pd.DataFrame,
                          save_dir: Path, sample_name: str):
    """生成样本级别的对比可视化。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 算法输出
    axes[0, 0].imshow(cv2.cvtColor(algo_mask, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Algorithm Output", fontsize=FONT_SIZES["title"])
    axes[0, 0].axis("off")

    # Ground Truth
    axes[0, 1].imshow(cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Ground Truth", fontsize=FONT_SIZES["title"])
    axes[0, 1].axis("off")

    # 差异图
    diff = cv2.absdiff(algo_mask, gt_mask)
    axes[0, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Difference", fontsize=FONT_SIZES["title"])
    axes[0, 2].axis("off")

    # 算法各层叠加 (RGB)
    h, w = algo_mask.shape[:2]
    colors_rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    algo_overlay = np.zeros((h, w, 3), dtype=np.float32)
    for i, layer_name in enumerate(STANDARD_LAYERS):
        if layer_name in algo_layers:
            mask = algo_layers[layer_name] > 0
            for c in range(3):
                algo_overlay[:, :, c][mask] = colors_rgb[i % len(colors_rgb)][c]
    axes[1, 0].imshow(algo_overlay)
    axes[1, 0].set_title("Algorithm Layers (R=L1, G=L2/3, B=L4, Y=L5/6)",
                         fontsize=FONT_SIZES["title"])
    axes[1, 0].axis("off")

    # GT各层叠加
    gt_overlay = np.zeros((h, w, 3), dtype=np.float32)
    for i, layer_name in enumerate(STANDARD_LAYERS):
        if layer_name in gt_layers:
            mask = gt_layers[layer_name] > 0
            for c in range(3):
                gt_overlay[:, :, c][mask] = colors_rgb[i % len(colors_rgb)][c]
    axes[1, 1].imshow(gt_overlay)
    axes[1, 1].set_title("GT Layers (R=L1, G=L2/3, B=L4, Y=L5/6)",
                         fontsize=FONT_SIZES["title"])
    axes[1, 1].axis("off")

    # 重叠区域
    overlap = np.zeros((h, w, 3), dtype=np.float32)
    for i, layer_name in enumerate(STANDARD_LAYERS):
        if layer_name in algo_layers and layer_name in gt_layers:
            a_m = algo_layers[layer_name] > 0
            g_m = gt_layers[layer_name] > 0
            overlap_m = np.logical_and(a_m, g_m)
            algo_only = np.logical_and(a_m, ~g_m)
            gt_only = np.logical_and(~a_m, g_m)
            val = 0.5 + i * 0.1
            overlap[:, :, 1][overlap_m] = val
            overlap[:, :, 0][algo_only] = val
            overlap[:, :, 2][gt_only] = val
    axes[1, 2].imshow(overlap)
    axes[1, 2].set_title("Overlap (G=match, R=algo only, B=GT only)",
                         fontsize=FONT_SIZES["title"])
    axes[1, 2].axis("off")

    plt.suptitle(f"Sample {sample_name} — Layer Segmentation Comparison",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = save_dir / "comparison.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════════
#  汇总统计与可视化
# ══════════════════════════════════════════════════════════════════════════════════


def compute_summary_stats(all_results: pd.DataFrame) -> pd.DataFrame:
    """计算所有样本的汇总统计量。"""
    metric_cols = ["iou", "dice", "precision", "recall",
                   "avg_boundary_error_px", "thickness_error_percent"]

    summary_rows = []
    for layer in STANDARD_LAYERS:
        layer_data = all_results[all_results["layer"] == layer]
        row = {"layer": layer}
        for col in metric_cols:
            if col in layer_data.columns:
                vals = layer_data[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = vals.mean()
                    row[f"{col}_std"] = vals.std()
                    row[f"{col}_min"] = vals.min()
                    row[f"{col}_q25"] = vals.quantile(0.25)
                    row[f"{col}_median"] = vals.median()
                    row[f"{col}_q75"] = vals.quantile(0.75)
                    row[f"{col}_max"] = vals.max()
                    row[f"{col}_cv"] = vals.std() / vals.mean() if vals.mean() > 0 else None
        summary_rows.append(row)

    # 总体平均
    overall = {"layer": "OVERALL"}
    for col in metric_cols:
        if col in all_results.columns:
            vals = all_results[col].dropna()
            if len(vals) > 0:
                overall[f"{col}_mean"] = vals.mean()
                overall[f"{col}_std"] = vals.std()
                overall[f"{col}_min"] = vals.min()
                overall[f"{col}_q25"] = vals.quantile(0.25)
                overall[f"{col}_median"] = vals.median()
                overall[f"{col}_q75"] = vals.quantile(0.75)
                overall[f"{col}_max"] = vals.max()
    summary_rows.append(overall)

    return pd.DataFrame(summary_rows)


def _save_summary_plot(fig: plt.Figure, filename: str):
    """保存汇总图到 figures 目录。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {path}")


def plot_layer_boxplots(all_results: pd.DataFrame):
    """各层指标箱线图。"""
    metrics = [
        ("iou", "IoU", (0, 1)),
        ("dice", "Dice", (0, 1)),
        ("precision", "Precision", (0, 1)),
        ("recall", "Recall", (0, 1)),
        ("avg_boundary_error_px", "Avg Boundary Error (px)", None),
        ("thickness_error_percent", "Thickness Error (%)", None),
    ]

    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, (col, label, ylim) in enumerate(metrics):
        ax = axes[idx]
        data = []
        labels = []
        for layer in STANDARD_LAYERS:
            vals = all_results[all_results["layer"] == layer][col].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                labels.append(layer)
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(data)))):
                patch.set_facecolor(color)
            ax.set_title(label, fontsize=FONT_SIZES["title"])
            ax.tick_params(labelsize=FONT_SIZES["tick"])
            if ylim:
                ax.set_ylim(ylim)
        else:
            ax.set_title(f"{label} (no data)", fontsize=FONT_SIZES["title"])

    # 隐藏多余的子图
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-Layer Metric Distribution (all samples)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_summary_plot(fig, "boxplots_metrics.png")


def plot_sample_barplots(all_results: pd.DataFrame):
    """样本维度柱状图 (每样本各层平均 Dice 和 IoU)。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric in enumerate(["dice", "iou"]):
        ax = axes[ax_idx]
        pivot = all_results.pivot_table(index="sample", columns="layer",
                                        values=metric, aggfunc="mean")
        if pivot.empty:
            ax.set_title(f"{metric.upper()} (no data)")
            continue

        x = np.arange(len(pivot))
        n_layers = len(pivot.columns)
        bar_width = 0.8 / n_layers

        for i, layer in enumerate(pivot.columns):
            bars = ax.bar(x + i * bar_width, pivot[layer].values,
                          width=bar_width, label=layer)
            # 在柱子上标注数值
            for bar, val in zip(bars, pivot[layer].values):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=FONT_SIZES["annotation"], rotation=45)

        ax.set_xlabel("Sample", fontsize=FONT_SIZES["label"])
        ax.set_ylabel(metric.upper(), fontsize=FONT_SIZES["label"])
        ax.set_title(f"{metric.upper()} by Sample and Layer",
                     fontsize=FONT_SIZES["title"])
        ax.set_xticks(x + bar_width * (n_layers - 1) / 2)
        ax.set_xticklabels(pivot.index, fontsize=FONT_SIZES["tick"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.set_ylim(0, 1.1)

    fig.tight_layout()
    _save_summary_plot(fig, "barplots_dice_iou.png")


def plot_heatmaps(all_results: pd.DataFrame):
    """热力图: 样本 × 层 的各项指标。"""
    metrics = ["dice", "iou", "avg_boundary_error_px", "thickness_error_percent"]
    titles = ["Dice", "IoU", "Avg Boundary Error (px)", "Thickness Error (%)"]

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        pivot = all_results.pivot_table(index="sample", columns="layer",
                                        values=metric, aggfunc="mean")
        if pivot.empty:
            ax.set_title(f"{title} (no data)")
            continue
        im = ax.imshow(pivot.values, cmap="YlOrRd" if metric in ("dice", "iou") else "YlGnBu",
                       aspect="auto")
        # 标注数值
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    color = "white" if (metric in ("dice", "iou") and val < 0.5) else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=FONT_SIZES["annotation"], color=color)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=FONT_SIZES["tick"])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=FONT_SIZES["tick"])
        ax.set_title(title, fontsize=FONT_SIZES["title"])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    _save_summary_plot(fig, "heatmaps.png")


def plot_radar(all_results: pd.DataFrame):
    """按层绘制雷达图: 平均 Dice / IoU / Precision / Recall。"""
    metrics = ["dice", "iou", "precision", "recall"]
    metric_labels = ["Dice", "IoU", "Precision", "Recall"]

    n_layers = len(STANDARD_LAYERS)
    n_cols = 2
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "polar"},
                             figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, layer in enumerate(STANDARD_LAYERS):
        ax = axes[idx]
        layer_data = all_results[all_results["layer"] == layer]
        values = []
        for m in metrics:
            v = layer_data[m].dropna().mean() if m in layer_data.columns else 0
            values.append(v)

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=layer)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=FONT_SIZES["tick"])
        ax.set_ylim(0, 1)
        ax.set_title(layer, fontsize=FONT_SIZES["title"], fontweight="bold",
                     pad=20)
        ax.grid(True)

    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Segmentation Quality Radar by Layer",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_summary_plot(fig, "radar_quality.png")


def plot_boundary_error(all_results: pd.DataFrame):
    """边界误差箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    for layer in STANDARD_LAYERS:
        vals = all_results[all_results["layer"] == layer]["avg_boundary_error_px"].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(layer)

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title("Boundary Error by Layer (all samples)",
                     fontsize=FONT_SIZES["title"])
        ax.set_ylabel("Avg Boundary Error (px)", fontsize=FONT_SIZES["label"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_title("Boundary Error (no data)")

    fig.tight_layout()
    _save_summary_plot(fig, "boundary_error.png")


def plot_thickness_comparison(all_results: pd.DataFrame):
    """厚度对比: 算法 vs GT 散点图 + 厚度误差%。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 算法 vs GT 厚度散点
    ax = axes[0]
    markers = ["o", "s", "^", "D"]
    for i, layer in enumerate(STANDARD_LAYERS):
        ld = all_results[all_results["layer"] == layer]
        ax.scatter(ld["gt_mean_thickness_px"], ld["algo_mean_thickness_px"],
                   label=layer, marker=markers[i % len(markers)], alpha=0.7, s=50)

    max_val = max(
        all_results["algo_mean_thickness_px"].max(),
        all_results["gt_mean_thickness_px"].max()
    )
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect match")
    ax.set_xlabel("GT Thickness (px)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Algo Thickness (px)", fontsize=FONT_SIZES["label"])
    ax.set_title("Thickness: Algo vs GT", fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # 右: 厚度误差%箱线图
    ax = axes[1]
    data = []
    labels = []
    for layer in STANDARD_LAYERS:
        vals = all_results[all_results["layer"] == layer]["thickness_error_percent"].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(layer)
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title("Thickness Error %", fontsize=FONT_SIZES["title"])
        ax.set_ylabel("Error (%)", fontsize=FONT_SIZES["label"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_summary_plot(fig, "thickness_comparison.png")


def plot_metrics_histograms(all_results: pd.DataFrame):
    """指标分布直方图。"""
    metrics = ["dice", "iou", "precision", "recall"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for layer in STANDARD_LAYERS:
            vals = all_results[all_results["layer"] == layer][metric].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=8, alpha=0.5, label=layer)
        ax.set_xlabel(metric.upper(), fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Count", fontsize=FONT_SIZES["label"])
        ax.set_title(f"{metric.upper()} Distribution", fontsize=FONT_SIZES["title"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])

    fig.tight_layout()
    _save_summary_plot(fig, "metrics_histograms.png")


def plot_cv_comparison(all_results: pd.DataFrame):
    """变异系数 (CV) 对比。"""
    metrics = ["iou", "dice", "precision", "recall"]
    metric_labels = ["IoU", "Dice", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(metrics))
    width = 0.15

    for i, layer in enumerate(STANDARD_LAYERS):
        layer_data = all_results[all_results["layer"] == layer]
        cv_vals = []
        for m in metrics:
            vals = layer_data[m].dropna()
            cv = vals.std() / vals.mean() if vals.mean() > 0 else 0
            cv_vals.append(cv)
        ax.bar(x + i * width, cv_vals, width, label=layer)

    ax.set_xticks(x + width * (len(STANDARD_LAYERS) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=FONT_SIZES["tick"])
    ax.set_ylabel("CV (Coefficient of Variation)", fontsize=FONT_SIZES["label"])
    ax.set_title("Metric Variability (CV) by Layer",
                 fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_summary_plot(fig, "cv_comparison.png")


def plot_error_correlation(all_results: pd.DataFrame):
    """边界误差 vs 厚度误差散点图。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D"]
    for i, layer in enumerate(STANDARD_LAYERS):
        ld = all_results[all_results["layer"] == layer].dropna(
            subset=["avg_boundary_error_px", "thickness_error_percent"])
        if len(ld) > 0:
            ax.scatter(ld["avg_boundary_error_px"], ld["thickness_error_percent"],
                       label=layer, marker=markers[i % len(markers)], alpha=0.7, s=60)

    ax.set_xlabel("Avg Boundary Error (px)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Thickness Error (%)", fontsize=FONT_SIZES["label"])
    ax.set_title("Error Correlation: Boundary vs Thickness",
                 fontsize=FONT_SIZES["title"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_summary_plot(fig, "error_correlation.png")


def generate_summary_table(all_results: pd.DataFrame):
    """生成汇总统计表并保存。"""
    summary = compute_summary_stats(all_results)
    csv_path = SUMMARY_DIR / "stats_summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  汇总统计已保存: {csv_path}")
    print(summary.to_string(index=False))
    return summary


def run_summary_analysis(all_results: pd.DataFrame):
    """运行完整汇总分析和可视化。"""
    print(f"\n{'=' * 60}")
    print(f"汇总分析: {len(all_results)} 行数据")
    print(f"{'=' * 60}")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 生成汇总统计
    summary = generate_summary_table(all_results)

    # 保存聚合数据 (所有样本所有层的指标)
    agg_path = SUMMARY_DIR / "all_samples_aggregated.csv"
    all_results.to_csv(agg_path, index=False, encoding="utf-8-sig")
    print(f"  聚合数据已保存: {agg_path}")

    # 绘图
    print("\n  生成可视化...")
    plot_layer_boxplots(all_results)
    plot_sample_barplots(all_results)
    plot_heatmaps(all_results)
    plot_radar(all_results)
    plot_boundary_error(all_results)
    plot_thickness_comparison(all_results)
    plot_metrics_histograms(all_results)
    plot_cv_comparison(all_results)
    plot_error_correlation(all_results)

    print(f"\n所有可视化已保存到: {FIGURES_DIR}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("自动分层结果分析")
    print("=" * 60)
    print(f"算法输出: {OUTPUTS_DIR}")
    print(f"GT标签:   {LABEL_DIR}")
    print(f"分析输出: {ANALYSIS_DIR}")
    print()

    # ── 获取所有样本 ──
    samples = sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()])
    print(f"发现 {len(samples)} 个样本: {samples}")

    # ── 逐样本分析 ──
    all_results = []
    for sample in samples:
        result = analyze_sample(sample)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n没有成功分析任何样本！")
        return

    # ── 合并所有结果 ──
    combined = pd.concat(all_results, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"所有样本分析完成! 共 {len(combined)} 行指标数据")
    print(f"{'=' * 60}")

    # ── 汇总分析 ──
    run_summary_analysis(combined)

    print(f"\n{'=' * 60}")
    print("分析完成!")
    print(f"  样本结果: {ANALYSIS_DIR}/<sample>/analysis_results.csv")
    print(f"  汇总统计: {SUMMARY_DIR}/stats_summary.csv")
    print(f"  汇总图表: {FIGURES_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
