#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密度梯度引导的逐列边界精炼模块。

对 dataset/outputs/<sample>/all_boundaries.csv 中的内部边界 (L1_2, L3_4, L4_5)，
利用局部细胞密度剖面进行逐列调整，使边界线更贴合局部组织特征。

用法:
  # 对所有样本运行精炼
  .\\venv-cellpose\\Scripts\\python src/refine_boundaries.py

  # 指定样本
  .\\venv-cellpose\\Scripts\\python src/refine_boundaries.py --samples 1 2 3

  # 自定义参数
  .\\venv-cellpose\\Scripts\\python src/refine_boundaries.py --cell-window 7 --smooth-sigma 2 --median-window 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from PIL import Image as PILImage

# ── 路径 ──
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "dataset" / "outputs"
LABEL_DIR = ROOT / "dataset" / "inputs" / "label"
ANALYSIS_DIR = ROOT / "dataset" / "analysis"

# ── 默认参数 ──
DEFAULT_CELL_WINDOW = 30      # 提取细胞的水平窗口半宽(紧凑像素, ≈300px 全分辨率)
DEFAULT_N_BINS = 50           # depth 分箱数
DEFAULT_SMOOTH_SIGMA = 2.5    # 密度剖面高斯平滑 sigma
DEFAULT_SEARCH_WINDOW = 0.12  # 边界搜索窗半宽(depth)
DEFAULT_MEDIAN_WINDOW = 9     # 列间平滑中值滤波窗口
DEFAULT_COLUMN_STEP = 200     # 精炼列间距(像素), 1=逐列, 200=每200px一列

# GT 颜色映射 (BGR) —— 用于从 label_mask.png 提取 GT 边界线
# 注意: label_mask.png 和 layers_color_mask.png 可能使用相同颜色方案
GT_COLOR_MAP_PRIORITY = [
    # 第一候选: (与算法输出相同的颜色)
    {   (255, 100, 100): 0,   # L1
        (100, 255, 100): 1,   # L2/3
        (100, 100, 255): 2,   # L4
        (255, 255, 100): 3,   # L5/6
    },
    # 第二候选: (resultanalysis.py 中定义的 GT 颜色)
    {   (255, 100, 100): 0,   # L1
        (100, 255, 100): 1,   # L2/3
        (100, 100, 255): 2,   # L4
        (255, 100, 255): 3,   # L5/6
    },
    # 第三候选: (旧版 GT 颜色)
    {   (25, 28, 252): 0,     # L1
        (18, 126, 126): 1,    # L2/3
        (255, 255, 76): 2,    # L4
        (127, 127, 248): 3,   # L5/6
    },
]
COLOR_TOL = 15

# ── 工具函数 ──


def _build_y_lookup(df_boundary: pd.DataFrame):
    """从 boundary DataFrame (x,y列) 构建 x→y 插值函数。"""
    pts = df_boundary[["x", "y"]].to_numpy(dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.nan
    d = pd.DataFrame(pts, columns=["x", "y"])
    g = d.groupby("x", as_index=False)["y"].median().sort_values("x")
    xv = g["x"].to_numpy(dtype=float)
    yv = g["y"].to_numpy(dtype=float)
    if len(xv) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), yv[0], dtype=float)

    def _lut(x_new):
        xa = np.asarray(x_new, dtype=float)
        return np.interp(np.clip(xa, xv[0], xv[-1]), xv, yv)
    return _lut


def _load_boundaries(abc_path: Path) -> dict[str, pd.DataFrame]:
    """加载 all_boundaries.csv 并按 boundary 分组。"""
    if not abc_path.exists():
        raise FileNotFoundError(f"边界文件不存在: {abc_path}")
    df = pd.read_csv(abc_path)
    groups = {}
    for name, sub in df.groupby("boundary"):
        groups[name] = sub[["x", "y"]].reset_index(drop=True)
    return groups


def _load_cells(csv_path: Path) -> pd.DataFrame:
    """加载细胞质心 CSV，统一列名为 X, Y。"""
    df = pd.read_csv(csv_path)
    col_map = {}
    for src in ["centroid_x", "Centroid_x", "centroidX"]:
        if src in df.columns and "X" not in col_map:
            col_map[src] = "X"
    for src in ["centroid_y", "Centroid_y", "centroidY"]:
        if src in df.columns and "Y" not in col_map:
            col_map[src] = "Y"
    for src in ["x", "X"]:
        if src in df.columns and "X" not in col_map:
            col_map[src] = "X"
    for src in ["y", "Y"]:
        if src in df.columns and "Y" not in col_map:
            col_map[src] = "Y"
    if col_map:
        df = df.rename(columns=col_map)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"细胞数据缺少坐标列: {list(df.columns)}")
    return df[["X", "Y"]].copy()


def _load_image_pil(path: Path) -> np.ndarray:
    """用 PIL 加载大图像 (返回 BGR uint8)。"""
    PILImage.MAX_IMAGE_PIXELS = None
    arr = np.array(PILImage.open(str(path)).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _extract_gt_boundaries_from_mask(gt_mask: np.ndarray) -> dict[str, np.ndarray]:
    """从 GT label_mask 提取 5 条边界线 (pia, L1_2, L3_4, L4_5, white)。

    自动尝试多个颜色映射候选，选覆盖最多像素的方案。
    返回 {boundary_name: (N,2) ndarray [x, y]}。
    """
    h, w = gt_mask.shape[:2]
    best_result = None
    best_count = 0

    for color_map in GT_COLOR_MAP_PRIORITY:
        layer_map = np.full((h, w), -1, dtype=np.int32)
        matched = np.zeros((h, w), dtype=bool)
        for (b, g, r), idx in color_map.items():
            mask = (
                (np.abs(gt_mask[:, :, 0].astype(np.int32) - b) <= COLOR_TOL) &
                (np.abs(gt_mask[:, :, 1].astype(np.int32) - g) <= COLOR_TOL) &
                (np.abs(gt_mask[:, :, 2].astype(np.int32) - r) <= COLOR_TOL)
            )
            layer_map[mask] = idx
            matched[mask] = True

        n_matched = int(matched.sum())
        if n_matched > best_count:
            best_count = n_matched
            best_map = layer_map.copy()
        if n_matched > h * w * 0.5:  # 覆盖超过 50% 即可接受
            break

    if best_count < h * w * 0.1:
        print(f"  [警告] GT 颜色匹配不良: 仅 {best_count}/{h * w} 像素匹配")
        return {k: np.empty((0, 2)) for k in ["pia", "L1_2", "L3_4", "L4_5", "white"]}

    layer_map = best_map

    # 对每列 x 扫描找出层变化点
    boundaries: dict[str, list] = {"pia": [], "L1_2": [], "L3_4": [], "L4_5": [], "white": []}
    for x in range(w):
        col = layer_map[:, x]
        valid = col >= 0
        if valid.sum() < 10:
            continue

        # pia = 最浅有效像素 y
        # white = 最深有效像素 y
        valid_ys = np.where(valid)[0]
        y_start = int(valid_ys[0])
        y_end = int(valid_ys[-1])

        # 层过渡点 (深度方向层索引变化)
        diffs = np.diff(col.astype(int))
        trans = np.where((diffs != 0) & (col[:-1] >= 0) & (col[1:] >= 0))[0]
        if len(trans) < 3:
            continue

        boundaries["pia"].append((float(x), float(y_start)))
        boundaries["white"].append((float(x), float(y_end)))
        boundaries["L1_2"].append((float(x), float(trans[0])))
        boundaries["L3_4"].append((float(x), float(trans[1])))
        boundaries["L4_5"].append((float(x), float(trans[2])))

    result = {}
    for name in boundaries:
        pts = np.array(boundaries[name])
        if len(pts) > 0:
            result[name] = pts
        else:
            result[name] = np.empty((0, 2))

    n_total = sum(len(v) for v in result.values())
    if n_total == 0:
        print(f"  [警告] GT 边界提取失败: 未找到有效列")
    return result


# ── 局部密度剖面 ──


def _col_local_cells(x: int, cells: pd.DataFrame,
                      pia_y: float, white_y: float,
                      window: int) -> np.ndarray | None:
    """获取列 x 附近细胞的 depth 值数组。

    返回 (N,) depth 数组，或 None (数据不足)。
    """
    if not (np.isfinite(pia_y) and np.isfinite(white_y) and abs(white_y - pia_y) > 10):
        return None
    nearby = cells[(cells["X"] >= x - window) & (cells["X"] <= x + window)]
    if len(nearby) < 10:
        return None
    depths = (nearby["Y"].values - pia_y) / (white_y - pia_y + 1e-8)
    depths = np.clip(depths, 0.0, 1.0)
    return depths


def _compute_cdf_percentiles_at_seeds(all_cell_depths: np.ndarray,
                                       global_seeds: dict[str, float]) -> dict[str, float]:
    """从全体细胞的 depth 分布计算每个全局种子 depth 处对应的 CDF 百分位值。

    这些百分位反映了"在全组织上，L1_2 边界以上有 x% 细胞"等比例信息。
    """
    percentiles = {}
    for name in ["L1_2", "L3_4", "L4_5"]:
        seed = global_seeds[name]
        pct = float(np.mean(all_cell_depths <= seed) * 100)
        percentiles[name] = max(1.0, min(99.0, pct))
    return percentiles


def _local_cdf_boundaries(cell_depths: np.ndarray,
                           global_seeds: dict[str, float],
                           target_percentiles: dict[str, float],
                           max_delta: float = 0.05) -> dict[str, float]:
    """基于局部 CDF 的边界估计。

    对每列，用全体细胞的 CDF 百分位作为"模板"，找到局部 CDF 中相同百分位
    对应的 depth。

    返回: {"L1_2": depth, "L3_4": depth, "L4_5": depth}
    """
    if len(cell_depths) < 10:
        return dict(global_seeds)

    sorted_depths = np.sort(cell_depths)
    cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)

    result = {}
    for name in ["L1_2", "L3_4", "L4_5"]:
        target_pct = target_percentiles[name]
        target_cdf = target_pct / 100.0

        # 在局部 CDF 中找对应的 depth
        idx = np.searchsorted(cdf, target_cdf)
        idx = np.clip(idx, 0, len(sorted_depths) - 1)
        local_depth = float(sorted_depths[idx])

        # 约束不超过 max_delta
        seed = global_seeds[name]
        clamped = np.clip(local_depth, seed - max_delta, seed + max_delta)
        result[name] = float(clamped)

    # 单调性
    b12 = max(0.02, min(result["L1_2"], result["L3_4"] - 0.01))
    b34 = max(b12 + 0.01, min(result["L3_4"], result["L4_5"] - 0.01))
    b456 = max(b34 + 0.01, min(result["L4_5"], 0.98))

    return {"L1_2": b12, "L3_4": b34, "L4_5": b456}


# ── 列间平滑 ──


def _smooth_boundary_line(xs: np.ndarray, ys: np.ndarray,
                           window: int) -> np.ndarray:
    """对边界线的 y 坐标应用中值滤波。"""
    if len(ys) < window:
        return ys
    # 确保 xs 连续，插值填补空缺
    xs_int = np.arange(int(xs.min()), int(xs.max()) + 1)
    y_interp = np.interp(xs_int, xs, ys)
    y_smooth = median_filter(y_interp, size=window)
    # 只返回有原始 x 的点
    idxs = np.searchsorted(xs_int, xs.astype(int))
    idxs = np.clip(idxs, 0, len(y_smooth) - 1)
    return y_smooth[idxs]


# ── 边界误差计算 ──


def _boundary_mae(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """计算两条边界线的平均绝对误差 (像素)。

    对 pred 中每个 x，插值 gt 的 y，计算 |y_pred - y_gt_interp|。
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    pred_df = pd.DataFrame(pred_pts, columns=["x", "y"])
    gt_df = pd.DataFrame(gt_pts, columns=["x", "y"])

    # 对 pred 的每个 x，找 gt 中最近的 y
    pred_x = pred_df.groupby("x")["y"].median().reset_index()
    gt_lookup = _build_y_lookup(gt_df[["x", "y"]])

    errors = []
    for _, row in pred_x.iterrows():
        xp, yp = float(row["x"]), float(row["y"])
        yg = gt_lookup(xp)
        if np.isfinite(yg):
            errors.append(abs(yp - yg))
    return float(np.mean(errors)) if errors else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  主精炼流程
# ══════════════════════════════════════════════════════════════════════════════


class BoundaryRefiner:
    """基于局部密度的逐列边界精炼器。"""

    def __init__(self, cell_window=DEFAULT_CELL_WINDOW,
                 n_bins=DEFAULT_N_BINS,
                 smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                 search_window=DEFAULT_SEARCH_WINDOW,
                 median_window=DEFAULT_MEDIAN_WINDOW,
                 column_step=DEFAULT_COLUMN_STEP):
        self.cell_window = cell_window
        self.n_bins = n_bins
        self.smooth_sigma = smooth_sigma
        self.search_window = search_window
        self.search_window_l12 = search_window
        self.median_window = median_window
        self.column_step = column_step

    def _estimate_compact_scale(self, out_dir: Path, cells: pd.DataFrame) -> float:
        """估算紧凑缩放比例。

        all_boundaries.csv 中 pia/white 为全分辨率坐标，但内部边界
        (L1_2, L3_4, L4_5) 和算法输出图像为紧凑分辨率 (compact_rate ≈ 0.1)。
        此方法估算 scale = compact_height / full_height。
        """
        algo_img = cv2.imread(str(out_dir / "layers_color_mask.png"))
        if algo_img is None:
            return 0.1  # 默认
        compact_h = algo_img.shape[0]
        # 用细胞 Y 最大值估计全分辨率高度
        full_h = float(cells["Y"].max())
        if full_h > 0:
            return compact_h / full_h
        return 0.1

    def refine(self, sample: str) -> dict:
        """对单个样本运行逐列精炼。

        工作流程:
          1. 加载全分辨率数据
          2. 转换到紧凑分辨率 (cells, pia/white 全部缩放)
          3. 在紧凑分辨率下逐列计算局部密度剖面并搜索边界
          4. 输出紧凑分辨率的精炼边界

        返回:
            {
                "sample": sample,
                "global_seeds": {name: depth},
                "compact_scale": float,
                "bounds_compact": {boundary: (N,2) ndarray},
                "mae_before": {boundary: float},
                "mae_after": {boundary: float},
            }
        """
        out_dir = OUTPUTS_DIR / sample
        label_dir = LABEL_DIR / sample

        # ── 加载数据 (全分辨率) ──
        bnd_groups = _load_boundaries(out_dir / "all_boundaries.csv")
        cells_full = _load_cells(out_dir / "cell_centroids.csv")

        compact_scale = self._estimate_compact_scale(out_dir, cells_full)

        # ── 转换到紧凑分辨率 ──
        cells = cells_full.copy()
        cells["X"] = cells["X"] * compact_scale
        cells["Y"] = cells["Y"] * compact_scale

        # pia/white 也缩放到紧凑分辨率
        bnd_compact = {}
        for bname in ["pia", "white"]:
            df = bnd_groups[bname].copy()
            df["x"] = df["x"] * compact_scale
            df["y"] = df["y"] * compact_scale
            bnd_compact[bname] = df

        pia_lut = _build_y_lookup(bnd_compact["pia"])
        white_lut = _build_y_lookup(bnd_compact["white"])

        # 全局种子边界 (depth 值，不依赖分辨率)
        seg = pd.read_csv(out_dir / "segmented_depth.csv")
        layers = seg.to_dict("records")
        global_seeds = {
            "L1_2": float(layers[0]["end"]),
            "L3_4": float(layers[1]["end"]),
            "L4_5": float(layers[2]["end"]),
        }

        # ── 加载 GT → 紧凑分辨率 ──
        gt_path = label_dir / "label_mask.png"
        gt_boundaries_compact = None
        if gt_path.exists():
            gt_img_full = _load_image_pil(gt_path)
            sample_algo = cv2.imread(str(out_dir / "layers_color_mask.png"))
            if sample_algo is not None and gt_img_full.shape != sample_algo.shape:
                gt_img_compact = cv2.resize(
                    gt_img_full,
                    (sample_algo.shape[1], sample_algo.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_img_compact = gt_img_full
            gt_boundaries_compact = _extract_gt_boundaries_from_mask(gt_img_compact)

        # ── 有效 x 范围 (紧凑分辨率) ──
        x_min = max(bnd_compact["pia"]["x"].min(), bnd_compact["white"]["x"].min())
        x_max = min(bnd_compact["pia"]["x"].max(), bnd_compact["white"]["x"].max())
        xs = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1)

        # ── 计算全局 CDF 百分位模板 ──
        # 用全体细胞计算每个全局种子 depth 对应的 CDF 百分位
        all_depths = _col_local_cells(
            int((x_min + x_max) / 2), cells,
            pia_lut((x_min + x_max) / 2), white_lut((x_min + x_max) / 2),
            int((x_max - x_min) * 0.4),  # 大窗口获取全局细胞
        )
        if all_depths is None or len(all_depths) < 50:
            # 回退: 逐列收集
            all_depths_list = []
            mid = int((x_min + x_max) / 2)
            for stride_x in range(int(x_min), int(x_max), 10):
                py = pia_lut(stride_x); wy = white_lut(stride_x)
                if np.isfinite(py) and np.isfinite(wy) and abs(wy-py) > 10:
                    nearby = cells[(cells["X"] >= stride_x - 50) & (cells["X"] <= stride_x + 50)]
                    if len(nearby) > 0:
                        d = (nearby["Y"].values - py) / (wy - py + 1e-8)
                        all_depths_list.extend(np.clip(d, 0, 1).tolist())
            all_depths = np.array(all_depths_list) if all_depths_list else None

        if all_depths is not None and len(all_depths) > 50:
            target_pcts = _compute_cdf_percentiles_at_seeds(all_depths, global_seeds)
        else:
            target_pcts = {"L1_2": 8.0, "L3_4": 35.0, "L4_5": 55.0}  # 硬编码回退
        print(f"  全局 CDF 百分位模板: L1_2={target_pcts['L1_2']:.0f}%, "
              f"L3_4={target_pcts['L3_4']:.0f}%, L4_5={target_pcts['L4_5']:.0f}%")

        # ── 逐列精炼 (间隔 column_step 列) ──
        internal_names = ["L1_2", "L3_4", "L4_5"]
        sample_xs = xs[::self.column_step]  # 采样列
        col_results_compact: dict[str, list] = {n: [] for n in internal_names}
        fallback_count = {n: 0 for n in internal_names}
        total_cols = len(sample_xs)

        for x in sample_xs:
            pia_y = pia_lut(x)
            white_y = white_lut(x)
            if not (np.isfinite(pia_y) and np.isfinite(white_y) and abs(white_y - pia_y) > 10):
                continue

            cell_depths = _col_local_cells(x, cells, pia_y, white_y, self.cell_window)
            if cell_depths is None:
                for name in internal_names:
                    d = global_seeds[name]
                    y = pia_y + d * (white_y - pia_y)
                    col_results_compact[name].append((float(x), float(y)))
                    fallback_count[name] += 1
                continue

            # 局部 CDF-based 边界检测
            local_bounds = _local_cdf_boundaries(
                cell_depths, global_seeds, target_pcts, max_delta=0.05,
            )

            for name in internal_names:
                d = local_bounds[name]
                y = pia_y + float(d) * (white_y - pia_y)
                col_results_compact[name].append((float(x), float(y)))

        # ── 插值到所有 x 位置 (使边界连续平滑) ──
        bounds_compact: dict[str, np.ndarray] = {}
        for name in internal_names:
            pts = np.array(col_results_compact[name])
            if len(pts) < 2:
                # 数据不足：用全局种子填充所有 x
                ys = []
                for x in xs:
                    py = pia_lut(x); wy = white_lut(x)
                    if np.isfinite(py) and np.isfinite(wy):
                        ys.append(py + global_seeds[name] * (wy - py))
                    else:
                        ys.append(0)
                bounds_compact[name] = np.column_stack([xs, np.array(ys)])
                continue

            # 对采样点排序后插值到所有 x
            order = np.argsort(pts[:, 0])
            x_sparse = pts[order, 0]
            y_sparse = pts[order, 1]

            # 只插值在有效范围内的 x
            x_dense = xs[(xs >= x_sparse[0]) & (xs <= x_sparse[-1])]
            if len(x_dense) > 1:
                y_dense = np.interp(x_dense, x_sparse, y_sparse)
                # 再应用中值滤波平滑
                y_smooth = median_filter(y_dense, size=self.median_window)
                bounds_compact[name] = np.column_stack([x_dense, y_smooth])
            else:
                bounds_compact[name] = pts

        # ── 计算 MAE (紧凑分辨率 vs GT 紧凑) ──
        mae_before = {}
        mae_after = {}
        if gt_boundaries_compact is not None:
            for name in internal_names:
                before_pts = []
                for x in xs:
                    pia_y = pia_lut(x)
                    white_y = white_lut(x)
                    if np.isfinite(pia_y) and np.isfinite(white_y):
                        y = pia_y + global_seeds[name] * (white_y - pia_y)
                        before_pts.append((float(x), float(y)))
                before_arr = np.array(before_pts) if before_pts else np.empty((0, 2))

                mae_before[name] = _boundary_mae(
                    before_arr, gt_boundaries_compact.get(name, np.empty((0, 2))))
                mae_after[name] = _boundary_mae(
                    bounds_compact[name], gt_boundaries_compact.get(name, np.empty((0, 2))))

        # ── 构建输出 all_boundaries_refined.csv ──
        out_rows = []
        for bname in ["pia", "white"]:
            # pia/white 输出全分辨率 (保持原格式)
            for _, row in bnd_groups[bname].iterrows():
                out_rows.append({"x": row["x"], "y": row["y"], "boundary": bname})
        for name in internal_names:
            # 内部边界输出紧凑分辨率 (保持原格式)
            pts = bounds_compact[name]
            for i in range(len(pts)):
                out_rows.append({"x": round(pts[i, 0], 1),
                                 "y": round(pts[i, 1], 1),
                                 "boundary": name})

        result = {
            "sample": sample,
            "global_seeds": global_seeds,
            "compact_scale": compact_scale,
            "bounds_compact": bounds_compact,
            "mae_before": mae_before,
            "mae_after": mae_after,
            "total_columns": total_cols,
            "fallback_counts": fallback_count,
            "out_rows": out_rows,
            "gt_boundaries_compact": gt_boundaries_compact,
            "pia_lut": pia_lut,
            "white_lut": white_lut,
            "xs": xs,
        }
        return result

    def save_refined_csv(self, result: dict, output_dir: Path):
        """保存精炼后的边界 CSV。"""
        out_df = pd.DataFrame(result["out_rows"])
        out_path = output_dir / "all_boundaries_refined.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  -> 精炼边界已保存: {out_path}")
        return out_path

    def save_comparison_plot(self, result: dict, output_dir: Path, sample: str):
        """保存精炼前后对比图 (紧凑分辨率下)。"""
        gt_bounds = result.get("gt_boundaries_compact")
        if gt_bounds is None:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        internal_names = ["L1_2", "L3_4", "L4_5"]
        titles = ["L1/L2 Boundary", "L2-L3/L4 Boundary", "L4/L5 Boundary"]
        colors = {"before": "orange", "after": "green", "gt": "blue"}

        for idx, name in enumerate(internal_names):
            ax = axes[idx]

            # 精炼前 (全局种子 → 紧凑分辨率)
            xs = result["xs"]
            pia_lut = result["pia_lut"]
            white_lut = result["white_lut"]
            seed = result["global_seeds"][name]
            before = []
            for x in xs:
                py = pia_lut(x)
                wy = white_lut(x)
                if np.isfinite(py) and np.isfinite(wy):
                    before.append((x, py + seed * (wy - py)))
            before_arr = np.array(before)

            # 精炼后 (紧凑分辨率)
            after_arr = result["bounds_compact"].get(name, np.empty((0, 2)))

            # GT (紧凑分辨率)
            gt_arr = gt_bounds.get(name, np.empty((0, 2)))

            if len(before_arr) > 0:
                ax.plot(before_arr[:, 0], before_arr[:, 1], "-",
                        color=colors["before"], alpha=0.6, linewidth=1.5,
                        label=f"Before (MAE={result['mae_before'].get(name, 0):.1f}px)")
            if len(after_arr) > 0:
                ax.plot(after_arr[:, 0], after_arr[:, 1], "-",
                        color=colors["after"], alpha=0.8, linewidth=1.5,
                        label=f"After (MAE={result['mae_after'].get(name, 0):.1f}px)")
            if len(gt_arr) > 0:
                ax.plot(gt_arr[:, 0], gt_arr[:, 1], "--",
                        color=colors["gt"], alpha=0.7, linewidth=1.5,
                        label="GT")

            ax.set_title(f"{titles[idx]} — Sample {sample}", fontsize=12)
            ax.set_xlabel("X (column)", fontsize=10)
            ax.set_ylabel("Y (pixel)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

            # 标注 fallback 比例
            total = result["total_columns"]
            fb = result["fallback_counts"].get(name, 0)
            if fb > 0:
                ax.text(0.02, 0.98, f"Fallback: {fb}/{total} ({fb / max(total, 1) * 100:.0f}%)",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.suptitle(f"Sample {sample} — Boundary Refinement Comparison",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()

        save_dir = output_dir / "comparison.png"
        fig.savefig(str(save_dir), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 对比图已保存: {save_dir}")

    def print_summary(self, result: dict):
        """打印样本精炼摘要。"""
        print(f"\n  样本 {result['sample']} 精炼摘要:")
        print(f"  总列数: {result['total_columns']}")
        print(f"  全局种子: {result['global_seeds']}")
        for name in ["L1_2", "L3_4", "L4_5"]:
            fb = result["fallback_counts"].get(name, 0)
            pct = fb / max(result["total_columns"], 1) * 100
            mae_b = result["mae_before"].get(name, np.nan)
            mae_a = result["mae_after"].get(name, np.nan)
            impr = (mae_b - mae_a) / max(mae_b, 1e-8) * 100 if np.isfinite(mae_b) and mae_b > 0 else 0
            print(f"    {name}: fallback={fb}/{result['total_columns']} ({pct:.0f}%), "
                  f"MAE {mae_b:.1f}→{mae_a:.1f}px ({impr:+.0f}%)")


# ── 汇总评估 ──


def compute_summary_results(all_results: list[dict]) -> pd.DataFrame:
    """汇总所有样本的精炼前后误差对比。"""
    rows = []
    for res in all_results:
        sample = res["sample"]
        for name in ["L1_2", "L3_4", "L4_5"]:
            rows.append({
                "sample": sample,
                "boundary": name,
                "mae_before_px": res["mae_before"].get(name, np.nan),
                "mae_after_px": res["mae_after"].get(name, np.nan),
                "improvement_px": (
                    res["mae_before"].get(name, np.nan) - res["mae_after"].get(name, np.nan)
                    if np.isfinite(res["mae_before"].get(name, np.nan))
                       and np.isfinite(res["mae_after"].get(name, np.nan))
                    else np.nan
                ),
                "improvement_pct": (
                    (res["mae_before"].get(name, np.nan) - res["mae_after"].get(name, np.nan))
                    / max(res["mae_before"].get(name, np.nan), 1e-8) * 100
                    if np.isfinite(res["mae_before"].get(name, np.nan))
                       and res["mae_before"].get(name, np.nan) > 0
                    else np.nan
                ),
                "total_columns": res["total_columns"],
                "fallback_count": res["fallback_counts"].get(name, 0),
                "fallback_pct": res["fallback_counts"].get(name, 0) / max(res["total_columns"], 1) * 100,
            })
    df = pd.DataFrame(rows)

    # 添加汇总行
    summary_row = {"sample": "OVERALL", "boundary": "all"}
    for col in ["mae_before_px", "mae_after_px", "improvement_px", "improvement_pct"]:
        vals = df[col].dropna()
        summary_row[col] = vals.mean() if len(vals) > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    return df


def save_summary_plot(df: pd.DataFrame, save_dir: Path):
    """保存精炼改进汇总图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 精炼前后 MAE 对比
    ax = axes[0]
    plot_df = df[df["boundary"] != "all"].copy()
    x = np.arange(len(plot_df))
    w = 0.35
    before_vals = plot_df["mae_before_px"].values
    after_vals = plot_df["mae_after_px"].values
    ax.bar(x - w / 2, before_vals, w, label="Before", alpha=0.7, color="orange")
    ax.bar(x + w / 2, after_vals, w, label="After", alpha=0.7, color="green")
    ax.set_xticks(x)
    labels = [f"{s.split('_')[0]}-{b.split('_')[0]}" if "_" in b else b
              for s, b in zip(plot_df["sample"], plot_df["boundary"])]
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel("MAE (px)", fontsize=11)
    ax.set_title("Boundary Error Before vs After Refinement", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 右: 改进率 (%) 热力图格式
    ax = axes[1]
    pivot = plot_df.pivot_table(index="sample", columns="boundary",
                                 values="improvement_pct", aggfunc="mean")
    if not pivot.empty:
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=60)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=9, color="black" if abs(v - 20) > 15 else "white")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title("Improvement Rate (%)", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    summary_path = save_dir / "refinement_improvement.png"
    fig.savefig(str(summary_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"汇总改进图已保存: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="密度梯度引导的逐列层边界精炼"
    )
    parser.add_argument("--samples", nargs="*", default=None,
                        help="指定样本 (默认: 所有)")
    parser.add_argument("--cell-window", type=int, default=DEFAULT_CELL_WINDOW,
                        help=f"提取细胞的水平窗口半宽 (默认: {DEFAULT_CELL_WINDOW})")
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA,
                        help=f"密度剖面高斯平滑 sigma (默认: {DEFAULT_SMOOTH_SIGMA})")
    parser.add_argument("--search-window", type=float, default=DEFAULT_SEARCH_WINDOW,
                        help=f"边界搜索窗半宽 depth (默认: {DEFAULT_SEARCH_WINDOW})")
    parser.add_argument("--median-window", type=int, default=DEFAULT_MEDIAN_WINDOW,
                        help=f"列间平滑中值滤波窗口 (默认: {DEFAULT_MEDIAN_WINDOW})")
    parser.add_argument("--column-step", type=int, default=DEFAULT_COLUMN_STEP,
                        help=f"精炼列间距 (像素, 默认: {DEFAULT_COLUMN_STEP})")
    parser.add_argument("--output-suffix", default="_refined",
                        help="输出文件名后缀 (默认: _refined)")
    parser.add_argument("--no-vis", action="store_true",
                        help="跳过生成可视化对比图")
    return parser.parse_args()


def main():
    args = parse_args()

    # 确定样本列表
    if args.samples:
        samples = args.samples
    else:
        samples = sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()])
        samples = [s for s in samples if (OUTPUTS_DIR / s / "all_boundaries.csv").exists()]

    if not samples:
        print("没有找到可用样本！")
        sys.exit(1)

    print(f"运行密度梯度引导的逐列边界精炼...")
    print(f"样本: {samples}")
    print(f"参数: cell_window={args.cell_window}, sigma={args.smooth_sigma}, "
          f"search={args.search_window}, median={args.median_window}, "
          f"column_step={args.column_step}")

    refiner = BoundaryRefiner(
        cell_window=args.cell_window,
        smooth_sigma=args.smooth_sigma,
        search_window=args.search_window,
        median_window=args.median_window,
        column_step=args.column_step,
    )

    all_results = []
    summary_dir = ANALYSIS_DIR / "refinement"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        out_dir = OUTPUTS_DIR / sample
        print(f"\n{'=' * 60}")
        print(f"精炼样本: {sample}")
        print(f"{'=' * 60}")

        result = refiner.refine(sample)
        refiner.save_refined_csv(result, out_dir)
        all_results.append(result)

        if not args.no_vis:
            vis_dir = summary_dir / sample
            vis_dir.mkdir(parents=True, exist_ok=True)
            refiner.save_comparison_plot(result, vis_dir, sample)

        refiner.print_summary(result)

    # 汇总改进
    print(f"\n{'=' * 60}")
    print("汇总精炼改进")
    print(f"{'=' * 60}")

    summary_df = compute_summary_results(all_results)
    summary_csv = summary_dir / "refinement_improvement.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"改进汇总已保存: {summary_csv}")
    print(summary_df.to_string(index=False))

    if not args.no_vis:
        save_summary_plot(summary_df, summary_dir)

    # 汇总统计
    overall = summary_df[summary_df["sample"] == "OVERALL"]
    if len(overall) > 0:
        row = overall.iloc[0]
        print(f"\n总体改进: {row['mae_before_px']:.2f}px → {row['mae_after_px']:.2f}px "
              f"({row['improvement_pct']:.1f}%)")

    print(f"\n所有精炼完成！")


if __name__ == "__main__":
    main()
