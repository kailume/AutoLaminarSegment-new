"""
局部边界精炼模块 —— 逐列预测精炼的层边界深度。

核心理念
────────
Coarse 方法给出全局平滑的 depth 阈值线（所有列共享相同边界深度），
但真实层边界在组织不同位置存在局部变异。

本模块以"每列"为单位:
    - 输入: 该列的 coarse 边界深度 + 局部密度特征 + 空间位置
    - 输出: 该列精炼后的 3 条边界深度
    - 模型: RandomForestRegressor (在所有训练样本的所有列上训练)

这样得到的边界线在不同 x 位置可以不同 —— 实现"局部精细拟合"。

用法:
    from src.localRefinement import LocalBoundaryRefiner

    refiner = LocalBoundaryRefiner()
    refiner.train(sample_list)    # sample_list: 1~10
    refiner.predict(sample)       # sample: 11 or 12
"""

import os
import pickle
import warnings
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# ── GT 颜色映射 ──
GT_LAYER_MAP = {
    (255, 100, 100): 0,   # L1 (red)
    (100, 255, 100): 1,   # L2/3 (green)
    (100, 100, 255): 2,   # L4 (blue)
    (255, 100, 255): 3,   # L5/6 (magenta)
}
COLOR_TOL = 15
LAYER_NAMES = ["1", "2/3", "4", "5/6"]
N_DEPTH_BINS = 20  # 每列按 depth 分箱数


def _build_y_lookup(points):
    """构建 x→y 插值函数。"""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.nan
    df = pd.DataFrame(pts, columns=["x", "y"])
    grouped = df.groupby("x", as_index=False)["y"].median().sort_values("x")
    x_vals = grouped["x"].to_numpy(dtype=float)
    y_vals = grouped["y"].to_numpy(dtype=float)
    if len(x_vals) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), y_vals[0], dtype=float)

    def lookup(x_new):
        x_arr = np.asarray(x_new, dtype=float)
        clipped = np.clip(x_arr, x_vals[0], x_vals[-1])
        return np.interp(clipped, x_vals, y_vals)
    return lookup


# ──────────────────────────────────────────────────────────────
# 1. 从 GT mask 逐列提取 3 条边界深度
# ──────────────────────────────────────────────────────────────

def extract_per_column_gt(gt_path, gm_pts, wm_pts, every_n=1):
    """
    从 GT mask 逐列提取 3 条边界深度。

    返回:
        list of dict: [{"x": x, "b12": d, "b34": d, "b456": d}, ...]
        每列一条记录，depth 值 [0,1]。
    """
    gt = cv2.imread(str(gt_path))
    h, w = gt.shape[:2]

    # 构建层索引图
    layer_map = np.full((h, w), -1, dtype=np.int32)
    for (b, g, r), idx in GT_LAYER_MAP.items():
        mask = (
            (np.abs(gt[:, :, 0].astype(np.int32) - b) <= COLOR_TOL) &
            (np.abs(gt[:, :, 1].astype(np.int32) - g) <= COLOR_TOL) &
            (np.abs(gt[:, :, 2].astype(np.int32) - r) <= COLOR_TOL)
        )
        layer_map[mask] = idx

    gm_func = _build_y_lookup(gm_pts)
    wm_func = _build_y_lookup(wm_pts)

    results = []
    for x in range(0, w, every_n):
        col = layer_map[:, x]
        valid = col >= 0
        if valid.sum() < 20:
            continue

        # 找 3 个层过渡
        diffs = np.diff(col.astype(int))
        trans = np.where((diffs != 0) & (col[:-1] >= 0) & (col[1:] >= 0))[0]
        if len(trans) < 3:
            continue

        gm_y = gm_func(x)
        wm_y = wm_func(x)
        if not np.isfinite(gm_y) or not np.isfinite(wm_y) or abs(wm_y - gm_y) < 10:
            continue

        y1, y2, y3 = float(trans[0]), float(trans[1]), float(trans[2])
        denom = wm_y - gm_y + 1e-8
        d1 = (y1 - gm_y) / denom
        d2 = (y2 - gm_y) / denom
        d3 = (y3 - gm_y) / denom

        if 0 <= d1 <= 1 and 0 <= d2 <= 1 and 0 <= d3 <= 1:
            results.append({"x": x, "b12": d1, "b34": d2, "b456": d3})

    return results


# ──────────────────────────────────────────────────────────────
# 2. 逐列特征提取
# ──────────────────────────────────────────────────────────────

def extract_per_column_features(
    x, cell_df, gm_func, wm_func,
    coarse_b12, coarse_b34, coarse_b456,
    w,  # 图像宽度，用于 x 归一化
    dapi_gray=None,  # DAPI 灰度图，用于提取局部强度特征
):
    """
    为单个列 x 提取局部特征。

    特征:
        - 空间位置: x / w (归一化)
        - coarse 边界: [b12, b34, b456]
        - DAPI 强度分箱: N_DEPTH_BINS 个 bin 的平均灰度
        - depth 分箱细胞密度: N_DEPTH_BINS 个 bin 的细胞计数
        - 总细胞数
    共: 1 + 3 + N_DEPTH_BINS*2 + 1 = 45 维
    """
    gm_y = gm_func(x)
    wm_y = wm_func(x)
    if not np.isfinite(gm_y) or not np.isfinite(wm_y) or abs(wm_y - gm_y) < 10:
        return None

    features = [
        float(x) / max(w, 1),           # 归一化 x 位置
        float(coarse_b12),               # coarse L1/L2/3 边界
        float(coarse_b34),               # coarse L2/3/L4 边界
        float(coarse_b456),              # coarse L4/L5/6 边界
    ]

    # DAPI 强度分箱 (沿该列采样)
    if dapi_gray is not None:
        dapi_col = dapi_gray[:, x].astype(float)
        h = len(dapi_col)
        dapi_depths = (np.arange(h, dtype=float) - gm_y) / (wm_y - gm_y + 1e-8)
        dapi_depths = np.clip(dapi_depths, 0.0, 1.0)
        bin_edges = np.linspace(0, 1, N_DEPTH_BINS + 1)
        dapi_binned = np.zeros(N_DEPTH_BINS, dtype=float)
        for i in range(N_DEPTH_BINS):
            mask = (dapi_depths >= bin_edges[i]) & (dapi_depths < bin_edges[i + 1])
            if np.any(mask):
                dapi_binned[i] = float(np.mean(dapi_col[mask]))
        features.extend(dapi_binned.tolist())
    else:
        features.extend([0.0] * N_DEPTH_BINS)

    # 细胞密度分箱：取该列附近 ±5px 范围内的细胞
    margin = 5
    nearby = cell_df[
        (cell_df["X"] >= x - margin) & (cell_df["X"] <= x + margin)
    ]
    n_total = len(nearby)
    features.append(float(n_total))

    if n_total > 0:
        depths = (nearby["Y"].values - gm_y) / (wm_y - gm_y + 1e-8)
        depths = np.clip(depths, 0.0, 1.0)
        bin_edges = np.linspace(0, 1, N_DEPTH_BINS + 1)
        hist, _ = np.histogram(depths, bins=bin_edges)
        features.extend(hist.astype(float))
    else:
        features.extend([0.0] * N_DEPTH_BINS)

    return np.array(features, dtype=float)


# ──────────────────────────────────────────────────────────────
# 3. LocalBoundaryRefiner 类
# ──────────────────────────────────────────────────────────────

class LocalBoundaryRefiner:
    """
    逐列局部边界精炼。

    在所有训练样本的所有列上训练 RandomForest 回归模型。
    """

    def __init__(self, random_state=42):
        self.model = None
        self.scaler = None
        self.random_state = random_state

    def prepare_sample(self, sample_dir, every_n=1):
        """
        为单个样本准备逐列训练数据。

        返回:
            X: (n_cols, n_features) array
            y: (n_cols, 3) array  [b12, b34, b456]
            col_info: [(x, name), ...]
        """
        sample_dir = str(sample_dir)
        name = os.path.basename(sample_dir)

        # 加载管道数据
        cell_df = pd.read_csv(os.path.join(sample_dir, "cell_centroids.csv"))
        # 统一列名
        cols_lower = {c.lower(): c for c in cell_df.columns}
        rename = {}
        if "centroid_x" in cols_lower and "X" not in cell_df.columns:
            rename[cols_lower["centroid_x"]] = "X"
        if "centroid_y" in cols_lower and "Y" not in cell_df.columns:
            rename[cols_lower["centroid_y"]] = "Y"
        if "x" in cols_lower and "X" not in cell_df.columns:
            rename[cols_lower["x"]] = "X"
        if "y" in cols_lower and "Y" not in cell_df.columns:
            rename[cols_lower["y"]] = "Y"
        if rename:
            cell_df = cell_df.rename(columns=rename)

        gm_csv = os.path.join(sample_dir, "GM.csv")
        wm_csv = os.path.join(sample_dir, "WM.csv")
        gm_pts = pd.read_csv(gm_csv)[["x", "y"]].to_numpy(dtype=float)
        wm_pts = pd.read_csv(wm_csv)[["x", "y"]].to_numpy(dtype=float)
        gm_func = _build_y_lookup(gm_pts)
        wm_func = _build_y_lookup(wm_pts)

        # Coarse 边界
        layers_csv = os.path.join(sample_dir, "segmented_layers.csv")
        layers = pd.read_csv(layers_csv).to_dict("records")
        coarse_b12 = float(layers[0]["end"])
        coarse_b34 = float(layers[1]["end"])
        coarse_b456 = float(layers[2]["end"])

        # GT 逐列边界
        sample_id = name  # e.g., "1", "2", etc.
        label_dir = os.path.join("dataset", "inputs", "label", sample_id)
        gt_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]
        if not gt_files:
            print(f"  [跳过] {name}: 无 GT")
            return None, None, []

        gt_path = os.path.join(label_dir, gt_files[0])
        gt_cols = extract_per_column_gt(gt_path, gm_pts, wm_pts, every_n=every_n)
        if len(gt_cols) < 10:
            print(f"  [跳过] {name}: GT 有效列不足 ({len(gt_cols)})")
            return None, None, []

        # 加载 DAPI 图像
        dapi_path = os.path.join(sample_dir, "dapi.png")
        dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
        w_img = dapi_img.shape[1] if dapi_img is not None else 8000

        # 构建特征 (目标: 相对于 coarse 边界的偏移量)
        X_list, y_offset_list, col_info = [], [], []
        for col_data in gt_cols:
            x_col = col_data["x"]
            feat = extract_per_column_features(
                x_col, cell_df, gm_func, wm_func,
                coarse_b12, coarse_b34, coarse_b456,
                w_img, dapi_gray=dapi_img,
            )
            if feat is None:
                continue
            X_list.append(feat)
            # 目标: GT - Coarse (模型学习偏移)
            y_offset_list.append([
                col_data["b12"] - coarse_b12,
                col_data["b34"] - coarse_b34,
                col_data["b456"] - coarse_b456,
            ])
            col_info.append((x_col, name))

        if len(X_list) < 10:
            return None, None, []

        return np.array(X_list), np.array(y_offset_list), col_info

    def train(self, sample_ids, every_n=5):
        """
        训练模型。

        参数:
            sample_ids: list of str, 训练样本 ID (如 ["1","2",...,"10"])
            every_n: 每 n 列取一列 (加速训练)
        """
        all_X, all_y = [], []

        for sid in sample_ids:
            raw_dir = os.path.join("dataset", "inputs", "raw", sid)
            if not os.path.isdir(raw_dir):
                print(f"  [跳过] Sample {sid}: 目录不存在")
                continue

            print(f"\n处理样本 {sid}...")
            X, y, info = self.prepare_sample(raw_dir, every_n=every_n)
            if X is None:
                continue

            all_X.append(X)
            all_y.append(y)
            n_cols = len(info)
            print(f"  {n_cols} 列, 特征维度 {X.shape[1]}")

        if len(all_X) == 0:
            raise ValueError("没有可用的训练样本！")

        X_all = np.vstack(all_X)
        y_all = np.vstack(all_y)
        print(f"\n总训练列数: {len(X_all)}, 特征维度: {X_all.shape[1]}")

        # 归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)

        # 训练
        base = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(base, n_jobs=1)
        self.model.fit(X_scaled, y_all)

        # 训练集误差
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y_all, y_pred)
        print(f"训练集 MAE: {mae:.4f}")

        return {"n_samples": len(all_X), "n_columns": len(X_all), "n_features": X_all.shape[1], "mae": mae}

    def predict_sample(self, sample_dir, every_n=1):
        """
        对单个样本逐列预测精炼边界。

        返回:
            col_results: [(x, b12, b34, b456), ...]
            refined_layers: list[dict] 兼容 segmentLayer_peak_based 格式
        """
        sample_dir = str(sample_dir)
        name = os.path.basename(sample_dir)

        cell_df = pd.read_csv(os.path.join(sample_dir, "cell_centroids.csv"))
        cols_lower = {c.lower(): c for c in cell_df.columns}
        rename = {}
        if "centroid_x" in cols_lower and "X" not in cell_df.columns:
            rename[cols_lower["centroid_x"]] = "X"
        if "centroid_y" in cols_lower and "Y" not in cell_df.columns:
            rename[cols_lower["centroid_y"]] = "Y"
        if "x" in cols_lower and "X" not in cell_df.columns:
            rename[cols_lower["x"]] = "X"
        if "y" in cols_lower and "Y" not in cell_df.columns:
            rename[cols_lower["y"]] = "Y"
        if rename:
            cell_df = cell_df.rename(columns=rename)

        gm_pts = pd.read_csv(os.path.join(sample_dir, "GM.csv"))[["x", "y"]].to_numpy(dtype=float)
        wm_pts = pd.read_csv(os.path.join(sample_dir, "WM.csv"))[["x", "y"]].to_numpy(dtype=float)
        gm_func = _build_y_lookup(gm_pts)
        wm_func = _build_y_lookup(wm_pts)

        layers = pd.read_csv(os.path.join(sample_dir, "segmented_layers.csv")).to_dict("records")
        coarse_b12 = float(layers[0]["end"])
        coarse_b34 = float(layers[1]["end"])
        coarse_b456 = float(layers[2]["end"])

        dapi = cv2.imread(os.path.join(sample_dir, "dapi.png"), cv2.IMREAD_GRAYSCALE)
        w_img = dapi.shape[1] if dapi is not None else 8000

        # 所有列逐列预测
        col_results = []
        X_cols = []
        valid_xs = []

        gm_func_full = _build_y_lookup(gm_pts)
        wm_func_full = _build_y_lookup(wm_pts)

        for x in range(0, w_img, every_n):
            gm_y = gm_func_full(x)
            wm_y = wm_func_full(x)
            if not np.isfinite(gm_y) or not np.isfinite(wm_y) or abs(wm_y - gm_y) < 10:
                continue

            feat = extract_per_column_features(
                x, cell_df, gm_func_full, wm_func_full,
                coarse_b12, coarse_b34, coarse_b456,
                w_img, dapi_gray=dapi,
            )
            if feat is None:
                continue

            X_cols.append(feat)
            valid_xs.append(x)

        if len(X_cols) == 0:
            return [], []

        X_arr = np.array(X_cols)
        X_scaled = self.scaler.transform(X_arr)
        offsets_pred = self.model.predict(X_scaled)

        for i, x in enumerate(valid_xs):
            # 重构: coarse + 预测偏移
            refined_b12 = coarse_b12 + offsets_pred[i, 0]
            refined_b34 = coarse_b34 + offsets_pred[i, 1]
            refined_b456 = coarse_b456 + offsets_pred[i, 2]
            # 裁剪
            refined_b12 = np.clip(refined_b12, 0.02, 0.98)
            refined_b34 = np.clip(refined_b34, refined_b12 + 0.01, 0.98)
            refined_b456 = np.clip(refined_b456, refined_b34 + 0.01, 0.98)
            col_results.append((x, float(refined_b12), float(refined_b34), float(refined_b456)))

        # 聚合为全局边界（取中位数，兼容原格式）
        b12_vals = [r[1] for r in col_results]
        b34_vals = [r[2] for r in col_results]
        b456_vals = [r[3] for r in col_results]

        refined_b12 = float(np.median(b12_vals))
        refined_b34 = float(np.median(b34_vals))
        refined_b456 = float(np.median(b456_vals))

        boundaries = [0.0, refined_b12, refined_b34, refined_b456, 1.0]
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = min(boundaries[i - 1] + 0.02, 1.0)

        refined_layers = []
        for i, lyr_name in enumerate(LAYER_NAMES):
            refined_layers.append({
                "layer": lyr_name,
                "start": float(boundaries[i]),
                "end": float(boundaries[i + 1]),
                "mean_density": 0.0,
            })

        return col_results, refined_layers

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
            }, f)
        print(f"[LocalBoundaryRefiner] 模型已保存: {model_path}")

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        print(f"[LocalBoundaryRefiner] 模型已加载")
