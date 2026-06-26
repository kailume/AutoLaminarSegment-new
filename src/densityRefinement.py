"""
密度驱动边界精炼 —— 以每列细胞点云密度剖面为输入，推理局部层边界。

核心: 对每个组织的每个列 x，提取该列附近细胞的 depth 分布直方图作为密度剖面，
模型学习: 密度剖面 → 边界位置（而不是 coarse→GT 的偏移修正）。
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

N_BINS = 20          # depth 分箱数
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)
LAYER_NAMES = ["1", "2/3", "4", "5/6"]

# GT 颜色映射
GT_LAYER_MAP = {
    (255, 100, 100): 0,
    (100, 255, 100): 1,
    (100, 100, 255): 2,
    (255, 100, 255): 3,
}
COLOR_TOL = 15


def _col_density_profile(x, cell_df, gm_func, wm_func, margin=5):
    """提取单列的细胞密度剖面 (20-dim)。

    取该列附近 ±margin 像素内的细胞，将每个细胞的 Y 坐标转为 depth，
    按 20 个 depth bin 统计细胞计数，归一化为密度。
    """
    nearby = cell_df[(cell_df["X"] >= x - margin) & (cell_df["X"] <= x + margin)]
    n_total = len(nearby)
    profile = np.zeros(N_BINS, dtype=float)

    if n_total == 0:
        return profile, n_total

    gm_y = gm_func(x)
    wm_y = wm_func(x)
    if not (np.isfinite(gm_y) and np.isfinite(wm_y) and abs(wm_y - gm_y) > 10):
        return profile, n_total

    depths = (nearby["Y"].values - gm_y) / (wm_y - gm_y + 1e-8)
    depths = np.clip(depths, 0.0, 1.0)

    hist, _ = np.histogram(depths, bins=BIN_EDGES)
    # 归一化为密度 (除以像素高度对应的比例)
    profile = hist.astype(float)  # / max(1, n_total)
    return profile, n_total


def _extract_gt_boundaries_per_col(gt_path, gm_pts, wm_pts, every_n=1):
    """从 GT mask 逐列提取 3 条边界深度。"""
    import cv2
    gt = cv2.imread(str(gt_path))
    h, w = gt.shape[:2]

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
        diffs = np.diff(col.astype(int))
        trans = np.where((diffs != 0) & (col[:-1] >= 0) & (col[1:] >= 0))[0]
        if len(trans) < 3:
            continue

        gm_y = gm_func(x)
        wm_y = wm_func(x)
        if not (np.isfinite(gm_y) and np.isfinite(wm_y) and abs(wm_y - gm_y) > 10):
            continue

        y1, y2, y3 = float(trans[0]), float(trans[1]), float(trans[2])
        denom = wm_y - gm_y + 1e-8
        d1, d2, d3 = (y1 - gm_y) / denom, (y2 - gm_y) / denom, (y3 - gm_y) / denom
        if 0 <= d1 <= 1 and 0 <= d2 <= 1 and 0 <= d3 <= 1:
            results.append({"x": x, "b12": d1, "b34": d2, "b456": d3})
    return results


def _build_y_lookup(points):
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
# 核心类
# ──────────────────────────────────────────────────────────────

class DensityDrivenRefiner:
    """
    密度驱动边界精炼。

    对每个样本的每一列:
        特征: 20-bin 细胞密度剖面 + coarse 边界深度
        目标: GT 边界深度

    模型学习: 密度模式 → 边界位置
    """

    def __init__(self):
        self.model = None
        self.scaler = None

    def _load_sample(self, sample_id, every_n=5):
        """加载单样本的逐列特征和目标。"""
        raw_dir = f"dataset/inputs/raw/{sample_id}"
        label_dir = f"dataset/inputs/label/{sample_id}"

        cell_df = pd.read_csv(f"{raw_dir}/cell_centroids.csv")
        # 统一列名
        for src, dst in [("centroid_x", "X"), ("centroid_y", "Y"), ("x", "X"), ("y", "Y")]:
            if src in cell_df.columns and dst not in cell_df.columns:
                cell_df = cell_df.rename(columns={src: dst})
        if "X" not in cell_df.columns or "Y" not in cell_df.columns:
            raise ValueError(f"Sample {sample_id}: no coordinate columns")

        gm_pts = pd.read_csv(f"{raw_dir}/GM.csv")[["x", "y"]].to_numpy(dtype=float)
        wm_pts = pd.read_csv(f"{raw_dir}/WM.csv")[["x", "y"]].to_numpy(dtype=float)
        gm_func = _build_y_lookup(gm_pts)
        wm_func = _build_y_lookup(wm_pts)

        layers = pd.read_csv(f"{raw_dir}/segmented_layers.csv").to_dict("records")
        coarse = [layers[0]["end"], layers[1]["end"], layers[2]["end"]]

        # GT
        gt_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]
        if not gt_files:
            return None, None
        gt_cols = _extract_gt_boundaries_per_col(
            f"{label_dir}/{gt_files[0]}", gm_pts, wm_pts, every_n=every_n
        )
        if len(gt_cols) < 5:
            return None, None

        # 构建特征矩阵
        X_list, y_list = [], []
        for c in gt_cols:
            x_col = c["x"]
            profile, n_total = _col_density_profile(x_col, cell_df, gm_func, wm_func)
            # 特征: 20-dim 密度剖面 + 3 coarse 边界 + 归一化 x
            feat = np.concatenate([profile, coarse, [x_col / 10000.0]])
            X_list.append(feat)
            y_list.append([c["b12"], c["b34"], c["b456"]])

        return np.array(X_list), np.array(y_list)

    def train(self, sample_ids, every_n=5):
        """训练模型。"""
        all_X, all_y = [], []
        for sid in sample_ids:
            sid = str(sid)
            X, y = self._load_sample(sid, every_n=every_n)
            if X is None:
                print(f"  Sample {sid}: skip")
                continue
            all_X.append(X)
            all_y.append(y)
            print(f"  Sample {sid}: {len(X)} columns")

        X_all = np.vstack(all_X)
        y_all = np.vstack(all_y)
        print(f"\nTotal: {len(X_all)} columns, {X_all.shape[1]} features")

        # 归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)

        # 训练随机森林
        base = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            min_samples_split=10, random_state=42, n_jobs=-1,
        )
        self.model = MultiOutputRegressor(base, n_jobs=1)
        self.model.fit(X_scaled, y_all)

        # 训练集误差
        y_pred = self.model.predict(X_scaled)
        mae = np.mean(np.abs(y_all - y_pred))
        print(f"Training MAE: {mae:.4f}")
        return {"n_columns": len(X_all), "n_features": X_all.shape[1], "mae": float(mae)}

    def predict(self, sample_id, every_n=5):
        """对单样本逐列预测精炼边界。

        返回:
            boundaries: [b12, b34, b456] 全局中位数
            per_column: [(x, b12, b34, b456), ...]
        """
        raw_dir = f"dataset/inputs/raw/{sample_id}"

        cell_df = pd.read_csv(f"{raw_dir}/cell_centroids.csv")
        for src, dst in [("centroid_x", "X"), ("centroid_y", "Y"), ("x", "X"), ("y", "Y")]:
            if src in cell_df.columns and dst not in cell_df.columns:
                cell_df = cell_df.rename(columns={src: dst})

        gm_pts = pd.read_csv(f"{raw_dir}/GM.csv")[["x", "y"]].to_numpy(dtype=float)
        wm_pts = pd.read_csv(f"{raw_dir}/WM.csv")[["x", "y"]].to_numpy(dtype=float)
        gm_func = _build_y_lookup(gm_pts)
        wm_func = _build_y_lookup(wm_pts)

        layers = pd.read_csv(f"{raw_dir}/segmented_layers.csv").to_dict("records")
        coarse = [layers[0]["end"], layers[1]["end"], layers[2]["end"]]

        # 获取图像宽度
        import cv2
        dapi = cv2.imread(f"{raw_dir}/dapi.png", cv2.IMREAD_GRAYSCALE)
        w_img = dapi.shape[1] if dapi is not None else 8000

        X_list, xs_list = [], []
        for x in range(0, w_img, every_n):
            gm_y = gm_func(x)
            wm_y = wm_func(x)
            if not (np.isfinite(gm_y) and np.isfinite(wm_y) and abs(wm_y - gm_y) > 10):
                continue
            profile, n_total = _col_density_profile(x, cell_df, gm_func, wm_func)
            if n_total < 3:
                continue
            feat = np.concatenate([profile, coarse, [x / 10000.0]])
            X_list.append(feat)
            xs_list.append(x)

        if len(X_list) == 0:
            return coarse, []

        X_arr = np.array(X_list)
        X_scaled = self.scaler.transform(X_arr)
        y_pred = self.model.predict(X_scaled)

        # 裁剪+单调保证
        per_column = []
        b12s, b34s, b456s = [], [], []
        for i, x in enumerate(xs_list):
            b12 = float(np.clip(y_pred[i, 0], 0.02, 0.98))
            b34 = float(np.clip(y_pred[i, 1], 0.02, 0.98))
            b456 = float(np.clip(y_pred[i, 2], 0.02, 0.98))
            # 排序保证
            depths = sorted([b12, b34, b456])
            per_column.append((x, depths[0], depths[1], depths[2]))
            b12s.append(depths[0])
            b34s.append(depths[1])
            b456s.append(depths[2])

        # 取中位数作为全局边界
        boundaries = [float(np.median(b12s)), float(np.median(b34s)), float(np.median(b456s))]
        for i in range(1, 3):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = min(boundaries[i - 1] + 0.02, 0.98)

        return boundaries, per_column

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"Model saved: {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        print(f"Model loaded: {path}")
