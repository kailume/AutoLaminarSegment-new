"""
边界精炼模型 —— 精简版。

基于 coarse 边界预测精炼后边界，支持两种模式:
    1. 偏差修正: 学习 coarse→GT 的系统性偏移
    2. 全模型: 以 depth-density 曲线为特征预测精炼边界

样本 11,12 的边界深度明显大于训练集 (1~10)，因此采用
稳健的偏差修正 + 简单回归组合，避免过拟合。
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.analyseDensity import analyze, computeAverage, segmentLayer_peak_based

warnings.filterwarnings("ignore")

LAYER_NAMES = ["1", "2/3", "4", "5/6"]


def prepare_training_data(sample_ids):
    """
    为多个样本准备训练数据。

    每个样本提取:
        - coarse 边界 (3 个 depth)
        - GT 边界 (3 个 depth) — 从 GT mask 提取

    返回:
        X: (n_samples, 3) — coarse 边界
        y: (n_samples, 3) — GT 边界偏移 (GT - coarse)
    """
    from src.localRefinement import extract_per_column_gt
    import cv2

    X_list, y_list, names = [], [], []

    for sid in sample_ids:
        raw_dir = f"dataset/inputs/raw/{sid}"
        label_dir = f"dataset/inputs/label/{sid}"

        if not os.path.isdir(raw_dir):
            print(f"  [跳过] Sample {sid}: 无 raw 目录")
            continue

        # Coarse 边界
        layers = pd.read_csv(f"{raw_dir}/segmented_layers.csv").to_dict("records")
        coarse = [float(layers[0]["end"]), float(layers[1]["end"]), float(layers[2]["end"])]

        # GT 边界 (从 mask 逐列提取后取中位数)
        gt_files = [f for f in os.listdir(label_dir) if f.endswith(".png")]
        if not gt_files:
            print(f"  [跳过] Sample {sid}: 无 GT")
            continue

        gm_pts = pd.read_csv(f"{raw_dir}/GM.csv")[["x", "y"]].to_numpy(dtype=float)
        wm_pts = pd.read_csv(f"{raw_dir}/WM.csv")[["x", "y"]].to_numpy(dtype=float)

        gt_cols = extract_per_column_gt(
            f"{label_dir}/{gt_files[0]}", gm_pts, wm_pts, every_n=10
        )
        if len(gt_cols) < 10:
            print(f"  [跳过] Sample {sid}: GT 有效列不足 ({len(gt_cols)})")
            continue

        gt_b12 = np.median([c["b12"] for c in gt_cols])
        gt_b34 = np.median([c["b34"] for c in gt_cols])
        gt_b456 = np.median([c["b456"] for c in gt_cols])
        gt = [gt_b12, gt_b34, gt_b456]

        # 偏移量
        offset = [gt[i] - coarse[i] for i in range(3)]

        X_list.append(coarse)
        y_list.append(offset)
        names.append(sid)

        print(f"  Sample {sid}: coarse=[{coarse[0]:.3f},{coarse[1]:.3f},{coarse[2]:.3f}] "
              f"GT=[{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f}] "
              f"Δ=[{offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}]")

    return np.array(X_list), np.array(y_list), names


class SimpleBoundaryRefiner:
    """
    简单边界精炼: 学习 coarse→GT 的偏移修正。
    """

    def __init__(self):
        self.model = None  # 3 个独立的 RFR, 每个预测一个边界的偏移
        self.offset_means = None  # 平均偏移 (fallback)

    def train(self, sample_ids):
        X, y, names = prepare_training_data(sample_ids)
        n = len(X)
        print(f"\n训练样本数: {n}")

        # 平均偏移 (作为简单基线)
        self.offset_means = np.mean(y, axis=0)
        print(f"平均偏移: Δb12={self.offset_means[0]:+.4f}, "
              f"Δb34={self.offset_means[1]:+.4f}, "
              f"Δb456={self.offset_means[2]:+.4f}")

        if n < 3:
            print("训练样本不足，仅使用平均偏移")
            self.model = None
            return

        # 训练 3 个独立的随机森林 (每个边界一个)
        self.model = []
        for i in range(3):
            m = RandomForestRegressor(
                n_estimators=100, max_depth=3, min_samples_leaf=2,
                random_state=42,
            )
            m.fit(X, y[:, i])
            self.model.append(m)

        # 留一法评估
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_absolute_error

        y_pred_offset = np.zeros_like(y)
        for train_idx, test_idx in LeaveOneOut().split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            for i in range(3):
                m = RandomForestRegressor(
                    n_estimators=50, max_depth=2, min_samples_leaf=2,
                    random_state=42,
                )
                m.fit(X_tr, y_tr[:, i])
                y_pred_offset[test_idx, i] = m.predict(X_te)

        print("留一法结果 (偏移 MAE):")
        for i, name in enumerate(["b12", "b34", "b456"]):
            mae = mean_absolute_error(y[:, i], y_pred_offset[:, i])
            baseline_mae = mean_absolute_error(y[:, i],
                                                np.full_like(y[:, i], self.offset_means[i]))
            print(f"  {name}: 模型 MAE={mae:.4f}, 平均偏移 MAE={baseline_mae:.4f}")

    def predict(self, coarse_boundaries):
        """
        预测精炼边界。

        返回:
            list[dict]: 与 segmentLayer_peak_based 兼容的格式
        """
        coarse = np.asarray(coarse_boundaries, dtype=float).reshape(1, -1)

        if self.model is not None and len(coarse_boundaries) == 3:
            offsets = np.array([m.predict(coarse)[0] for m in self.model])
        else:
            offsets = self.offset_means

        boundaries = [0.0]
        for i in range(3):
            b = float(np.clip(coarse_boundaries[i] + offsets[i], 0.02, 0.98))
            # 保证单调
            if b <= boundaries[-1]:
                b = boundaries[-1] + 0.02
            boundaries.append(b)
        boundaries.append(1.0)

        return [
            {"layer": LAYER_NAMES[i], "start": boundaries[i],
             "end": boundaries[i + 1], "mean_density": 0.0}
            for i in range(4)
        ]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "offset_means": self.offset_means}, f)
        print(f"模型已保存: {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.offset_means = data["offset_means"]
        print(f"模型已加载: {path}")
        print(f"  偏移: b12={self.offset_means[0]:+.4f}, "
              f"b34={self.offset_means[1]:+.4f}, "
              f"b456={self.offset_means[2]:+.4f}")
