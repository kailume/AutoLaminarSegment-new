"""
皮层边界精炼模块 —— 以 depth-density 曲线为输入，回归预测精炼的层边界深度。

核心理念
────────
当前 peak-based 分层从 depth-density 曲线中通过二阶导数过零点确定边界，
但该过程高度依赖高斯平滑参数 sigma 和峰值检测的可靠性。

本模块将这条曲线 + coarse 边界作为特征，用回归模型直接学习从曲线到
真实边界位置的映射，实现"精细拟合"。

输入:
    - 50-bin depth-density 曲线 (来自 computeAverage)
    - 3 个 coarse 边界深度 (来自 peak-based)
    - 衍生特征 (峰位置、峰高、曲线统计量)

输出:
    - 3 个精炼的边界深度: [b12, b34, b456]
      对应 L1/L2/3, L2/3/L4, L4/L5/6 之间的边界

模型:
    sklearn.ensemble.RandomForestRegressor
    适合小样本场景，自带正则化，可输出特征重要性。

Ground Truth 提取:
    从 groundtruth.png 多层 mask 中逐列扫描层间过渡位置，
    再用 GM/WM 边界将 Y 坐标转换为归一化深度 [0,1]。

用法:
    from src.layerBoundaryRefinement import BoundaryRefinement

    refiner = BoundaryRefinement()
    # 训练
    result = refiner.train(sample_list)
    # 推理
    b12, b34, b456 = refiner.predict(binned_density, coarse_boundaries)
"""

import os
import pickle
import warnings
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────
# GT 颜色映射（dataset/test/ 中的 groundtruth.png）
#   BGR(255,100,100) -> L1 (red)
#   BGR(100,255,100) -> L2/3 (green)
#   BGR(100,100,255) -> L4 (blue)
#   BGR(255,100,255) -> L5/6 (magenta)
# ──────────────────────────────────────────────────────────────
GT_LAYER_MAP = {
    (255, 100, 100): 0,   # L1
    (100, 255, 100): 1,   # L2/3
    (100, 100, 255): 2,   # L4
    (255, 100, 255): 3,   # L5/6
}
COLOR_TOLERANCE = 15
LAYER_NAMES = ["1", "2/3", "4", "5/6"]
N_BINS = 50

# ──────────────────────────────────────────────────────────────
# 1. GT 边界提取
# ──────────────────────────────────────────────────────────────


def _load_gt_layer_map(gt_path):
    """将 groundtruth.png 转换为层索引图 (h, w)，每像素值 0~3 或 -1(未定义)。"""
    gt = cv2.imread(str(gt_path))
    if gt is None:
        raise FileNotFoundError(f"无法读取 GT: {gt_path}")

    h, w = gt.shape[:2]
    layer_map = np.full((h, w), -1, dtype=np.int32)
    for (b, g, r), idx in GT_LAYER_MAP.items():
        mask = (
            (np.abs(gt[:, :, 0].astype(np.int32) - b) <= COLOR_TOLERANCE)
            & (np.abs(gt[:, :, 1].astype(np.int32) - g) <= COLOR_TOLERANCE)
            & (np.abs(gt[:, :, 2].astype(np.int32) - r) <= COLOR_TOLERANCE)
        )
        layer_map[mask] = idx
    return layer_map


def extract_gt_boundaries(gt_path, gm_csv, wm_csv, scale=10):
    """
    从 ground Truth mask 中提取 3 条层边界的归一化深度。

    注意: GT mask 通常为全分辨率 DAPI 的 1/10 缩放版本。
          GM/WM 边界应使用 _clipped 版本 (已裁剪到 ROI 的 40x 坐标)。

    参数:
        gt_path: groundtruth.png 路径
        gm_csv: GM 边界 CSV (建议使用 GM_40x_clipped.csv)
        wm_csv: WM 边界 CSV (建议使用 WM_40x_clipped.csv)
        scale: GT 坐标到 40x 坐标的缩放因子 (default: 10)

    返回:
        dict: {
            "boundaries_depth": [b12, b34, b456],   # 3 个归一化边界深度
            "n_columns": int,                         # 有效列数
        }
    """
    layer_map = _load_gt_layer_map(gt_path)
    h, w = layer_map.shape[:2]

    # 加载 GM/WM 边界
    gm_df = pd.read_csv(gm_csv) if isinstance(gm_csv, str) else gm_csv
    wm_df = pd.read_csv(wm_csv) if isinstance(wm_csv, str) else wm_csv
    gm_pts = gm_df[["x", "y"]].to_numpy(dtype=float)
    wm_pts = wm_df[["x", "y"]].to_numpy(dtype=float)

    # 用 KDTree 快速查找每列对应的 GM/WM Y
    gm_tree = cKDTree(gm_pts)
    wm_tree = cKDTree(wm_pts)

    b12_list, b34_list, b456_list = [], [], []
    n_valid_cols = 0

    for x in range(w):
        col = layer_map[:, x]
        valid = col >= 0
        if valid.sum() < 20:
            continue

        diffs = np.diff(col.astype(int))
        trans = np.where((diffs != 0) & (col[:-1] >= 0) & (col[1:] >= 0))[0]
        if len(trans) < 3:
            continue

        # GT x/w 坐标 -> 40x 坐标
        x40 = x * scale
        # 找到此 x 处的 GM/WM 边界 Y (40x 坐标)
        gm_idx = gm_tree.query([[x40, 0]])[1][0]
        wm_idx = wm_tree.query([[x40, 0]])[1][0]
        gm_y = gm_pts[gm_idx][1]
        wm_y = wm_pts[wm_idx][1]

        # 过渡 Y 从 GT 坐标转换到 40x 坐标
        y1_40 = trans[0] * scale
        y2_40 = trans[1] * scale
        y3_40 = trans[2] * scale

        denom = wm_y - gm_y + 1e-8
        d1 = (y1_40 - gm_y) / denom
        d2 = (y2_40 - gm_y) / denom
        d3 = (y3_40 - gm_y) / denom

        if 0 <= d1 <= 1 and 0 <= d2 <= 1 and 0 <= d3 <= 1:
            b12_list.append(d1)
            b34_list.append(d2)
            b456_list.append(d3)
            n_valid_cols += 1

    if n_valid_cols < 10:
        print(f"[extract_gt_boundaries] 警告: 仅 {n_valid_cols} 列有效，使用默认边界")

    boundaries_depth = [
        float(np.median(b12_list)) if b12_list else 0.07,
        float(np.median(b34_list)) if b34_list else 0.35,
        float(np.median(b456_list)) if b456_list else 0.50,
    ]

    print(f"[extract_gt_boundaries] {n_valid_cols} 列, "
          f"边界: [{boundaries_depth[0]:.4f}, {boundaries_depth[1]:.4f}, {boundaries_depth[2]:.4f}]")

    return {
        "boundaries_depth": boundaries_depth,
        "n_columns": n_valid_cols,
    }


# ──────────────────────────────────────────────────────────────
# 2. 从 depth-density 曲线提取特征
# ──────────────────────────────────────────────────────────────


def extract_curve_features(binned_density, bin_centers, coarse_boundaries):
    """
    从 depth-density 曲线和 coarse 边界中提取回归特征。

    使用精简特征集以避免小样本过拟合:
        - 50 维原始密度曲线
        - 3 维 coarse 边界
        - 4 维曲线统计量 (最大值, 峰值位置, 均值, 曲线下面积)
        - 2 维峰特征 (峰数, 最深峰位置)

    返回:
        np.ndarray, shape = (n_features,)
    """
    depth = np.asarray(bin_centers, dtype=float)
    density = np.asarray(binned_density, dtype=float)

    # ---- 1. 原始曲线 (50 维) ----
    features = density.copy()

    # ---- 2. Coarse 边界 (3 维) ----
    features = np.append(features, np.clip(coarse_boundaries, 0.0, 1.0))

    # ---- 3. 曲线统计量 (4 维) ----
    smooth = gaussian_filter1d(density, sigma=2)
    features = np.append(features, [
        float(np.max(smooth)),
        float(np.argmax(smooth)) / len(density),
        float(np.mean(density)),
        float(trapezoid(density, depth)),
    ])

    # ---- 4. 峰特征 (2 维) ----
    try:
        peaks, _ = find_peaks(density, distance=5, prominence=np.ptp(density) * 0.05)
        features = np.append(features, [
            min(len(peaks), 5),
            float(np.max(depth[peaks])) if len(peaks) > 0 else 0.0,
        ])
    except Exception:
        features = np.append(features, [0, 0.0])

    return features


# ──────────────────────────────────────────────────────────────
# 3. 边界精炼器
# ──────────────────────────────────────────────────────────────


class BoundaryRefinement:
    """
    基于 depth-density 曲线特征的层边界精炼。

    以曲线的 50-bin 密度值 + 衍生特征为输入，
    用回归模型预测 3 个精炼边界深度。
    """

    def __init__(self, model_type="random_forest", random_state=42):
        self.random_state = random_state
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.coarse_biases = None  # coarse 边界的系统性偏差

    def _build_model(self):
        """根据 model_type 构建回归模型。"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=3,
                min_samples_split=5,
                random_state=self.random_state,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=3,
                learning_rate=0.05,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def prepare_sample(self, binned_density, bin_centers, coarse_boundaries,
                       gt_boundaries=None):
        """
        准备单个样本的特征和目标。

        返回:
            features: (n_features,) 特征向量
            targets: (3,) 目标边界 [b12, b34, b456] (若 gt_boundaries 为 None 则返回 None)
        """
        features = extract_curve_features(
            binned_density, bin_centers, coarse_boundaries
        )
        targets = np.asarray(gt_boundaries, dtype=float) if gt_boundaries is not None else None
        return features, targets

    def train(self, sample_list):
        """
        训练边界精炼模型。

        参数:
            sample_list: list of dict, 每个 dict 包含:
                - "binned_density": (50,) array
                - "bin_centers": (50,) array
                - "coarse_boundaries": [b12, b34, b456]
                - "gt_boundaries": [b12, b34, b456]  (GT 从 mask 中提取)
                - "name": str (可选)

        返回:
            dict: 训练结果 (MAE, R2, 特征重要性, 留一法验证)
        """
        X_list, y_list, names = [], [], []
        for s in sample_list:
            feat, tgt = self.prepare_sample(
                s["binned_density"], s["bin_centers"],
                s["coarse_boundaries"], s.get("gt_boundaries"),
            )
            if tgt is None or np.any(~np.isfinite(tgt)):
                print(f"  [跳过] {s.get('name', '?')}: GT 边界无效")
                continue
            X_list.append(feat)
            y_list.append(tgt)
            names.append(s.get("name", f"sample_{len(names)}"))

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"训练样本数: {len(X)}")
        print(f"特征维度: {X.shape[1]}")

        if len(X) < 3:
            raise ValueError(f"训练样本不足 ({len(X)}), 需要至少 3 个")

        # 特征归一化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型 (3 个输出, 用多输出回归)
        from sklearn.multioutput import MultiOutputRegressor
        base = self._build_model()
        self.model = MultiOutputRegressor(base, n_jobs=1)
        self.model.fit(X_scaled, y)

        # 计算 coarse 边界的系统性偏差 (用于简单基线)
        # coarse 边界在特征向量中的位置: 50, 51, 52 (前 50 维是密度曲线)
        COARSE_IDX = 50
        self.coarse_biases = np.mean(y - X[:, COARSE_IDX:COARSE_IDX+3], axis=0)

        # 留一法交叉验证
        loo = LeaveOneOut()
        y_pred_loo = np.full_like(y, np.nan)
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train = y[train_idx]
            m = deepcopy(base)
            m.fit(X_train, y_train)
            y_pred_loo[test_idx] = m.predict(X_test)

        mae = mean_absolute_error(y, y_pred_loo)
        r2 = r2_score(y, y_pred_loo, multioutput="variance_weighted")

        # 每边界的 MAE
        mae_per = [mean_absolute_error(y[:, i], y_pred_loo[:, i]) for i in range(3)]

        # 特征重要性 (取 3 个输出的平均)
        if hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
        else:
            importances = np.mean(
                [est.feature_importances_ for est in self.model.estimators_], axis=0
            )

        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        print(f"\n留一法 CV 结果:")
        print(f"  平均 MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        print(f"  边界 1 (L1/L2/3) MAE: {mae_per[0]:.4f}")
        print(f"  边界 2 (L2/3/L4) MAE: {mae_per[1]:.4f}")
        print(f"  边界 3 (L4/L5/6) MAE: {mae_per[2]:.4f}")

        # 样本级详细结果 (coarse 边界在特征索引 50,51,52)
        CI = COARSE_IDX  # = 50
        print(f"\n逐样本结果:")
        for i, name in enumerate(names):
            print(f"  {name}:")
            print(f"    Coarse: [{X[i, CI]:.3f}, {X[i, CI+1]:.3f}, {X[i, CI+2]:.3f}]")
            print(f"    GT:     [{y[i, 0]:.3f}, {y[i, 1]:.3f}, {y[i, 2]:.3f}]")
            print(f"    LOO CV: [{y_pred_loo[i, 0]:.3f}, {y_pred_loo[i, 1]:.3f}, {y_pred_loo[i, 2]:.3f}]")
            print(f"    Coarse 偏差: [{y[i,0]-X[i,CI]:.3f}, {y[i,1]-X[i,CI+1]:.3f}, {y[i,2]-X[i,CI+2]:.3f}]")

        return {
            "mae": mae,
            "mae_per_boundary": mae_per,
            "r2": r2,
            "loo_predictions": y_pred_loo,
            "loo_targets": y,
            "coarse_biases": self.coarse_biases,
            "n_samples": len(X),
            "n_features": X.shape[1],
        }

    def predict(self, binned_density, bin_centers, coarse_boundaries):
        """
        预测精炼的层边界深度。

        返回:
            list[dict]: 与 segmentLayer_peak_based 兼容的格式
                [{"layer": "1", "start": 0.0, "end": b12, "mean_density": ...}, ...]
        """
        if self.model is None:
            raise RuntimeError("模型未训练或未加载。")

        feat, _ = self.prepare_sample(binned_density, bin_centers, coarse_boundaries)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        pred = self.model.predict(feat_scaled)[0]

        # 裁剪到有效范围
        pred = np.clip(pred, 0.02, 0.98)
        # 保证单调递增
        pred.sort()

        boundaries = [0.0, float(pred[0]), float(pred[1]), float(pred[2]), 1.0]
        # 保证最小间距
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = min(boundaries[i - 1] + 0.02, 1.0)

        # 计算 mean_density
        depth = np.asarray(bin_centers, dtype=float)
        density = np.asarray(binned_density, dtype=float)

        layers = []
        for i, name in enumerate(LAYER_NAMES):
            start = boundaries[i]
            end = boundaries[i + 1]
            mask = (depth >= start) & (depth <= end)
            mean_density = float(np.mean(density[mask])) if np.any(mask) else 0.0
            layers.append({
                "layer": name,
                "start": start,
                "end": end,
                "mean_density": mean_density,
            })

        return layers

    def save_model(self, model_path, scaler_path=None):
        """保存模型和 scaler。"""
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "coarse_biases": self.coarse_biases,
            "model_type": self.model_type,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"[BoundaryRefinement] 模型已保存: {model_path}")

        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            print(f"[BoundaryRefinement] Scaler 已保存: {scaler_path}")

    def load_model(self, model_path):
        """加载模型。"""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.coarse_biases = model_data.get("coarse_biases")
        self.model_type = model_data.get("model_type", "random_forest")
        print(f"[BoundaryRefinement] 模型已加载: {model_path}")
        print(f"  模型类型: {self.model_type}")


# ──────────────────────────────────────────────────────────────
# 4. 辅助：从 pipeline 输出中提取特征
# ──────────────────────────────────────────────────────────────


def prepare_from_pipeline(sample_dir, depth_method="legacy"):
    """
    对单个样本目录运行 pipeline 的前几步，提取特征和 GT 边界。

    参数:
        sample_dir: 样本目录路径 (含 WM.csv, GM.csv, cell_centroids.csv, groundtruth.png)

    返回:
        dict: {
            "name": str,
            "binned_density": (50,) array,
            "bin_centers": (50,) array,
            "coarse_boundaries": [b12, b34, b456],
            "gt_boundaries": [b12, b34, b456] 或 None,
        }
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.analyseDensity import analyze, computeAverage, segmentLayer_peak_based

    sample_dir = os.path.abspath(sample_dir)
    name = os.path.basename(sample_dir)

    wm_csv = os.path.join(sample_dir, "WM.csv")
    gm_csv = os.path.join(sample_dir, "GM.csv")
    cell_csv = os.path.join(sample_dir, "cell_centroids.csv")
    gt_png = os.path.join(sample_dir, "groundtruth.png")
    layers_csv = os.path.join(sample_dir, "segmented_layers.csv")

    # 运行分析
    depth, density = analyze(wm=wm_csv, gm=gm_csv, cell=cell_csv,
                             depth_method=depth_method)
    avg_density, bin_centers = computeAverage(depth, density)

    # Coarse 边界
    if os.path.exists(layers_csv):
        coarse_layers = pd.read_csv(layers_csv).to_dict("records")
    else:
        coarse_layers = segmentLayer_peak_based(
            avg_density, bin_centers, sigma=2, merge_layer23=True, issave=False
        )

    coarse_b12 = float(coarse_layers[0]["end"])
    coarse_b34 = float(coarse_layers[1]["end"])
    coarse_b456 = float(coarse_layers[2]["end"])

    # GT 边界
    gt_boundaries = None
    if os.path.exists(gt_png):
        try:
            gt_result = extract_gt_boundaries(gt_png, gm_csv, wm_csv)
            gt_boundaries = gt_result["boundaries_depth"]
            if gt_result["n_columns"] < 10:
                print(f"  [警告] {name}: 仅 {gt_result['n_columns']} 列有效，GT 边界可能不稳定")
        except Exception as e:
            print(f"  [警告] {name}: GT 边界提取失败: {e}")

    return {
        "name": name,
        "binned_density": np.asarray(avg_density, dtype=float),
        "bin_centers": np.asarray(bin_centers, dtype=float),
        "coarse_boundaries": [coarse_b12, coarse_b34, coarse_b456],
        "gt_boundaries": gt_boundaries,
        "gt_n_columns": gt_result["n_columns"] if gt_boundaries else 0,
    }


# ──────────────────────────────────────────────────────────────
# 5. 快速自检
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("边界精炼模块 — 快速自检")
    print("=" * 50)

    # 从 dataset/test/1 提取 GT 边界
    print("\n测试 GT 边界提取 (dataset/test/1):")
    result = extract_gt_boundaries(
        "dataset/test/1/groundtruth.png",
        "dataset/test/1/GM_40x.csv",
        "dataset/test/1/WM_40x.csv",
    )
    print(f"  有效列数: {result['n_columns']}")
    print(f"  GT 边界深度: {result['boundaries_depth']}")
