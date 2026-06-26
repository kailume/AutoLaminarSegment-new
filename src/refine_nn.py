#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经网络边界精炼 —— 基于 PyTorch 的逐列层边界微调。

用 GT 标签训练一个简单 MLP，学习从局部细胞密度模式到边界位置的映射。

用法:
  .\\venv-cellpose\\Scripts\\python src/refine_nn.py                     # 留一法完整评估
  .\\venv-cellpose\\Scripts\\python src/refine_nn.py --quick           # 快速模式 (仅 3 折)
  .\\venv-cellpose\\Scripts\\python src/refine_nn.py --samples 1 2 3   # 训练指定样本
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ──
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "dataset" / "outputs"
LABEL_DIR = ROOT / "dataset" / "inputs" / "label"
ANALYSIS_DIR = ROOT / "dataset" / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "refinement_nn"
MODEL_DIR = ROOT / "models" / "refine_nn"

# ── 参数 ──
N_DEPTH_BINS = 20              # 密度剖面和 DAPI 强度分箱数
CELL_WINDOW = 80               # 提取细胞的窗口半宽 (紧凑像素, ≈800px 全分辨率)
FEATURE_DIM = 45               # 20(density) + 20(DAPI) + 3(coarse) + 1(x_pos) + 1(cell_count)
OUTPUT_SMOOTH_WINDOW = 15      # 输出边界线的中值滤波平滑窗口 (奇数) — 0=不平滑
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 20
SMOOTH_LOSS_WEIGHT = 0.1       # 平滑损失权重
TRAIN_SAMPLE_EVERY = 5         # 训练时每 N 列取一列
PREDICT_EVERY = 1              # 预测时每 N 列取一列 (输出所有列)

# GT 颜色映射 (BGR)
GT_COLOR_MAP = {
    (255, 100, 100): 0,   # L1
    (100, 255, 100): 1,   # L2/3
    (100, 100, 255): 2,   # L4
    (255, 255, 100): 3,   # L5/6
}
COLOR_TOL = 15


# ═══════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════


def _build_y_lookup(pts_array: np.ndarray):
    """从 Nx2 点阵构建 x→y 插值函数。"""
    pts = pts_array[np.isfinite(pts_array).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.nan
    df = pd.DataFrame(pts, columns=["x", "y"])
    g = df.groupby("x", as_index=False)["y"].median().sort_values("x")
    xv, yv = g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)
    if len(xv) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), yv[0], dtype=float)

    def _lut(x_new):
        xa = np.asarray(x_new, dtype=float)
        return np.interp(np.clip(xa, xv[0], xv[-1]), xv, yv)
    return _lut


def _load_cells(csv_path: Path) -> pd.DataFrame:
    """加载细胞质心，统一列名为 X, Y。"""
    df = pd.read_csv(csv_path)
    col_map = {}
    for src in ["centroid_x", "Centroid_x", "centroidX", "x", "X"]:
        if src in df.columns and "X" not in col_map:
            col_map[src] = "X"
    for src in ["centroid_y", "Centroid_y", "centroidY", "y", "Y"]:
        if src in df.columns and "Y" not in col_map:
            col_map[src] = "Y"
    if col_map:
        df = df.rename(columns=col_map)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"缺少坐标列: {list(df.columns)}")
    return df[["X", "Y"]].copy().astype(float)


def _col_density_profile(x: int, cells: pd.DataFrame,
                          pia_y: float, white_y: float,
                          window: int, n_bins: int) -> np.ndarray:
    """获取列 x 的细胞密度剖面 (n_bins 维向量)。

    在 [0,1] depth 空间分箱，返回每 bin 细胞计数。
    细胞不足时返回全零向量。
    """
    if not (np.isfinite(pia_y) and np.isfinite(white_y) and abs(white_y - pia_y) > 10):
        return np.zeros(n_bins, dtype=float)
    nearby = cells[(cells["X"] >= x - window) & (cells["X"] <= x + window)]
    if len(nearby) < 3:
        return np.zeros(n_bins, dtype=float)
    depths = (nearby["Y"].values - pia_y) / (white_y - pia_y + 1e-8)
    depths = np.clip(depths, 0.0, 1.0)
    hist, _ = np.histogram(depths, bins=np.linspace(0, 1, n_bins + 1))
    return hist.astype(float)


def _estimate_compact_scale(out_dir: Path, cells: pd.DataFrame) -> float:
    """估算紧凑缩放比例。"""
    algo = cv2.imread(str(out_dir / "layers_color_mask.png"))
    if algo is None:
        return 0.1
    compact_h = algo.shape[0]
    full_h = float(cells["Y"].max())
    return compact_h / full_h if full_h > 0 else 0.1


def _load_dapi_compact(out_dir: Path, scale: float) -> np.ndarray | None:
    """加载 DAPI 图像并缩放到紧凑分辨率。

    优先从 preprocessed 目录加载原始 DAPI，回退到输出目录。
    """
    from PIL import Image as PILImage
    sample_name = out_dir.name
    preprocessed_path = ROOT / "dataset" / "inputs" / "preprocessed" / sample_name / "dapi.png"
    candidates = []
    if preprocessed_path.exists():
        candidates.append(preprocessed_path)
    for ext in [".png", ".tif", ".tiff"]:
        candidates.append(out_dir / f"dapi{ext}")
    for path in candidates:
        if path.exists():
            try:
                PILImage.MAX_IMAGE_PIXELS = None
                img = np.array(PILImage.open(str(path)).convert("L")).astype(float)
                algo = cv2.imread(str(out_dir / "layers_color_mask.png"))
                if algo is not None and img.shape[:2] != algo.shape[:2]:
                    img = cv2.resize(img, (algo.shape[1], algo.shape[0]),
                                      interpolation=cv2.INTER_AREA)
                return img
            except Exception:
                continue
    return None


def _col_dapi_intensity(x: int, dapi_gray: np.ndarray,
                         pia_y: float, white_y: float,
                         n_bins: int) -> np.ndarray:
    """提取列 x 的 DAPI 强度剖面 (n_bins 维)。"""
    h, w = dapi_gray.shape[:2]
    if x < 0 or x >= w:
        return np.zeros(n_bins, dtype=float)
    if not (np.isfinite(pia_y) and np.isfinite(white_y) and abs(white_y - pia_y) > 10):
        return np.zeros(n_bins, dtype=float)
    col = dapi_gray[:, x].astype(float)
    depths = (np.arange(h, dtype=float) - pia_y) / (white_y - pia_y + 1e-8)
    depths = np.clip(depths, 0.0, 1.0)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    binned = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        mask = (depths >= bin_edges[i]) & (depths < bin_edges[i + 1])
        if np.any(mask):
            binned[i] = float(np.mean(col[mask]))
    return binned


def _extract_gt_boundaries(label_csv: Path) -> dict[str, np.ndarray]:
    """从 GT 边界 CSV 提取每类边界的 (x, y) 点集。

    返回: {boundary_name: (N,2) ndarray}
    """
    df = pd.read_csv(label_csv)
    groups = {}
    for name, sub in df.groupby("boundary"):
        groups[name] = sub[["x", "y"]].to_numpy(dtype=float)
    return groups


def _boundary_mae(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """计算两条边界线的 MAE (像素)。"""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    pred_x = pd.DataFrame(pred_pts, columns=["x", "y"]).groupby("x")["y"].median()
    gt_lut = _build_y_lookup(gt_pts)
    errors = []
    for x, yp in pred_x.items():
        yg = gt_lut(x)
        if np.isfinite(yg):
            errors.append(abs(yp - yg))
    return float(np.mean(errors)) if errors else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  特征与目标提取
# ═══════════════════════════════════════════════════════════════════════


def extract_sample_data(sample: str, every_n: int = 1):
    """提取单个样本的逐列特征和目标。

    返回:
        features: (N, FEATURE_DIM) ndarray  [20 density | 20 DAPI | 3 coarse | 1 x | 1 n_cells]
        targets:  (N, 3) ndarray  [b12, b34, b456] GT depths
        col_xs:   (N,) list 列 x 坐标
        pix_info: dict {pia_lut, white_lut, compact_scale, global_seeds}
                 用于将深度转像素坐标
    """
    out_dir = OUTPUTS_DIR / sample
    label_dir = LABEL_DIR / sample

    # ── 加载数据 ──
    cells_full = _load_cells(out_dir / "cell_centroids.csv")
    scale = _estimate_compact_scale(out_dir, cells_full)

    # 缩放到紧凑分辨率
    cells = cells_full.copy()
    cells["X"] = cells["X"] * scale
    cells["Y"] = cells["Y"] * scale

    # pia/white 边界
    bnd = pd.read_csv(out_dir / "all_boundaries.csv")
    pia_df = bnd[bnd["boundary"] == "pia"][["x", "y"]].copy()
    white_df = bnd[bnd["boundary"] == "white"][["x", "y"]].copy()
    pia_df = pia_df * scale
    white_df = white_df * scale
    pia_lut = _build_y_lookup(pia_df.to_numpy())
    white_lut = _build_y_lookup(white_df.to_numpy())

    # 全局粗边界
    seg = pd.read_csv(out_dir / "segmented_depth.csv")
    layers = seg.to_dict("records")
    global_b12 = float(layers[0]["end"])
    global_b34 = float(layers[1]["end"])
    global_b456 = float(layers[2]["end"])

    # DAPI 图像 (紧凑分辨率)
    dapi_gray = _load_dapi_compact(out_dir, scale)
    h_dapi, w_dapi = dapi_gray.shape[:2] if dapi_gray is not None else (0,
                            int(pia_df["x"].max()) + 1)

    # GT 边界
    gt_path = label_dir / "label_boundaries.csv"
    if not gt_path.exists():
        print(f"  [跳过] {sample}: 无 GT 边界文件")
        return None, None, None, None
    gt_groups = _extract_gt_boundaries(gt_path)
    gt_lut_b12 = _build_y_lookup(gt_groups.get("L1_2", np.empty((0, 2))))
    gt_lut_b34 = _build_y_lookup(gt_groups.get("L3_4", np.empty((0, 2))))
    gt_lut_b456 = _build_y_lookup(gt_groups.get("L4_5", np.empty((0, 2))))
    gt_pia_lut = _build_y_lookup(gt_groups.get("pia", np.empty((0, 2))))
    gt_white_lut = _build_y_lookup(gt_groups.get("white", np.empty((0, 2))))

    # 有效 x 范围 (限制在 DAPI 图像宽度内)
    x_min = max(pia_df["x"].min(), white_df["x"].min())
    x_max = min(pia_df["x"].max(), white_df["x"].max())
    if dapi_gray is not None:
        x_max = min(x_max, dapi_gray.shape[1] - 1)
        x_min = max(x_min, 0)
    xs = np.arange(int(np.ceil(x_min)), int(np.floor(x_max)) + 1, every_n)

    features, targets, col_xs = [], [], []
    w_img = pia_df["x"].max()  # 图像宽度

    for x in xs:
        pia_y = pia_lut(x)
        white_y = white_lut(x)
        if not (np.isfinite(pia_y) and np.isfinite(white_y) and abs(white_y - pia_y) > 10):
            continue

        # 密度剖面
        density = _col_density_profile(x, cells, pia_y, white_y,
                                        CELL_WINDOW, N_DEPTH_BINS)

        # GT 深度
        gt_b12 = gt_lut_b12(x)
        gt_b34 = gt_lut_b34(x)
        gt_b456 = gt_lut_b456(x)
        # 用 GT pia/white 归一化深度
        gt_pia = gt_pia_lut(x)
        gt_white = gt_white_lut(x)
        if np.isfinite(gt_pia) and np.isfinite(gt_white) and abs(gt_white - gt_pia) > 10:
            d12 = (gt_b12 - gt_pia) / (gt_white - gt_pia + 1e-8)
            d34 = (gt_b34 - gt_pia) / (gt_white - gt_pia + 1e-8)
            d456 = (gt_b456 - gt_pia) / (gt_white - gt_pia + 1e-8)
            if not (0 <= d12 <= 1 and 0 <= d34 <= 1 and 0 <= d456 <= 1):
                continue
            if not (d12 < d34 < d456):
                continue
        else:
            continue

        # DAPI 强度剖面
        dapi_feat = _col_dapi_intensity(x, dapi_gray, pia_y, white_y, N_DEPTH_BINS) \
            if dapi_gray is not None else np.zeros(N_DEPTH_BINS)

        # 细胞总数 (归一化)
        nearby = cells[(cells["X"] >= x - CELL_WINDOW) & (cells["X"] <= x + CELL_WINDOW)]
        n_cells = float(len(nearby)) / 100.0

        feat = np.concatenate([
            density,                           # 0-19: 密度剖面
            dapi_feat,                         # 20-39: DAPI 强度剖面
            [global_b12, global_b34, global_b456],  # 40-42: 粗边界
            [float(x) / max(w_img, 1)],         # 43: x 位置
            [n_cells],                          # 44: 细胞总数
        ])
        features.append(feat)
        targets.append([d12, d34, d456])
        col_xs.append(x)

    if len(features) < 5:
        return None, None, None, None

    pix_info = {
        "pia_lut": pia_lut,
        "white_lut": white_lut,
        "compact_scale": scale,
        "global_seeds": (global_b12, global_b34, global_b456),
        "w_img": w_img,
    }
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32), col_xs, pix_info


# ═══════════════════════════════════════════════════════════════════════
#  PyTorch 模型与数据集
# ═══════════════════════════════════════════════════════════════════════


class BoundaryDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BoundaryMLP(nn.Module):
    """小型残差 MLP 用于边界预测。

    学习 coarse 边界的偏移量 (residual)，而非绝对深度。
    输出 = clamp(coarse + delta, 0, 1)

    设计为低容量防过拟合 (12 samples only, ~1800 rows):
      features[40:43] = coarse boundaries
    """

    def __init__(self, input_dim: int = FEATURE_DIM, dropout: float = 0.15):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 3),
        )
        # 初始化最后线性层接近零 → delta ≈ 0 → 初始预测 ≈ coarse
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        coarse = x[:, 40:43]  # coarse boundaries 在特征索引 40-42
        delta = self.fc(x)
        out = coarse + delta
        return torch.clamp(out, 0.02, 0.98)


def save_model(model: BoundaryMLP, path: Path, scaler_mean: np.ndarray | None = None,
               scaler_std: np.ndarray | None = None):
    """保存模型权重及归一化参数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "input_dim": model.fc[0].in_features,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }
    torch.save(state, str(path))
    print(f"  模型已保存: {path}")


def load_model(path: Path, device: str = "cpu") -> tuple[BoundaryMLP, np.ndarray | None, np.ndarray | None]:
    """加载模型及归一化参数。"""
    state = torch.load(str(path), map_location=device, weights_only=True)
    model = BoundaryMLP(input_dim=state["input_dim"]).to(device)
    model.load_state_dict(state["model_state"])
    print(f"  模型已加载: {path}")
    return model, state.get("scaler_mean"), state.get("scaler_std")


def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """相邻列预测差的 L2 范数作为平滑正则。"""
    if predictions.shape[0] < 2:
        return torch.tensor(0.0, device=predictions.device)
    diffs = predictions[1:] - predictions[:-1]
    return torch.mean(diffs ** 2)


def train_model(train_feat: np.ndarray, train_tgt: np.ndarray,
                val_feat: np.ndarray | None = None,
                val_tgt: np.ndarray | None = None,
                max_epochs: int = MAX_EPOCHS,
                batch_size: int = TRAIN_BATCH_SIZE,
                lr: float = LEARNING_RATE,
                weight_decay: float = WEIGHT_DECAY,
                patience: int = PATIENCE,
                smooth_weight: float = SMOOTH_LOSS_WEIGHT,
                device: str = "cpu") -> BoundaryMLP:
    """训练边界预测 MLP。"""
    model = BoundaryMLP(input_dim=train_feat.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3, verbose=False)

    train_loader = DataLoader(
        BoundaryDataset(train_feat, train_tgt),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    val_loader = None
    if val_feat is not None and val_tgt is not None and len(val_feat) > 0:
        val_loader = DataLoader(
            BoundaryDataset(val_feat, val_tgt),
            batch_size=batch_size * 2, shuffle=False,
        )

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss_l1 = nn.functional.l1_loss(pred, yb)
            loss_smooth = smoothness_loss(pred) * smooth_weight
            loss = loss_l1 + loss_smooth
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_train += len(xb)
        train_loss /= max(n_train, 1)

        # ── 验证 ──
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = nn.functional.l1_loss(pred, yb)
                    val_loss += loss.item() * len(xb)
                    n_val += len(xb)
            val_loss /= max(n_val, 1)
            scheduler.step(val_loss)
            current_loss = val_loss
        else:
            current_loss = train_loss
            scheduler.step(train_loss)

        # ── Early stopping ──
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # 恢复最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════
#  预测与评估
# ═══════════════════════════════════════════════════════════════════════


def predict_sample(model: BoundaryMLP, sample: str,
                   device: str = "cpu") -> dict | None:
    """对单样本进行逐列边界预测。

    返回:
        {
            "sample": str,
            "col_results": [(x, b12, b34, b456), ...],
            "global_median": [b12, b34, b456],
            "pix_info": {...},
            "mae_before": {name: float},
            "mae_after": {name: float},
        }
    """
    feat, _, col_xs, pix_info = extract_sample_data(sample, every_n=PREDICT_EVERY)
    if feat is None or len(feat) == 0:
        return None

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(feat).to(device)).cpu().numpy()

    # 按 x 排序
    xs_arr = np.array(col_xs)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    pred_sorted = pred[order]

    # 输出平滑: 对每类边界 depth 值应用中值滤波
    b12_raw = np.clip(pred_sorted[:, 0], 0.02, 0.98)
    b34_raw = np.clip(pred_sorted[:, 1], 0.02, 0.98)
    b456_raw = np.clip(pred_sorted[:, 2], 0.02, 0.98)

    if OUTPUT_SMOOTH_WINDOW > 1 and len(b12_raw) > OUTPUT_SMOOTH_WINDOW:
        b12_s = median_filter(b12_raw, size=OUTPUT_SMOOTH_WINDOW)
        b34_s = median_filter(b34_raw, size=OUTPUT_SMOOTH_WINDOW)
        b456_s = median_filter(b456_raw, size=OUTPUT_SMOOTH_WINDOW)
    else:
        b12_s, b34_s, b456_s = b12_raw, b34_raw, b456_raw

    # 单调性 + 收集结果
    col_results = []
    b12s, b34s, b456s = [], [], []
    for i, x in enumerate(xs_sorted):
        depths = sorted([float(b12_s[i]), float(b34_s[i]), float(b456_s[i])])
        col_results.append((x, depths[0], depths[1], depths[2]))
        b12s.append(depths[0])
        b34s.append(depths[1])
        b456s.append(depths[2])

    global_med = [
        float(np.median(b12s)),
        float(np.median(b34s)),
        float(np.median(b456s)),
    ]
    for i in range(1, 3):
        if global_med[i] <= global_med[i - 1]:
            global_med[i] = global_med[i - 1] + 0.02

    # ── 评估 MAE (紧凑像素) ──
    gt_path = LABEL_DIR / sample / "label_boundaries.csv"
    gt_groups = _extract_gt_boundaries(gt_path) if gt_path.exists() else {}

    pia_lut = pix_info["pia_lut"]
    white_lut = pix_info["white_lut"]
    scale = pix_info["compact_scale"]
    gb12, gb34, gb456 = pix_info["global_seeds"]

    # 深度转紧凑 y
    def _depth_to_y(x, d):
        py, wy = pia_lut(x), white_lut(x)
        if np.isfinite(py) and np.isfinite(wy):
            return py + d * (wy - py)
        return np.nan

    # 构造精炼前后边界点 (紧凑分辨率)
    internal_names = ["L1_2", "L3_4", "L4_5"]
    seeds = [gb12, gb34, gb456]
    mae_before, mae_after = {}, {}

    for idx, (name, seed) in enumerate(zip(internal_names, seeds)):
        before_pts, after_pts = [], []
        for x, *depths in col_results:
            y_before = _depth_to_y(x, seed)
            y_after = _depth_to_y(x, depths[idx])
            if np.isfinite(y_before):
                before_pts.append((float(x), float(y_before)))
            if np.isfinite(y_after):
                after_pts.append((float(x), float(y_after)))
        gt_pts = gt_groups.get(name, np.empty((0, 2))) if gt_groups else np.empty((0, 2))
        # GT 也缩放到紧凑分辨率
        if len(gt_pts) > 0:
            gt_pts = gt_pts.copy()
            gt_pts[:, 0] = gt_pts[:, 0] * scale
            gt_pts[:, 1] = gt_pts[:, 1] * scale

        mae_before[name] = _boundary_mae(np.array(before_pts), gt_pts)
        mae_after[name] = _boundary_mae(np.array(after_pts), gt_pts)

    return {
        "sample": sample,
        "col_results": col_results,
        "global_median": global_med,
        "pix_info": pix_info,
        "mae_before": mae_before,
        "mae_after": mae_after,
    }


# ═══════════════════════════════════════════════════════════════════════
#  输出与可视化
# ═══════════════════════════════════════════════════════════════════════


def save_refined_csv(result: dict, output_dir: Path):
    """保存精炼边界为 CSV。"""
    out_dir = OUTPUTS_DIR / result["sample"]
    bnd = pd.read_csv(out_dir / "all_boundaries.csv")

    # 用精炼结果替换内部边界
    pia_df = bnd[bnd["boundary"] == "pia"]
    white_df = bnd[bnd["boundary"] == "white"]
    pia_lut = _build_y_lookup(pia_df[["x", "y"]].to_numpy())
    white_lut = _build_y_lookup(white_df[["x", "y"]].to_numpy())
    scale = result["pix_info"]["compact_scale"]

    out_rows = []
    for _, row in pia_df.iterrows():
        out_rows.append({"x": row["x"], "y": row["y"], "boundary": "pia"})
    for _, row in white_df.iterrows():
        out_rows.append({"x": row["x"], "y": row["y"], "boundary": "white"})

    internal_names = ["L1_2", "L3_4", "L4_5"]
    for idx, name in enumerate(internal_names):
        for x, *depths in result["col_results"]:
            d = depths[idx]
            # 全分辨率 y
            py = pia_lut(x / scale) if scale > 0 else 0
            wy = white_lut(x / scale) if scale > 0 else 0
            if np.isfinite(py) and np.isfinite(wy):
                y = py + d * (wy - py)
                out_rows.append({"x": round(x / scale, 1) if scale > 0 else x,
                                 "y": round(y, 1),
                                 "boundary": name})

    out_df = pd.DataFrame(out_rows)
    out_path = output_dir / "all_boundaries_nn.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  -> {out_path}")
    return out_path


def save_comparison_plot(result: dict, save_dir: Path):
    """保存精炼前后 vs GT 对比图。"""
    sample = result["sample"]
    pix_info = result["pix_info"]
    pia_lut = pix_info["pia_lut"]
    white_lut = pix_info["white_lut"]
    seeds = pix_info["global_seeds"]
    scale = pix_info["compact_scale"]

    gt_path = LABEL_DIR / sample / "label_boundaries.csv"
    gt_groups = _extract_gt_boundaries(gt_path) if gt_path.exists() else {}
    # GT 缩放到紧凑
    for name in gt_groups:
        gt_groups[name] = gt_groups[name].copy()
        gt_groups[name][:, 0] *= scale
        gt_groups[name][:, 1] *= scale

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    internal_names = ["L1_2", "L3_4", "L4_5"]
    titles = ["L1/L2 Boundary", "L2-L3/L4 Boundary", "L4/L5 Boundary"]
    colors = {"before": "orange", "after": "green", "gt": "blue", "global": "red"}

    for idx, name in enumerate(internal_names):
        ax = axes[idx]
        col_xs = [r[0] for r in result["col_results"]]

        # 精炼前 (全局种子)
        before = []
        for x in col_xs:
            py, wy = pia_lut(x), white_lut(x)
            if np.isfinite(py) and np.isfinite(wy):
                before.append((x, py + seeds[idx] * (wy - py)))
        before_arr = np.array(before)

        # 精炼后
        after = [(r[0], _depth_to_y_g(pia_lut(r[0]), white_lut(r[0]), r[1 + idx]))
                 for r in result["col_results"]]
        after_arr = np.array([(x, y) for x, y in after if np.isfinite(y)])

        # GT
        gt_arr = gt_groups.get(name, np.empty((0, 2)))

        if len(before_arr) > 0:
            ax.plot(before_arr[:, 0], before_arr[:, 1], "-",
                    color=colors["before"], alpha=0.6, lw=1.5,
                    label=f"Coarse (MAE={result['mae_before'].get(name, 0):.1f}px)")
        if len(after_arr) > 0:
            ax.plot(after_arr[:, 0], after_arr[:, 1], "-",
                    color=colors["after"], alpha=0.8, lw=1.5,
                    label=f"NN (MAE={result['mae_after'].get(name, 0):.1f}px)")
        if len(gt_arr) > 0:
            ax.plot(gt_arr[:, 0], gt_arr[:, 1], "--",
                    color=colors["gt"], alpha=0.7, lw=1.5, label="GT")

        ax.set_title(f"{titles[idx]} — Sample {sample}", fontsize=12)
        ax.set_xlabel("X (compact pixel)", fontsize=10)
        ax.set_ylabel("Y (compact pixel)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Sample {sample} — NN Boundary Refinement", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "comparison_nn.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _depth_to_y_g(pia_y, white_y, d):
    if np.isfinite(pia_y) and np.isfinite(white_y):
        return pia_y + d * (white_y - pia_y)
    return np.nan


# ═══════════════════════════════════════════════════════════════════════
#  留一法交叉验证
# ═══════════════════════════════════════════════════════════════════════


def run_leave_one_out(samples: list[str], device: str = "cpu"):
    """留一法交叉验证: 每次用一个样本做测试，其余训练。"""
    print(f"\n{'=' * 60}")
    print(f"留一法交叉验证: {len(samples)} 个样本")
    print(f"{'=' * 60}")

    all_results = []
    summary_rows = []

    for test_sample in samples:
        train_samples = [s for s in samples if s != test_sample]
        print(f"\n{'─' * 50}")
        print(f"测试样本: {test_sample}, 训练样本: {train_samples}")
        print(f"{'─' * 50}")

        # ── 准备训练数据 ──
        train_feats, train_tgts = [], []
        for s in train_samples:
            feat, tgt, _, _ = extract_sample_data(s, every_n=TRAIN_SAMPLE_EVERY)
            if feat is not None and len(feat) > 0:
                train_feats.append(feat)
                train_tgts.append(tgt)
                print(f"  {s}: {len(feat)} 列")

        if len(train_feats) == 0:
            print(f"  [错误] 无训练数据!")
            continue

        X_train = np.vstack(train_feats)
        y_train = np.vstack(train_tgts)
        print(f"  训练集: {len(X_train)} 列 × {X_train.shape[1]} 特征")

        # ── 训练 ──
        model = train_model(X_train, y_train, device=device)
        save_model(model, MODEL_DIR / f"loov_train_on_{'_'.join(train_samples)}_test_on_{test_sample}.pt")
        print(f"  模型训练完成")

        # ── 预测测试样本 ──
        result = predict_sample(model, test_sample, device=device)
        if result is None:
            print(f"  [跳过] {test_sample}: 预测失败")
            continue

        # 保存结果
        save_refined_csv(result, OUTPUTS_DIR / test_sample)
        save_comparison_plot(result, SUMMARY_DIR / test_sample)
        all_results.append(result)

        # 打印摘要
        print(f"\n  样本 {test_sample} 精炼摘要:")
        for name in ["L1_2", "L3_4", "L4_5"]:
            mb = result["mae_before"].get(name, np.nan)
            ma = result["mae_after"].get(name, np.nan)
            impr = (mb - ma) / max(mb, 1e-8) * 100 if np.isfinite(mb) and mb > 0 else 0
            print(f"    {name}: {mb:.1f}→{ma:.1f}px ({impr:+.0f}%)")
            summary_rows.append({
                "test_sample": test_sample,
                "boundary": name,
                "mae_before": mb,
                "mae_after": ma,
                "improvement_pct": impr,
            })

    # ── 汇总 ──
    return all_results, pd.DataFrame(summary_rows)


def save_summary(summary_df: pd.DataFrame):
    """保存汇总统计和可视化。"""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = SUMMARY_DIR / "nn_refinement_summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表: {csv_path}")

    # 各层汇总
    print(f"\n{'=' * 60}")
    print("各层汇总 (MAE 像素, 紧凑分辨率)")
    print(f"{'=' * 60}")
    for name in ["L1_2", "L3_4", "L4_5"]:
        sub = summary_df[summary_df["boundary"] == name]
        if len(sub) == 0:
            continue
        mb = sub["mae_before"].mean()
        ma = sub["mae_after"].mean()
        impr = (mb - ma) / max(mb, 1e-8) * 100
        print(f"  {name}: {mb:.1f}→{ma:.1f}px ({impr:+.0f}%)")

    mb_all = summary_df["mae_before"].mean()
    ma_all = summary_df["mae_after"].mean()
    impr_all = (mb_all - ma_all) / max(mb_all, 1e-8) * 100
    print(f"  总体: {mb_all:.1f}→{ma_all:.1f}px ({impr_all:+.0f}%)")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 样本-边界 对比柱状图
    ax = axes[0]
    plot_df = summary_df.copy()
    x = np.arange(len(plot_df))
    w = 0.35
    ax.bar(x - w / 2, plot_df["mae_before"], w, label="Coarse", alpha=0.7, color="orange")
    ax.bar(x + w / 2, plot_df["mae_after"], w, label="NN Refined", alpha=0.7, color="green")
    labels = [f"{r.test_sample}-{r.boundary.replace('_','')}"
              for _, r in plot_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel("MAE (px)", fontsize=11)
    ax.set_title("Boundary Error: Coarse vs NN Refined", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 右: 改进率热力图
    ax = axes[1]
    pivot = plot_df.pivot_table(index="test_sample", columns="boundary",
                                 values="improvement_pct", aggfunc="mean")
    if not pivot.empty:
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=60)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=9, color="black" if abs(v) < 20 else "white")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title("Improvement Rate (%)", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig_path = SUMMARY_DIR / "nn_refinement_summary.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"汇总图: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="神经网络逐列层边界精炼 (PyTorch MLP)"
    )
    parser.add_argument("--samples", nargs="*", default=None,
                        help="指定样本 (默认: 所有)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 仅 3 折验证 (样本 1-8 训练, 9-10 验证, 11-12 测试)")
    parser.add_argument("--device", default="auto",
                        help='训练设备: "cpu", "cuda", "auto" (默认: auto)')
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"最大训练 epoch (默认: {MAX_EPOCHS})")
    parser.add_argument("--smooth-window", type=int, default=OUTPUT_SMOOTH_WINDOW,
                        help=f"输出平滑中值滤波窗口 (默认: {OUTPUT_SMOOTH_WINDOW}, 0=关闭)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 样本列表
    if args.samples:
        samples = args.samples
    else:
        samples = sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()])
        samples = [s for s in samples if (OUTPUTS_DIR / s / "all_boundaries.csv").exists()
                   and (LABEL_DIR / s / "label_boundaries.csv").exists()]

    if not samples:
        print("没有找到可用样本!")
        sys.exit(1)
    print(f"样本: {samples} ({len(samples)} 个)")

    if args.quick:
        # 快速模式: 简单训练/验证/测试划分
        print(f"\n快速模式: 训练={samples[:6]}, 验证={samples[6:8]}, 测试={samples[8:]}")
        train_s, val_s, test_s = samples[:6], samples[6:8], samples[8:]

        # 收集训练+验证数据
        train_feats, train_tgts = [], []
        for s in train_s + val_s:
            feat, tgt, _, _ = extract_sample_data(s, every_n=TRAIN_SAMPLE_EVERY)
            if feat is not None:
                train_feats.append(feat)
                train_tgts.append(tgt)

        X_train = np.vstack(train_feats)
        y_train = np.vstack(train_tgts)
        print(f"训练集: {len(X_train)} 列")

        model = train_model(X_train, y_train, device=device)
        model_tag = "_".join(train_s + val_s)
        save_model(model, MODEL_DIR / f"quick_{model_tag}.pt")

        all_results = []
        summary_rows = []
        for s in test_s:
            result = predict_sample(model, s, device=device)
            if result:
                save_refined_csv(result, OUTPUTS_DIR / s)
                save_comparison_plot(result, SUMMARY_DIR / s)
                all_results.append(result)
                for name in ["L1_2", "L3_4", "L4_5"]:
                    mb = result["mae_before"].get(name, np.nan)
                    ma = result["mae_after"].get(name, np.nan)
                    impr = (mb - ma) / max(mb, 1e-8) * 100 if np.isfinite(mb) and mb > 0 else 0
                    summary_rows.append({
                        "test_sample": s, "boundary": name,
                        "mae_before": mb, "mae_after": ma, "improvement_pct": impr,
                    })
                    print(f"  {s} {name}: {mb:.1f}→{ma:.1f}px ({impr:+.0f}%)")

        save_summary(pd.DataFrame(summary_rows))
    else:
        # 留一法交叉验证
        results, summary_df = run_leave_one_out(samples, device=device)
        if len(results) > 0:
            save_summary(summary_df)

    print(f"\n所有结果已保存到 {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
