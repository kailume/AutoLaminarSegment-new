#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
形变式边界精炼 —— 学习粗分层线的局部弯曲/移动微调。

核心: 将粗边界线视为点序列，网络学习逐点 Δy 偏移以逼近 GT。
偏移受平滑性约束 (相邻点 Δy 连续)。

数据: 训练 1-10, 测试 11-12
特征: per-point KDE 密度 + 上下文 + 粗种子信息

用法:
  .\\venv-cellpose\\Scripts\\python src/refine_deform.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "dataset" / "outputs"
LABEL_DIR = ROOT / "dataset" / "inputs" / "label"
ANALYSIS_DIR = ROOT / "dataset" / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "refine_deform"
MODEL_DIR = ROOT / "models" / "refine_deform"

CELL_RADIUS = 30
KDE_MARGIN = 40
N_FEATURES = 12
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
SMOOTH_WEIGHT = 0.5


# ═══════════════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════════════


def _build_y_lookup(pts_array: np.ndarray):
    pts = pts_array[np.isfinite(pts_array).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.nan
    df = pd.DataFrame(pts, columns=["x", "y"])
    g = df.groupby("x", as_index=False)["y"].median().sort_values("x")
    xv, yv = g["x"].values.astype(float), g["y"].values.astype(float)
    if len(xv) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), yv[0])
    def _lut(x_new):
        return np.interp(np.asarray(x_new, dtype=float), xv, yv)
    return _lut


def _load_cells(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for dst, srcs in [("X", ["centroid_x", "x", "X"]), ("Y", ["centroid_y", "y", "Y"])]:
        for s in srcs:
            if s in df.columns:
                if dst not in df.columns or df.columns.get_loc(s) < df.columns.get_loc(dst) if dst in df.columns else True:
                    pass
    col_map = {}
    for src in ["centroid_x", "x", "X"]:
        if src in df.columns and "X" not in col_map:
            col_map[src] = "X"
    for src in ["centroid_y", "y", "Y"]:
        if src in df.columns and "Y" not in col_map:
            col_map[src] = "Y"
    if col_map:
        df = df.rename(columns=col_map)
    return df[["X", "Y"]].copy().astype(float)


def _kde_at(kde, x: float, y: float) -> float:
    """log-KDE 密度值。"""
    if kde is None:
        return 0.0
    try:
        v = float(kde.evaluate(np.array([[x], [y]]))[0])
        return float(np.log(max(v, 1e-12))) if v > 0 else -20.0
    except Exception:
        return -20.0


def _local_cell_count(cells: pd.DataFrame, x: float, y: float, r: float) -> int:
    return int(((cells["X"] - x) ** 2 + (cells["Y"] - y) ** 2 <= r ** 2).sum())


def _boundary_mae(pred_pts, gt_pts) -> float:
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    gt_lut = _build_y_lookup(gt_pts)
    xs = np.unique(pred_pts[:, 0].astype(int))
    err = []
    for x in xs:
        yp = np.median(pred_pts[pred_pts[:, 0].astype(int) == x, 1])
        yg = gt_lut(float(x))
        if np.isfinite(yg):
            err.append(abs(yp - yg))
    return float(np.mean(err)) if err else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  特征提取 (紧凑分辨率)
# ═══════════════════════════════════════════════════════════════════════


def extract_boundary_data(sample: str, every_n: int = 1):
    """提取边界点特征与 Δy 目标。

    全部工作在紧凑分辨率下 (y ≈ 0-3215)。
    返回: (features, targets, col_info) 或 (None, None, [])
    """
    out_dir = OUTPUTS_DIR / sample
    label_dir = LABEL_DIR / sample

    # 紧凑缩放
    cells_full = _load_cells(out_dir / "cell_centroids.csv")
    algo = cv2.imread(str(out_dir / "layers_color_mask.png"))
    compact_h = algo.shape[0] if algo is not None else 3215
    full_h = float(cells_full["Y"].max())
    cs = compact_h / full_h if full_h > 0 else 0.1

    cells = cells_full.copy()
    cells["X"] *= cs
    cells["Y"] *= cs

    # KDE
    coords = cells[["X", "Y"]].to_numpy(dtype=float).T
    kde = gaussian_kde(coords, bw_method="scott") if coords.shape[1] >= 10 else None
    print(f"  {sample}: {len(cells)} cells, scale={cs:.4f}")

    # 粗边界
    bnd = pd.read_csv(out_dir / "all_boundaries.csv")

    # GT → 紧凑
    gt_path = label_dir / "label_boundaries.csv"
    if not gt_path.exists():
        return None, None, []
    gt_groups = {}
    for nm, sub in pd.read_csv(gt_path).groupby("boundary"):
        arr = sub[["x", "y"]].to_numpy(dtype=float)
        arr[:, 0] *= cs
        arr[:, 1] *= cs
        gt_groups[nm] = arr

    seg = pd.read_csv(out_dir / "segmented_depth.csv")
    layers = seg.to_dict("records")
    seed_depths = {"L1_2": float(layers[0]["end"]),
                   "L3_4": float(layers[1]["end"]),
                   "L4_5": float(layers[2]["end"])}

    internal_names = ["L1_2", "L3_4", "L4_5"]
    features, targets, col_info = [], [], []

    for bname in internal_names:
        sub = bnd[bnd["boundary"] == bname]
        if len(sub) == 0:
            continue
        pts = sub[["x", "y"]].to_numpy(dtype=float)
        # 每 x 中位数 y
        x_u = np.sort(np.unique(pts[:, 0].astype(int)))
        y_m = np.array([np.median(pts[pts[:, 0].astype(int) == xi, 1]) for xi in x_u])
        x_u = x_u[::every_n]
        y_m = y_m[::every_n]

        gt_lut = _build_y_lookup(gt_groups.get(bname, np.empty((0, 2))))
        seed = seed_depths[bname]
        btype = internal_names.index(bname) / 2.0

        total_n = float(np.sqrt(len(cells)))
        compact_w = float(np.max(x_u) + 1) if len(x_u) > 0 else 1.0

        for i, (x, y_c) in enumerate(zip(x_u, y_m)):
            y_gt = float(gt_lut(float(x)))
            if not np.isfinite(y_gt):
                continue
            delta_y = y_gt - y_c

            feat = np.array([
                float(x) / compact_w,          # 0: x 归一化
                float(y_c) / compact_h,        # 1: y 归一化
                _kde_at(kde, x, y_c),           # 2: KDE
                _kde_at(kde, x, y_c - KDE_MARGIN),  # 3: KDE up
                _kde_at(kde, x, y_c + KDE_MARGIN),  # 4: KDE down
                0.0,                            # 5: 密度梯度 (填充)
                float(_local_cell_count(cells, x, y_c, CELL_RADIUS)) / 500.0,  # 6
                float(y_m[max(0, i - 2)]) / compact_h,  # 7: 左邻居
                float(y_m[min(len(y_m) - 1, i + 2)]) / compact_h,  # 8: 右邻居
                seed,                           # 9: 粗深度
                btype,                          # 10: 边界类型
                total_n / 200.0,                # 11: sqrt 细胞数
            ], dtype=np.float32)
            features.append(feat)
            targets.append(delta_y)
            col_info.append((sample, bname, int(x), y_c, y_gt))

    if len(features) < 10:
        return None, None, []
    return np.array(features), np.array(targets, dtype=np.float32), col_info


# ═══════════════════════════════════════════════════════════════════════
#  模型
# ═══════════════════════════════════════════════════════════════════════


class MLPOffset(nn.Module):
    """MLP: 每点独立预测 Δy。"""
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResMLP(nn.Module):
    """残差 MLP: 学习 Δy + 跳跃连接保平滑。"""
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BoundaryCNN(nn.Module):
    """1D CNN: 以整个边界序列为单位, 卷积保证平滑性。

    注意: 训练时需用 SequenceDataset, 当前仅用于测试。
    """
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1),
        )
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(self.cnn[-1].weight)
        nn.init.zeros_(self.cnn[-1].bias)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        return self.cnn(x.transpose(1, 2)).squeeze(1)  # -> (batch, seq_len)


# ═══════════════════════════════════════════════════════════════════════
#  数据集
# ═══════════════════════════════════════════════════════════════════════


class BoundaryPointDataset(Dataset):
    def __init__(self, f: np.ndarray, t: np.ndarray):
        self.f = torch.from_numpy(f)
        self.t = torch.from_numpy(t)
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.f[i], self.t[i]


# ═══════════════════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════════════════


def smoothness_loss_fn(pred, alpha=0.5):
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    return alpha * torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)


def train_model(model, train_feat, train_tgt, val_feat=None, val_tgt=None, device="cpu"):
    train_loader = DataLoader(BoundaryPointDataset(train_feat, train_tgt),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = None
    if val_feat is not None and len(val_feat) > 0:
        val_loader = DataLoader(BoundaryPointDataset(val_feat, val_tgt),
                                 batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE // 3)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            p = model(xb)
            loss = nn.functional.l1_loss(p, yb) + smoothness_loss_fn(p, SMOOTH_WEIGHT)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(xb)
        tl /= len(train_loader.dataset)

        if val_loader:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += nn.functional.l1_loss(model(xb), yb).item() * len(xb)
            vl /= len(val_loader.dataset)
            scheduler.step(vl)
            cl = vl
        else:
            cl = tl
            scheduler.step(tl)

        if cl < best_loss:
            best_loss = cl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ═══════════════════════════════════════════════════════════════════════
#  预测
# ═══════════════════════════════════════════════════════════════════════


def predict_sample(model, sample: str, smooth_output: int = 11, device="cpu"):
    """边界精炼预测。"""
    out_dir = OUTPUTS_DIR / sample
    label_dir = LABEL_DIR / sample

    cells_full = _load_cells(out_dir / "cell_centroids.csv")
    algo = cv2.imread(str(out_dir / "layers_color_mask.png"))
    compact_h = algo.shape[0] if algo is not None else 3215
    cs = compact_h / float(cells_full["Y"].max()) if cells_full["Y"].max() > 0 else 0.1
    cells = cells_full.copy()
    cells["X"] *= cs
    cells["Y"] *= cs
    coords = cells[["X", "Y"]].to_numpy(dtype=float).T
    kde = gaussian_kde(coords, bw_method="scott") if coords.shape[1] >= 10 else None

    bnd = pd.read_csv(out_dir / "all_boundaries.csv")
    seg = pd.read_csv(out_dir / "segmented_depth.csv")
    seed_depths = {"L1_2": float(seg.iloc[0]["end"]),
                   "L3_4": float(seg.iloc[1]["end"]),
                   "L4_5": float(seg.iloc[2]["end"])}

    gt_path = label_dir / "label_boundaries.csv"
    gt_groups = {}
    if gt_path.exists():
        for nm, sub in pd.read_csv(gt_path).groupby("boundary"):
            arr = sub[["x", "y"]].to_numpy(dtype=float)
            arr[:, 0] *= cs
            arr[:, 1] *= cs
            gt_groups[nm] = arr

    internal_names = ["L1_2", "L3_4", "L4_5"]
    boundaries, mae_before, mae_after = {}, {}, {}

    for bname in internal_names:
        sub = bnd[bnd["boundary"] == bname]
        if len(sub) == 0:
            continue
        pts = sub[["x", "y"]].to_numpy(dtype=float)
        x_u = np.sort(np.unique(pts[:, 0].astype(int)))
        y_m = np.array([np.median(pts[pts[:, 0].astype(int) == xi, 1]) for xi in x_u])

        feats = []
        for i, (x, y_c) in enumerate(zip(x_u, y_m)):
            feats.append(np.array([
                float(x) / float(max(x_u[-1], 1)),
                float(y_c) / compact_h,
                _kde_at(kde, x, y_c),
                _kde_at(kde, x, y_c - KDE_MARGIN),
                _kde_at(kde, x, y_c + KDE_MARGIN),
                0.0,
                float(_local_cell_count(cells, x, y_c, CELL_RADIUS)) / 500.0,
                float(y_m[max(0, i - 2)]) / compact_h,
                float(y_m[min(len(y_m) - 1, i + 2)]) / compact_h,
                seed_depths[bname],
                internal_names.index(bname) / 2.0,
                float(np.sqrt(len(cells))) / 200.0,
            ], dtype=np.float32))

        X = np.array(feats)
        model.eval()
        with torch.no_grad():
            deltas = model(torch.from_numpy(X).to(device)).cpu().numpy()

        if smooth_output > 1 and len(deltas) > smooth_output:
            deltas = median_filter(deltas, size=smooth_output)

        y_ref = y_m + deltas
        bounds = np.column_stack([x_u, y_ref])
        boundaries[bname] = bounds

        coarse_arr = np.column_stack([x_u, y_m])
        gt_pts = gt_groups.get(bname, np.empty((0, 2)))
        mae_before[bname] = _boundary_mae(coarse_arr, gt_pts)
        mae_after[bname] = _boundary_mae(bounds, gt_pts)

    return {"sample": sample, "boundaries": boundaries,
            "mae_before": mae_before, "mae_after": mae_after}


# ═══════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════


def run_experiment(train_samples, test_samples, model_cls, model_name, device="cpu"):
    print(f"\n{'=' * 60}")
    print(f"模型: {model_name}")
    print(f"训练: {train_samples}, 测试: {test_samples}")
    print(f"{'=' * 60}")

    all_feats, all_tgts = [], []
    for s in train_samples:
        feat, tgt, _ = extract_boundary_data(s, every_n=3)
        if feat is not None:
            all_feats.append(feat)
            all_tgts.append(tgt)
            print(f"  {s}: {len(feat)} 点")
    if not all_feats:
        return None
    X_train = np.vstack(all_feats)
    y_train = np.concatenate(all_tgts)
    print(f"  训练集: {len(X_train)} 点 × {X_train.shape[1]} 特征")

    model = model_cls().to(device)
    model = train_model(model, X_train, y_train, device=device)

    tag = "_".join(train_samples[:3]) + f"_{model_name}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_name": model_name,
                 "n_features": X_train.shape[1]}, MODEL_DIR / f"{tag}.pt")
    print(f"  模型: {MODEL_DIR / f'{tag}.pt'}")

    results = []
    for s in test_samples:
        r = predict_sample(model, s, device=device)
        if r:
            results.append(r)
            print(f"\n  {s}:")
            for b in ["L1_2", "L3_4", "L4_5"]:
                mb, ma = r["mae_before"].get(b, np.nan), r["mae_after"].get(b, np.nan)
                impr = (mb - ma) / max(mb, 1e-8) * 100 if np.isfinite(mb) and mb > 0 else 0
                print(f"    {b}: {mb:.1f}→{ma:.1f}px ({impr:+.0f}%)")
    return results


def save_results(results, model_name):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        out_dir = OUTPUTS_DIR / r["sample"]
        bnd = pd.read_csv(out_dir / "all_boundaries.csv")
        rows = []
        for b in ["pia", "white"]:
            for _, row in bnd[bnd["boundary"] == b].iterrows():
                rows.append({"x": row["x"], "y": row["y"], "boundary": b})
        for b in ["L1_2", "L3_4", "L4_5"]:
            pts = r["boundaries"].get(b)
            if pts is not None:
                for x, y in pts:
                    rows.append({"x": round(x, 1), "y": round(y, 1), "boundary": b})
        pd.DataFrame(rows).to_csv(out_dir / f"all_boundaries_{model_name}.csv", index=False)

        # 对比图
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        for idx, b in enumerate(["L1_2", "L3_4", "L4_5"]):
            ax = axes[idx]
            pts = r["boundaries"].get(b)
            if pts is None:
                continue
            bnd_f = pd.read_csv(out_dir / "all_boundaries.csv")
            cp = bnd_f[bnd_f["boundary"] == b][["x", "y"]].to_numpy(dtype=float)
            gt = pd.read_csv(LABEL_DIR / r["sample"] / "label_boundaries.csv")
            gp = gt[gt["boundary"] == b][["x", "y"]].to_numpy(dtype=float)
            ax.plot(cp[:, 0], cp[:, 1], ".", color="orange", alpha=0.3, ms=1,
                    label=f"Coarse ({r['mae_before'].get(b,0):.1f}px)")
            ax.plot(pts[:, 0], pts[:, 1], "-", color="green", lw=1.5,
                    label=f"{model_name} ({r['mae_after'].get(b,0):.1f}px)")
            ax.plot(gp[:, 0], gp[:, 1], "--", color="blue", alpha=0.7, lw=1.5, label="GT")
            ax.set_title(f"{b} — Sample {r['sample']}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(SUMMARY_DIR / f"{r['sample']}_{model_name}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="形变式边界精炼")
    parser.add_argument("--train", nargs="*", default=None)
    parser.add_argument("--test", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["mlp", "resmlp"],
                        help="mlp / resmlp / cnn")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    print(f"设备: {device}")

    all_s = sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()],
                   key=lambda s: int(s) if s.isdigit() else s)
    all_s = [s for s in all_s
             if (OUTPUTS_DIR / s / "all_boundaries.csv").exists()
             and (LABEL_DIR / s / "label_boundaries.csv").exists()]
    train_s = args.train if args.train else all_s[:10]
    test_s = args.test if args.test else all_s[10:12]
    print(f"训练: {train_s}, 测试: {test_s}")

    model_registry = {"mlp": (MLPOffset, "mlp"), "resmlp": (ResMLP, "resmlp"),
                      "cnn": (BoundaryCNN, "cnn")}

    summary_rows = []
    for mname in args.models:
        if mname not in model_registry:
            print(f"未知模型: {mname}, 跳过")
            continue
        cls, tag = model_registry[mname]
        results = run_experiment(train_s, test_s, cls, tag, device)
        if results:
            save_results(results, tag)
            for r in results:
                for b in ["L1_2", "L3_4", "L4_5"]:
                    mb, ma = r["mae_before"].get(b, np.nan), r["mae_after"].get(b, np.nan)
                    impr = (mb - ma) / max(mb, 1e-8) * 100 if np.isfinite(mb) and mb > 0 else 0
                    summary_rows.append({"model": tag, "sample": r["sample"],
                        "boundary": b, "mae_before": mb, "mae_after": ma, "improvement": impr})

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(SUMMARY_DIR / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n汇总: {SUMMARY_DIR / 'summary.csv'}")
        for m in args.models:
            sub = df[df["model"] == m]
            if len(sub):
                print(f"\n{m}:")
                for b in ["L1_2", "L3_4", "L4_5"]:
                    s = sub[sub["boundary"] == b]
                    if len(s):
                        mb, ma = s["mae_before"].mean(), s["mae_after"].mean()
                        impr = (mb - ma) / max(mb, 1e-8) * 100
                        print(f"  {b}: {mb:.1f}→{ma:.1f}px ({impr:+.0f}%)")


if __name__ == "__main__":
    main()
