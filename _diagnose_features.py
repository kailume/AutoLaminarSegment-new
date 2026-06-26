#!/usr/bin/env python
"""验证模型是否真的在学习密度特征，还是在盲目偏移。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("dataset")
OUTPUTS_DIR = ROOT / "outputs"
LABEL_DIR = ROOT / "inputs" / "label"
MODEL_DIR = Path("models") / "refine_deform"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载最新模型
model_path = sorted(MODEL_DIR.glob("*.pt"))[-1]
print(f"加载模型: {model_path}")
state = torch.load(str(model_path), map_location=device, weights_only=True)

from src.refine_deform import ResMLP, N_FEATURES
model = ResMLP().to(device)
model.load_state_dict(state["model_state"])
model.eval()

# 2. 检查第一层权重: 36→128
w1 = model.net[0].weight.data.cpu().numpy()  # (128, 36)
print(f"\n第一层权重形状: {w1.shape}")

# 按特征组统计权重绝对值
parts = {
    "密度剖面(0-30)": w1[:, 0:31],
    "x位置(31)": w1[:, 31:32],
    "y位置(32)": w1[:, 32:33],
    "左邻y(33)": w1[:, 33:34],
    "右邻y(34)": w1[:, 34:35],
    "梯度(35)": w1[:, 35:36],
}
print(f"\n第一层权重各特征组平均绝对值:")
for name, w in parts.items():
    avg = np.mean(np.abs(w))
    print(f"  {name}: {avg:.4f}")

# 3. 生成特征重要性的简单估计: 每个特征列对输出的影响
# 用单位输入 + 逐特征扰动来测试
test_input = torch.zeros(1, N_FEATURES, device=device)
with torch.no_grad():
    baseline = model(test_input).item()

sensitivities = {}
for i in range(N_FEATURES):
    x = test_input.clone()
    x[0, i] = 1.0  # 单独激活该特征
    with torch.no_grad():
        out = model(x).item()
    sensitivities[i] = abs(out - baseline)

# 分组
groups = {"密度剖面(0-30)": list(range(31)),
          "x位置(31)": [31], "y位置(32)": [32],
          "左邻y(33)": [33], "右邻y(34)": [34], "梯度(35)": [35]}
print(f"\n逐特征灵敏度 (单位激励对输出Δy的影响):")
for name, idxs in groups.items():
    avg = np.mean([sensitivities[i] for i in idxs])
    print(f"  {name}: {avg:.4f} px")

# 4. 实际数据验证: 取几个边界点, 分别用完整特征和仅位置特征预测
print(f"\n{'='*60}")
print(f"实际数据验证: 比较完整特征 vs 仅位置特征")
print(f"{'='*60}")

from src.refine_deform import _load_cells, _get_compact_scale, _build_density_grid
from src.refine_deform import _make_feat, _profile_fast

for sample in ["1", "11"]:
    out_dir = OUTPUTS_DIR / sample
    cf = _load_cells(out_dir / "cell_centroids.csv")
    cs = _get_compact_scale(out_dir, cf)
    cells = cf.copy(); cells["X"] *= cs; cells["Y"] *= cs
    grid, gs = _build_density_grid(cells)

    import pandas as pd
    bnd = pd.read_csv(out_dir / "all_boundaries.csv" if (out_dir / "all_boundaries.csv").exists()
                      else out_dir / "all_boundaries_v2.csv")
    seg = pd.read_csv(out_dir / "segmented_depth.csv")

    # 从 L3_4 边界随机取 5 个点
    sub = bnd[bnd["boundary"] == "L3_4"]
    pts = sub[["x","y"]].to_numpy(dtype=float)
    pts[:,0] *= cs; pts[:,1] *= cs
    xs = np.sort(np.unique(pts[:,0].astype(int)))
    ym = np.array([np.median(pts[pts[:,0].astype(int)==xi,1]) for xi in xs])

    idxs = np.linspace(0, len(xs)-1, 5, dtype=int)
    print(f"\n样本 {sample} L3_4 边界 (取5个位置):")
    for idx in idxs:
        x, y_c = xs[idx], ym[idx]
        # 完整特征
        feat_full = _make_feat(x, y_c, ym, idx, grid, gs, 3215, float(max(xs)+1))
        # 仅位置特征 (密度剖面置零)
        feat_pos = feat_full.copy()
        feat_pos[0:31] = 0.0

        with torch.no_grad():
            pred_full = model(torch.from_numpy(feat_full).unsqueeze(0).to(device)).item()
            pred_pos = model(torch.from_numpy(feat_pos).unsqueeze(0).to(device)).item()
        diff = pred_full - pred_pos
        print(f"  x={x:.0f} y={y_c:.0f}: 完整={pred_full:.1f}px, "
              f"仅位置={pred_pos:.1f}px, 差值={diff:.1f}px "
              f"{'✅' if abs(diff) > 2 else '⚠️'}")
