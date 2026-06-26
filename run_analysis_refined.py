#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粗分层 vs NN精炼 对比评估。

对每个样本，分别计算 coarse (layers_color_mask.png) 和
refined (layers_color_mask_refined.png) 相对于 GT 的各项指标，
生成逐样本、逐层、总体三个层次的粗/精对比。

用法:
  .\\venv-cellpose\\Scripts\\python run_analysis_refined.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "dataset" / "outputs"
LABEL_DIR = ROOT / "dataset" / "inputs" / "label"
ANALYSIS_DIR = ROOT / "dataset" / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "refined_comparison"
FIGURES_DIR = SUMMARY_DIR / "figures"

STANDARD_LAYERS = ["L1", "L2/3", "L4", "L5/6"]
PX_TO_UM = 0.1625
COLOR_TOL = 15

# 颜色映射 (BGR)
ALGO_COLORS = {
    "L1": (255, 100, 100), "L2/3": (100, 255, 100),
    "L4": (100, 100, 255), "L5/6": (255, 255, 100),
}
GT_COLORS_CANDIDATES = [
    {"L1": (255, 100, 100), "L2/3": (100, 255, 100),
     "L4": (100, 100, 255), "L5/6": (255, 255, 100)},
    {"L1": (255, 100, 100), "L2/3": (100, 255, 100),
     "L4": (100, 100, 255), "L5/6": (255, 100, 255)},
    {"L1": (25, 28, 252), "L2/3": (18, 126, 126),
     "L4": (255, 255, 76), "L5/6": (127, 127, 248)},
]

# ══════════════════════════════════════════════════════════════════════
#  工具函数 (复用 run_analysis.py)
# ══════════════════════════════════════════════════════════════════════


def _extract_color_mask(img, color, tol=COLOR_TOL):
    diff = np.abs(img.astype(np.int16) - np.array(color, dtype=np.uint8))
    return np.all(diff <= tol, axis=2).astype(np.uint8) * 255


def _load_pil(path):
    PILImage.MAX_IMAGE_PIXELS = None
    arr = np.array(PILImage.open(str(path)).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _detect_layer_colors(mask):
    """自动检测 mask 中的层颜色, 按 y 排序映射到 L1/L2/3/L4/L5/6。"""
    pix = mask.reshape(-1, 3)
    colored = pix[np.any(pix > 10, axis=1)]
    if len(colored) == 0:
        return {}
    buckets = {}
    for px in colored:
        k = tuple((px // 20).tolist())
        buckets.setdefault(k, []).append(px)
    colors = [tuple(np.mean(v, axis=0).astype(np.uint8).tolist()) for v in buckets.values()]
    colors = sorted([c for c in colors if not all(v < 20 for v in c)])

    # 先尝试标准匹配
    for cmap in GT_COLORS_CANDIDATES:
        matched = {}
        ok = True
        for layer in STANDARD_LAYERS:
            m = _extract_color_mask(mask, cmap[layer], 15)
            if np.sum(m > 0) < 100:
                ok = False
                break
            matched[layer] = cmap[layer]
        if ok:
            return matched

    # fallback: 按 y 排序
    ycs = []
    for c in colors:
        m = _extract_color_mask(mask, c, 15)
        ys = np.where(m > 0)[0]
        if len(ys) > 0:
            ycs.append((c, ys.mean()))
    ycs.sort(key=lambda x: x[1])
    result = {}
    for i, (c, _) in enumerate(ycs[:4]):
        result[STANDARD_LAYERS[i]] = c
    for layer in STANDARD_LAYERS:
        if layer not in result:
            result[layer] = list(ALGO_COLORS.values())[STANDARD_LAYERS.index(layer)]
    return result


def _parse_layers(mask, color_map):
    layers = {}
    for layer, color in color_map.items():
        layers[layer] = _extract_color_mask(mask, color, COLOR_TOL)
    return layers


def calc_iou(m1, m2):
    i = np.logical_and(m1 > 0, m2 > 0).sum()
    u = np.logical_or(m1 > 0, m2 > 0).sum()
    return i / u if u > 0 else 0.0


def calc_dice(m1, m2):
    i = np.logical_and(m1 > 0, m2 > 0).sum()
    s = (m1 > 0).sum() + (m2 > 0).sum()
    return 2 * i / s if s > 0 else 0.0


def calc_pr(m1, m2):
    tp = np.logical_and(m1 > 0, m2 > 0).sum()
    fp = np.logical_and(m1 > 0, m2 == 0).sum()
    fn = np.logical_and(m1 == 0, m2 > 0).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r


def extract_boundaries(layers):
    valid = [m for m in layers.values() if m is not None]
    if not valid:
        return {}
    h, w = valid[0].shape[:2]
    bnds = {}
    for name, mask in layers.items():
        if mask is None:
            continue
        grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        tops, bots = [], []
        for x in range(w):
            col = grad[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                tops.append((x, int(ys.min())))
                bots.append((x, int(ys.max())))
        bnds[name] = {"top": np.array(tops) if tops else None,
                       "bottom": np.array(bots) if bots else None}
    return bnds


def boundary_dist(b1, b2):
    if b1 is None or b2 is None or len(b1) == 0 or len(b2) == 0:
        return None
    d1 = {p[0]: p[1] for p in b1}
    d2 = {p[0]: p[1] for p in b2}
    common = set(d1.keys()) & set(d2.keys())
    if not common:
        return None
    return float(np.mean([abs(d1[x] - d2[x]) for x in common]))


def calc_thickness(mask):
    h, w = mask.shape[:2]
    thicks = []
    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            thicks.append(int(ys.max() - ys.min() + 1))
    return np.array(thicks) if thicks else None


# ══════════════════════════════════════════════════════════════════════
#  单样本分析
# ══════════════════════════════════════════════════════════════════════


def analyze_one(mask, gt_layers, gt_colors, tag=""):
    """分析一个 mask 相对于 GT 的指标。"""
    layer_colors = _detect_layer_colors(mask)
    layers = _parse_layers(mask, layer_colors)
    results = []
    for layer in STANDARD_LAYERS:
        a = layers.get(layer)
        g = gt_layers.get(layer)
        if a is None or g is None:
            results.append({"layer": layer, "status": "missing"})
            continue
        results.append({
            "layer": layer, "status": "ok",
            "iou": calc_iou(a, g), "dice": calc_dice(a, g),
            "precision": calc_pr(a, g)[0], "recall": calc_pr(a, g)[1],
        })
    df = pd.DataFrame(results)

    # 边界误差
    ab = extract_boundaries(layers)
    gb = extract_boundaries(gt_layers)
    b_rows = []
    for layer in STANDARD_LAYERS:
        t = boundary_dist(ab.get(layer, {}).get("top"),
                          gb.get(layer, {}).get("top"))
        b = boundary_dist(ab.get(layer, {}).get("bottom"),
                          gb.get(layer, {}).get("bottom"))
        avg = np.mean([v for v in [t, b] if v is not None]) if any(v is not None for v in [t, b]) else None
        b_rows.append({"layer": layer, "top_boundary_error_px": t,
                        "bottom_boundary_error_px": b, "avg_boundary_error_px": avg})
    bdf = pd.DataFrame(b_rows)

    # 厚度误差
    t_rows = []
    for layer in STANDARD_LAYERS:
        a = layers.get(layer)
        g = gt_layers.get(layer)
        if a is None or g is None:
            continue
        at = calc_thickness(a)
        gt = calc_thickness(g)
        if at is None or gt is None or len(at) == 0 or len(gt) == 0:
            continue
        am, gm = float(np.mean(at)), float(np.mean(gt))
        err = abs(am - gm)
        t_rows.append({"layer": layer, "algo_mean_thickness_px": round(am, 2),
                        "gt_mean_thickness_px": round(gm, 2),
                        "thickness_error_px": round(err, 2),
                        "thickness_error_percent": round(err / gm * 100, 2) if gm > 0 else 0})
    tdf = pd.DataFrame(t_rows)

    result = df.merge(bdf, on="layer", how="outer").merge(tdf, on="layer", how="outer")
    # 微米制
    for c in ["top_boundary_error_px", "bottom_boundary_error_px", "avg_boundary_error_px",
              "algo_mean_thickness_px", "gt_mean_thickness_px", "thickness_error_px"]:
        if c in result.columns and result[c].notna().any():
            result[c.replace("_px", "_um")] = result[c].apply(
                lambda x: round(x * PX_TO_UM, 3) if pd.notna(x) else None)
    return result, layers


def analyze_sample(sample):
    """分析单个样本: coarse vs refined vs GT。"""
    out_dir = OUTPUTS_DIR / sample
    label_dir = LABEL_DIR / sample
    save_dir = ANALYSIS_DIR / sample
    save_dir.mkdir(parents=True, exist_ok=True)

    coarse_path = out_dir / "layers_color_mask.png"
    refined_path = out_dir / "layers_color_mask_refined.png"
    gt_path = label_dir / "label_mask.png"

    if not coarse_path.exists() or not gt_path.exists():
        print(f"  [跳过] {sample}")
        return None

    print(f"\n{'=' * 60}")
    print(f"样本: {sample}")
    print(f"{'=' * 60}")

    # 加载
    gt_raw = _load_pil(gt_path)
    coarse_raw = _load_pil(coarse_path)
    refined_raw = _load_pil(refined_path) if refined_path.exists() else None

    # 统一尺寸
    if gt_raw.shape[:2] != coarse_raw.shape[:2]:
        gt_raw = cv2.resize(gt_raw, (coarse_raw.shape[1], coarse_raw.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    if refined_raw is not None and refined_raw.shape[:2] != coarse_raw.shape[:2]:
        refined_raw = cv2.resize(refined_raw, (coarse_raw.shape[1], coarse_raw.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    # GT 层
    gt_colors = _detect_layer_colors(gt_raw)
    gt_layers = _parse_layers(gt_raw, gt_colors)

    # 分析 coarse
    coarse_result, coarse_layers = analyze_one(coarse_raw, gt_layers, gt_colors, "coarse")
    coarse_result.insert(0, "type", "coarse")
    coarse_result.insert(0, "sample", sample)

    # 分析 refined
    if refined_raw is not None:
        refined_result, refined_layers = analyze_one(refined_raw, gt_layers, gt_colors, "refined")
        refined_result.insert(0, "type", "refined")
        refined_result.insert(0, "sample", sample)
    else:
        refined_result = None

    # 合并
    combined = pd.concat([coarse_result] + ([refined_result] if refined_result is not None else []),
                          ignore_index=True)

    # 保存
    csv_path = save_dir / "analysis_comparison.csv"
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> {csv_path}")

    # 打印
    for typ in ["coarse", "refined"]:
        sub = combined[combined["type"] == typ]
        if len(sub) == 0:
            continue
        print(f"  [{typ}]")
        for _, r in sub.iterrows():
            iou = r.get("iou", None)
            dice = r.get("dice", None)
            be = r.get("avg_boundary_error_px", None)
            te = r.get("thickness_error_percent", None)
            iou_s = f"{iou:.4f}" if pd.notna(iou) else "-"
            dice_s = f"{dice:.4f}" if pd.notna(dice) else "-"
            be_s = f"{be:.1f}px" if pd.notna(be) else "-"
            te_s = f"{te:.1f}%" if pd.notna(te) else "-"
            print(f"    {r['layer']}: IoU={iou_s} Dice={dice_s} Bnd={be_s} Thk={te_s}")

    # 对比图
    _plot_comparison(coarse_raw, refined_raw, gt_raw, sample, save_dir)

    return combined


def _plot_comparison(coarse, refined, gt, sample, save_dir):
    """Coarse / Refined / GT 三栏对比（Nature/Science 风格配色）。

    上方三图: 用 Nature 调色板映射层颜色
    下方三图: 绿色=比GT多(过分割), 红色=比GT少(欠分割)
    """
    # Nature/Science 风格配色 (RGB, 低饱和度高对比度)
    NATURE_CMAP = {
        "L1": (0.329, 0.180, 0.561),    # 紫
        "L2/3": (0.216, 0.596, 0.420),  # 绿
        "L4": (0.878, 0.475, 0.212),    # 橙
        "L5/6": (0.204, 0.490, 0.737),  # 蓝
        "BG": (0.95, 0.95, 0.95),       # 浅灰背景
    }
    LAYER_ORDER = ["L1", "L2/3", "L4", "L5/6"]

    def _remap_to_nature(bgr_img):
        """将 BGR 层掩码重映射为 Nature 配色 RGB。"""
        h, w = bgr_img.shape[:2]
        out = np.full((h, w, 3), NATURE_CMAP["BG"], dtype=np.float32)
        for layer in LAYER_ORDER:
            color_bgr = ALGO_COLORS.get(layer)
            if color_bgr is None:
                continue
            mask = _extract_color_mask(bgr_img, color_bgr, COLOR_TOL) > 0
            rgb = NATURE_CMAP[layer]
            for c in range(3):
                out[mask, c] = rgb[c]
        return out

    def _diff_map(algo_bgr, gt_bgr):
        """差异图: 绿色=algo多(过分割), 红色=algo少(欠分割), 白=一致, 灰=背景。"""
        h, w = algo_bgr.shape[:2]
        out = np.full((h, w, 3), 0.95, dtype=np.float32)  # 浅灰背景
        # 逐层比较
        for layer in LAYER_ORDER:
            c_algo = ALGO_COLORS.get(layer)
            if c_algo is None:
                continue
            a_mask = _extract_color_mask(algo_bgr, c_algo, COLOR_TOL) > 0
            # GT 用自动检测的颜色
            g_color = None
            for cmap in GT_COLORS_CANDIDATES:
                if layer in cmap:
                    g_mask = _extract_color_mask(gt_bgr, cmap[layer], COLOR_TOL) > 0
                    if g_mask.sum() > 100:
                        break
            else:
                continue

            over = a_mask & ~g_mask   # algo 有, GT 无 → 绿色
            under = ~a_mask & g_mask  # GT 有, algo 无 → 红色
            out[over] = (0.42, 0.80, 0.32)   # 绿色
            out[under] = (0.84, 0.28, 0.24)  # 红色
            agree = a_mask & g_mask
            out[agree] = (1.0, 1.0, 1.0)     # 白色 = 一致
        return np.clip(out, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ── 上方: 三栏 Nature 配色 ──
    titles = ["Coarse", "Refined", "Ground Truth"]
    imgs = [coarse, refined if refined is not None else coarse, gt]
    for idx, (ax, img, title) in enumerate(zip(axes[0], imgs, titles)):
        nature_img = _remap_to_nature(img)
        ax.imshow(nature_img)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
        ax.axis("off")

    # ── 下方: 差异图 ──
    if refined is not None:
        diff_c = _diff_map(coarse, gt)
        diff_r = _diff_map(refined, gt)
        axes[1, 0].imshow(diff_c)
        axes[1, 0].set_title("Coarse vs GT\n(Green=Over, Red=Under)",
                              fontsize=13, fontweight="bold", pad=8)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(diff_r)
        axes[1, 1].set_title("Refined vs GT\n(Green=Over, Red=Under)",
                              fontsize=13, fontweight="bold", pad=8)
        axes[1, 1].axis("off")

        # 右侧: 差异变化 (Coarse差异 → Refined差异)
        # 统计: 各层过/欠分割像素数
        layers_coarse = _parse_layers(coarse, ALGO_COLORS)
        layers_refined = _parse_layers(refined, ALGO_COLORS) if refined is not None else {}
        gt_colors = _detect_layer_colors(gt)
        layers_gt = _parse_layers(gt, gt_colors)

        ax = axes[1, 2]
        stats = []
        for layer in LAYER_ORDER:
            a_c = layers_coarse.get(layer)
            a_r = layers_refined.get(layer)
            g = layers_gt.get(layer)
            if a_c is None or a_r is None or g is None:
                continue
            # coarse 差异
            over_c = np.logical_and(a_c > 0, g == 0).sum()
            under_c = np.logical_and(a_c == 0, g > 0).sum()
            # refined 差异
            over_r = np.logical_and(a_r > 0, g == 0).sum()
            under_r = np.logical_and(a_r == 0, g > 0).sum()
            stats.append((layer, over_c, under_c, over_r, under_r))

        # 柱状图比较
        if stats:
            x = np.arange(len(stats))
            w = 0.2
            for i, (layer, oc, uc, or_, ur_) in enumerate(stats):
                ax.bar(i - w * 1.5, oc, w, color=(0.84, 0.28, 0.24), alpha=0.5, label="Coarse Over" if i == 0 else "")
                ax.bar(i - w * 0.5, uc, w, color=(0.42, 0.80, 0.32), alpha=0.5, label="Coarse Under" if i == 0 else "")
                ax.bar(i + w * 0.5, or_, w, color=(0.84, 0.28, 0.24), alpha=0.9, label="Refined Over" if i == 0 else "")
                ax.bar(i + w * 1.5, ur_, w, color=(0.42, 0.80, 0.32), alpha=0.9, label="Refined Under" if i == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels([s[0] for s in stats], fontsize=10)
            ax.set_title("Segmentation Error Pixels\n(Lighter=Coarse, Solid=Refined)",
                         fontsize=13, fontweight="bold", pad=8)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.3)
    else:
        for ax in axes[1]:
            ax.axis("off")

    fig.suptitle(f"Sample {sample} — Coarse vs Refined vs GT  (Nature/Science Style)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = save_dir / "comparison_refined_vs_coarse.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ══════════════════════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════════════════════


def build_summary(all_dfs, samples):
    """汇总所有样本的粗/精对比统计。"""
    combined = pd.concat(all_dfs, ignore_index=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 保存全量数据
    combined.to_csv(SUMMARY_DIR / "all_samples_comparison.csv", index=False, encoding="utf-8-sig")

    # 逐层: coarse vs refined
    metric_cols = ["iou", "dice", "precision", "recall",
                   "avg_boundary_error_px", "thickness_error_percent"]
    summary_rows = []
    for layer in STANDARD_LAYERS:
        for typ in ["coarse", "refined"]:
            sub = combined[(combined["layer"] == layer) & (combined["type"] == typ)]
            row = {"layer": layer, "type": typ}
            for c in metric_cols:
                vals = sub[c].dropna()
                if len(vals) > 0:
                    row[f"{c}_mean"] = vals.mean()
            summary_rows.append(row)

    # 总体
    for typ in ["coarse", "refined"]:
        sub = combined[combined["type"] == typ]
        row = {"layer": "OVERALL", "type": typ}
        for c in metric_cols:
            vals = sub[c].dropna()
            if len(vals) > 0:
                row[f"{c}_mean"] = vals.mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_DIR / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("粗分层 vs NN精炼 对比汇总")
    print(f"{'=' * 60}")
    for layer in STANDARD_LAYERS + ["OVERALL"]:
        print(f"\n  {layer}:")
        for typ in ["coarse", "refined"]:
            sub = summary_df[(summary_df["layer"] == layer) & (summary_df["type"] == typ)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            iou = r.get("iou_mean", None)
            dice = r.get("dice_mean", None)
            be = r.get("avg_boundary_error_px_mean", None)
            te = r.get("thickness_error_percent_mean", None)
            iou_s = f"{iou:.4f}" if pd.notna(iou) else "-"
            dice_s = f"{dice:.4f}" if pd.notna(dice) else "-"
            be_s = f"{be:.1f}px" if pd.notna(be) else "-"
            te_s = f"{te:.1f}%" if pd.notna(te) else "-"
            print(f"    {typ:>8s}: IoU={iou_s} Dice={dice_s} Bnd={be_s} ThkErr={te_s}")

    # 改进率
    print(f"\n  改进率 (refined vs coarse):")
    for layer in STANDARD_LAYERS + ["OVERALL"]:
        c = summary_df[(summary_df["layer"] == layer) & (summary_df["type"] == "coarse")]
        r = summary_df[(summary_df["layer"] == layer) & (summary_df["type"] == "refined")]
        if len(c) == 0 or len(r) == 0:
            continue
        ci = c.iloc[0].get("iou_mean", np.nan)
        ri = r.iloc[0].get("iou_mean", np.nan)
        cb = c.iloc[0].get("avg_boundary_error_px_mean", np.nan)
        rb = r.iloc[0].get("avg_boundary_error_px_mean", np.nan)
        iou_impr = (ri - ci) / ci * 100 if pd.notna(ci) and ci > 0 else 0
        bnd_impr = (cb - rb) / cb * 100 if pd.notna(cb) and cb > 0 else 0
        print(f"    {layer:>8s}: IoU {iou_impr:+.1f}%  Boundary {bnd_impr:+.1f}%")

    # ── 可视化 ──
    _plot_comparison_charts(summary_df, combined)

    return summary_df


def _plot_comparison_charts(summary_df, raw_df):
    """粗/精对比可视化。"""

    # 1. IoU / Dice / Boundary Error 分组柱状图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("iou_mean", "IoU", (0, 1)),
               ("dice_mean", "Dice", (0, 1)),
               ("avg_boundary_error_px_mean", "Avg Boundary Error (px)", None)]

    for idx, (col, label, ylim) in enumerate(metrics):
        ax = axes[idx]
        layers = STANDARD_LAYERS
        x = np.arange(len(layers))
        w = 0.3
        cvals = [summary_df[(summary_df["layer"] == l) & (summary_df["type"] == "coarse")][col].values[0]
                 if len(summary_df[(summary_df["layer"] == l) & (summary_df["type"] == "coarse")]) > 0 else 0
                 for l in layers]
        rvals = [summary_df[(summary_df["layer"] == l) & (summary_df["type"] == "refined")][col].values[0]
                 if len(summary_df[(summary_df["layer"] == l) & (summary_df["type"] == "refined")]) > 0 else 0
                 for l in layers]
        ax.bar(x - w / 2, cvals, w, label="Coarse", alpha=0.7, color="orange")
        ax.bar(x + w / 2, rvals, w, label="Refined", alpha=0.7, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    fig.suptitle("Coarse vs Refined — Per-Layer Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "coarse_vs_refined_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. 各样本改进率热力图
    if "sample" in raw_df.columns and "type" in raw_df.columns:
        # 计算每样本每层改进
        impr_rows = []
        for s in raw_df["sample"].unique():
            for layer in STANDARD_LAYERS:
                c_sub = raw_df[(raw_df["sample"] == s) & (raw_df["type"] == "coarse") & (raw_df["layer"] == layer)]
                r_sub = raw_df[(raw_df["sample"] == s) & (raw_df["type"] == "refined") & (raw_df["layer"] == layer)]
                if len(c_sub) == 0 or len(r_sub) == 0:
                    continue
                for metric in ["iou", "dice", "avg_boundary_error_px"]:
                    cv = c_sub[metric].values[0]
                    rv = r_sub[metric].values[0]
                    if pd.notna(cv) and pd.notna(rv) and cv != 0:
                        impr = (rv - cv) / abs(cv) * 100
                        if metric == "avg_boundary_error_px":
                            impr = -impr  # 正向: 误差降低
                        impr_rows.append({"sample": s, "layer": layer, f"{metric}_improvement": impr})

        if impr_rows:
            impr_df = pd.DataFrame(impr_rows)
            pivot = impr_df.pivot_table(index="sample", columns="layer",
                                         values="iou_improvement", aggfunc="mean")

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=60)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                                fontsize=9, color="black" if abs(v) < 15 else "white")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=10)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=10)
            ax.set_title("IoU Improvement Rate (%) — Refined vs Coarse", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(str(FIGURES_DIR / "improvement_heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  改进热力图: {FIGURES_DIR / 'improvement_heatmap.png'}")

    print(f"  对比图: {FIGURES_DIR}")


# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("粗分层 vs NN精炼 对比评估")
    print("=" * 60)

    samples = sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()])
    samples = [s for s in samples
               if (OUTPUTS_DIR / s / "layers_color_mask.png").exists()
               and (LABEL_DIR / s / "label_mask.png").exists()]
    print(f"发现 {len(samples)} 个样本: {samples}")

    all_dfs = []
    for s in samples:
        result = analyze_sample(s)
        if result is not None:
            all_dfs.append(result)

    if not all_dfs:
        print("\n无有效分析结果!")
        return

    build_summary(all_dfs, samples)

    print(f"\n{'=' * 60}")
    print("分析完成!")
    print(f"  每样本对比: {ANALYSIS_DIR}/<sample>/analysis_comparison.csv")
    print(f"  汇总:        {SUMMARY_DIR}/comparison_summary.csv")
    print(f"  图表:        {FIGURES_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
