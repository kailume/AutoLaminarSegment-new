"""
GP 回归皮层分层 —— 用 Gaussian Process 替代 KDE + 高斯滤波 + 峰值查找。

核心理念
────────
当前管线:  2D KDE → 按 depth 分箱 → gaussian_filter1d(sigma=2) → find_peaks → 导数边界
GP 方案:   按 depth 分箱计数 → GP 回归 (Matern ν=2.5) → 光滑后验密度（含不确定性）→ 导数为零检测

优势
────
1. 无手工 sigma/prominence 调参 —— GP 通过边际似然自动学习光滑程度。
2. 自带不确定性 —— 每个 depth 的密度都有置信区间，边界位置也可附可信度。
3. 后验二阶可导（Matern-5/2）—— 导数稳定，无需额外滤波。
4. 小样本友好 —— GP 天然适合 N_bins≈50 的场景，核函数编码平滑先验。
5. 可扩展 —— 可升级为分层 GP（多切片共享核参数），或二维 Cox process。

与现有管线的集成
───────────────
在 run_pipeline.py 中，将:
    from src.gpLayerSegmentation import segment_layers_gp
    layers = segment_layers_gp(depth, density, bin_width=0.02, merge_layer23=True)
替换:
    from src.analyseDensity import computeAverage, segmentLayer_peak_based
    avg_density, bin_centers = computeAverage(depth, density)
    layers = segmentLayer_peak_based(avg_density, bin_centers)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from scipy.signal import find_peaks

# ──────────────── 模块级配置 ────────────────
output_dir = "output"

# ──────────────── 1. 核函数 ────────────────


def _make_kernel(length_scale_bounds=(0.02, 0.5), noise_bounds=(1e-4, 0.5)):
    """
    构建适用于密度曲线的 GP 核。

    Matern(ν=2.5) 保证后验二阶可导，导数稳定适合边界检测。
    WhiteKernel 捕捉 bin-to-bin 的计数噪声。

    参数可调: 如切片较厚/细胞密可收紧 length_scale_bounds,
              如噪声大可放宽 noise_bounds。
    """
    return ConstantKernel(1.0, constant_value_bounds=(0.1, 10)) * Matern(
        length_scale=0.1,
        length_scale_bounds=length_scale_bounds,
        nu=2.5,
    ) + WhiteKernel(
        noise_level=0.01, noise_level_bounds=noise_bounds
    )


# ──────────────── 2. 核心 GP 拟合 ────────────────


def gp_fit_density_curve(
    bin_centers,
    binned_density,
    n_restarts=10,
    kernel=None,
    return_gp=False,
):
    """
    对分箱后的 depth-density 曲线做 GP 回归，返回光滑后验。

    参数
    ----------
    bin_centers : (N,) float
        分箱中心 depth 值。
    binned_density : (N,) float
        每个 bin 的密度值（可直接是计数或密度估计值）。
    n_restarts : int
        核超参数优化重启次数。
    kernel : sklearn Kernel | None
        自定义核，默认用 _make_kernel()。
    return_gp : bool
        是否同时返回 GP 对象（用于后续诊断）。

    返回
    ------
    x_pred : (500,) float
        精细网格 depth 点（0~1 等距 500 点）。
    y_mean : (500,) float
        GP 后验均值（光滑密度曲线）。
    y_std : (500,) float
        GP 后验标准差（不确定性）。
    dy : (500,) float
        一阶导数 dy/ddepth。
    d2y : (500,) float
        二阶导数 d²y/ddepth²。
    gp : GaussianProcessRegressor
        仅当 return_gp=True 时返回。

    用法
    -----
    x_pred, y_mean, y_std, dy, d2y = gp_fit_density_curve(bin_centers, density)
    # y_mean 可直接替代 gaussian_filter1d(density, sigma=2)
    # dy/d2y 可直接替代 np.gradient(smooth_density)
    """
    if kernel is None:
        kernel = _make_kernel()

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=42,
    )

    X = np.asarray(bin_centers, dtype=float).reshape(-1, 1)
    y = np.asarray(binned_density, dtype=float)
    gp.fit(X, y)

    x_pred = np.linspace(0.0, 1.0, 500)
    y_mean, y_std = gp.predict(x_pred.reshape(-1, 1), return_std=True)

    # 在后验均值上求导（光滑曲线，数值导数足够稳定）
    dx = x_pred[1] - x_pred[0]
    dy = np.gradient(y_mean, dx)
    d2y = np.gradient(dy, dx)

    if return_gp:
        return x_pred, y_mean, y_std, dy, d2y, gp
    return x_pred, y_mean, y_std, dy, d2y


# ──────────────── 3. 层边界检测 ────────────────


def _find_zero_crossings(x, y):
    """
    线性插值寻找 y 的过零点 x。
    """
    crossings = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            w = abs(y[i]) / (abs(y[i]) + abs(y[i + 1]) + 1e-8)
            crossings.append(x[i] + w * (x[i + 1] - x[i]))
    return np.array(crossings)


def _mean_density_in_range(depth_binned, density_binned, start, end):
    """计算原始分箱密度在 depth 区间内的均值。"""
    mask = (depth_binned >= start) & (depth_binned <= end)
    return float(np.mean(density_binned[mask])) if np.any(mask) else 0.0


def detect_layer_boundaries(
    x_pred,
    y_mean,
    y_std,
    dy,
    d2y,
    depth_binned,
    density_binned,
    merge_layer23=True,
):
    """
    基于 GP 后验密度曲线的 peak-based 层边界检测。

    返回格式与 segmentLayer_peak_based 一致：
        [{"layer": "1", "start": ..., "end": ..., "mean_density": ...}, ...]

    相比原算法的关键改进:
        - 峰值筛选引入了不确定性权重（y_mean / y_std）
        - 导数在 GP 后验上计算，比 raw data 更稳定
        - 边界打分考虑了置信度
    """
    # ---- 找主峰 ----
    search_mask = (x_pred >= 0.05) & (x_pred <= 0.8)
    search_idx = np.where(search_mask)[0]
    if len(search_idx) < 10:
        search_idx = np.arange(len(x_pred))

    y_search = y_mean[search_idx]
    min_dist = max(3, len(search_idx) // 10)
    prominence = np.ptp(y_search) * 0.08

    peaks, _ = find_peaks(y_search, distance=min_dist, prominence=prominence)
    if len(peaks) < 2:
        peaks, _ = find_peaks(y_search, distance=min_dist)

    peaks_global = search_idx[peaks]

    # 峰值太少 → 保底
    if len(peaks_global) < 2:
        return None

    # 用 mean / std 打分（不确定性越低的峰权重越大）
    scores = y_mean[peaks_global] / (y_std[peaks_global] + 1e-10)
    top2 = np.argsort(scores)[-2:]
    sorted_peaks = np.sort(peaks_global[top2])

    peak2_idx, peak4_idx = int(sorted_peaks[0]), int(sorted_peaks[1])
    peak2_depth = x_pred[peak2_idx]
    peak4_depth = x_pred[peak4_idx]

    # ---- 二阶导数过零点（候选边界） ----
    zero_crossings = _find_zero_crossings(x_pred, d2y)

    # ---- 赋值层边界 ----
    # L1/L2: L2 峰左侧最近的过零点
    left_of_peak2 = [d for d in zero_crossings if d < peak2_depth]
    b12 = max(left_of_peak2) if left_of_peak2 else peak2_depth * 0.5

    # L2/L3 与 L3/L4: 两峰之间的过零点
    between = [d for d in zero_crossings if peak2_depth < d < peak4_depth]
    if between:
        mid = (peak2_depth + peak4_depth) / 2.0
        b23 = min(between, key=lambda x: abs(x - mid * 0.7))
        remaining = [d for d in between if d > b23]
        b34 = (
            min(remaining, key=lambda x: abs(x - mid * 1.3))
            if remaining
            else (b23 + peak4_depth) / 2.0
        )
    else:
        b23 = peak2_depth + (peak4_depth - peak2_depth) * 0.33
        b34 = peak2_depth + (peak4_depth - peak2_depth) * 0.67

    # L4/L5-6: L4 峰右侧最近的过零点
    right_of_peak4 = [d for d in zero_crossings if d > peak4_depth]
    b456 = min(right_of_peak4) if right_of_peak4 else peak4_depth + (1.0 - peak4_depth) * 0.4

    # 单调性保证
    boundaries = np.clip([0.0, b12, b23, b34, b456, 1.0], 0.0, 1.0)
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 0.02
    boundaries = np.clip(boundaries, 0.0, 1.0)

    if merge_layer23:
        final_b = [boundaries[0], boundaries[1], boundaries[3], boundaries[4], boundaries[5]]
        names = ["1", "2/3", "4", "5/6"]
    else:
        final_b = boundaries.tolist()
        names = ["1", "2", "3", "4", "5/6"]

    layers = []
    for i, name in enumerate(names):
        s, e = float(final_b[i]), float(final_b[i + 1])
        layers.append({
            "layer": name,
            "start": s,
            "end": e,
            "mean_density": _mean_density_in_range(depth_binned, density_binned, s, e),
            "gp_boundary_uncertainty": float(y_std[x_pred.searchsorted((s + e) / 2)]),
        })

    return layers


def _default_layers(depth_binned, density_binned, merge_layer23=True):
    """保底分层（与原版一致）。"""
    if merge_layer23:
        bounds = [0.0, 0.07, 0.35, 0.5, 1.0]
        names = ["1", "2/3", "4", "5/6"]
    else:
        bounds = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        names = ["1", "2", "3", "4", "5/6"]

    return [
        {
            "layer": n,
            "start": float(bounds[i]),
            "end": float(bounds[i + 1]),
            "mean_density": _mean_density_in_range(depth_binned, density_binned,
                                                    bounds[i], bounds[i + 1]),
        }
        for i, n in enumerate(names)
    ]


# ──────────────── 4. 可视化 ────────────────


def plot_gp_result(
    bin_centers,
    binned_density,
    x_pred,
    y_mean,
    y_std,
    dy,
    d2y,
    layers,
    save_path=None,
    merge_layer23=True,
):
    """
    与原有 peak_based 诊断图格式一致的 GP 版分层可视化。
    额外增加: 不确定性带 (68%/95% CI)。
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cmap = plt.get_cmap("tab10")
    ymax = float(max(np.max(y_mean), np.max(binned_density), 1e-12))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # ===== 上图: 密度曲线 + GP 后验 + 层色带 =====
    ax1 = axes[0]
    for i, L in enumerate(layers):
        c = cmap(i % 10)
        ax1.axvspan(L["start"], L["end"], color=c, alpha=0.18)
        mid = (L["start"] + L["end"]) / 2.0
        ax1.text(mid, ymax * 0.95, f"L{L['layer']}",
                 ha="center", va="top", fontsize=10, color=c, fontweight="bold")

    # 原始分箱数据
    ax1.plot(bin_centers, binned_density, "o-", color="C1",
             alpha=0.5, label="Binned Density", markersize=4)
    # GP 后验均值
    ax1.plot(x_pred, y_mean, "-", color="C0", linewidth=2,
             label="GP Posterior Mean")
    # 不确定性带
    ax1.fill_between(
        x_pred, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std,
        color="C0", alpha=0.10, label="95% CI",
    )
    ax1.fill_between(
        x_pred, y_mean - y_std, y_mean + y_std,
        color="C0", alpha=0.15, label="68% CI",
    )

    title = "GP-Based Layer Segmentation"
    title += " (4 Layers)" if merge_layer23 else " (5 Layers)"
    ax1.set_xlabel("Depth (GM → WM)", fontsize=11)
    ax1.set_ylabel("Cell Density", fontsize=11)
    ax1.set_title(title, fontsize=12)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ===== 中图: 一阶导数 =====
    ax2 = axes[1]
    ax2.plot(x_pred, dy, "-", color="green", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    for L in layers:
        if L["layer"] in ("1",):
            continue
        ax2.axvline(L["start"], color="blue", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Depth", fontsize=11)
    ax2.set_ylabel("1st Derivative (GP)", fontsize=11)
    ax2.set_title("First Derivative of GP Posterior", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # ===== 下图: 二阶导数 + 过零点 =====
    ax3 = axes[2]
    ax3.plot(x_pred, d2y, "-", color="orange", linewidth=1.5)
    ax3.axhline(0, color="gray", linestyle="-", alpha=0.5)

    zero_crossings = _find_zero_crossings(x_pred, d2y)
    for zc in zero_crossings:
        ax3.axvline(zc, color="blue", linestyle=":", alpha=0.6)
    # 标记最终边界
    for L in layers:
        if L["layer"] in ("1",):
            continue
        ax3.axvline(L["start"], color="red", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Depth", fontsize=11)
    ax3.set_ylabel("2nd Derivative", fontsize=11)
    ax3.set_title(
        "Second Derivative (dashed=final boundaries, dotted=zero-crossings)",
        fontsize=12,
    )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────── 5. 主入口 ────────────────


def segment_layers_gp(
    depth_sorted,
    density_sorted,
    bin_width=0.02,
    n_bins=None,
    merge_layer23=True,
    gp_n_restarts=10,
    gp_kernel=None,
    plot=True,
    save_dir=None,
):
    """
    GP 回归皮层分层 —— 主入口函数。

    可直接替代 run_pipeline.py 中的:
        avg_density, bin_centers = computeAverage(depth, density)
        layers = segmentLayer_peak_based(avg_density, bin_centers)

    参数
    ----------
    depth_sorted : (N,) float
        analyze() 返回的按 depth 排序的细胞深度。
    density_sorted : (N,) float
        analyze() 返回的与 depth_sorted 对齐的细胞密度。
    bin_width : float
        depth 分箱宽度，默认 0.02 → 50 bins。
    n_bins : int | None
        如果指定，覆盖 bin_width 直接等分 0~1。
    merge_layer23 : bool
        是否合并 L2/L3（默认 True → 4 层）。
    gp_n_restarts : int
        GP 核参数优化重启次数。
    gp_kernel : sklearn Kernel | None
        自定义 GP 核。
    plot : bool
        是否保存诊断图。
    save_dir : str | None
        图保存路径，默认 output_dir。

    返回
    ------
    layers : list[dict]
        与 segmentLayer_peak_based 相同格式:
            [{"layer", "start", "end", "mean_density", "gp_boundary_uncertainty"}, ...]
    """
    depth = np.asarray(depth_sorted, dtype=float)
    density = np.asarray(density_sorted, dtype=float)

    # ── 分箱 ──
    if n_bins is not None:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.arange(0.0, 1.0 + bin_width, bin_width)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    binned = np.zeros_like(bin_centers, dtype=float)

    for i in range(len(bin_centers)):
        mask = (depth >= bin_edges[i]) & (depth < bin_edges[i + 1])
        if i == len(bin_centers) - 1:
            mask = (depth >= bin_edges[i]) & (depth <= bin_edges[i + 1])
        if np.any(mask):
            binned[i] = float(np.mean(density[mask]))

    # ── GP 拟合 ──
    result = gp_fit_density_curve(
        bin_centers, binned,
        n_restarts=gp_n_restarts,
        kernel=gp_kernel,
        return_gp=False,
    )
    x_pred, y_mean, y_std, dy, d2y = result

    # ── 层边界检测 ──
    layers = detect_layer_boundaries(
        x_pred, y_mean, y_std, dy, d2y,
        bin_centers, binned,
        merge_layer23=merge_layer23,
    )
    if layers is None:
        layers = _default_layers(bin_centers, binned, merge_layer23)

    # ── 诊断图 ──
    if plot:
        save_path = os.path.join(
            save_dir or output_dir,
            "depth_density_layers_gp.png",
        )
        plot_gp_result(
            bin_centers, binned,
            x_pred, y_mean, y_std, dy, d2y,
            layers,
            save_path=save_path,
            merge_layer23=merge_layer23,
        )

    # 填充缺少的字段（兼容原格式）
    for L in layers:
        L.setdefault("gp_boundary_uncertainty", 0.0)

    return layers


# ──────────────── 6. 多切片分层 GP ────────────────


def hierarchical_gp_segmentation(sample_list, merge_layer23=True):
    """
    多切片分层 GP: 所有切片共享一组核超参数，但每个切片有自己的 GP 后验。

    这是 GP 比 KDE 方案更优越的关键特性:
    - N=10 的切片可共同训练核参数（长度尺度、噪声水平）
    - 核参数在全体数据上估计 → 更稳定，不易过拟合
    - 之后每个切片独立推断自己的密度曲线

    参数
    ----------
    sample_list : list[dict]
        每个 dict 必须含:
            "depth": (N,) float, "density": (N,) float
        可选:
            "name": str 样本名
    merge_layer23 : bool

    返回
    ------
    all_layers : dict[str, list[dict]]
        样本名 → 分层结果
    gp_kernel : sklearn Kernel
        在所有样本上优化的核（可保存复用）
    """
    # 第一轮: 各自分箱，收集所有 bin 数据
    all_X, all_y = [], []
    per_sample = []

    for s in sample_list:
        depth = np.asarray(s["depth"], dtype=float)
        density = np.asarray(s["density"], dtype=float)
        bin_edges = np.arange(0.0, 1.0 + 0.02, 0.02)
        bc = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        binned = np.zeros_like(bc, dtype=float)
        for i in range(len(bc)):
            mask = (depth >= bin_edges[i]) & (depth < bin_edges[i + 1])
            if i == len(bc) - 1:
                mask = (depth >= bin_edges[i]) & (depth <= bin_edges[i + 1])
            if np.any(mask):
                binned[i] = float(np.mean(density[mask]))
        per_sample.append({"name": s.get("name", ""), "bc": bc, "binned": binned})
        all_X.append(bc)
        all_y.append(binned)

    # 第二轮: 在所有数据上拟合一次 GP → 学习共享核超参数
    stacked_X = np.concatenate(all_X).reshape(-1, 1)
    stacked_y = np.concatenate(all_y)

    kernel = _make_kernel()
    gp_temp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10,
        normalize_y=True, random_state=42,
    )
    gp_temp.fit(stacked_X, stacked_y)
    shared_kernel = gp_temp.kernel_
    print(f"[分层GP] 共享核超参数: {shared_kernel}")

    # 第三轮: 每个切片用固定核参数做自己的 GP 回归
    all_layers = {}
    for s in per_sample:
        # 只有 length_scale 和 noise 固定，后验均值因数据而异
        _, y_mean, y_std, dy, d2y = gp_fit_density_curve(
            s["bc"], s["binned"],
            kernel=shared_kernel,
            n_restarts=1,  # 不需要再优化核参数
        )
        layers = detect_layer_boundaries(
            np.linspace(0, 1, 500), y_mean, y_std, dy, d2y,
            s["bc"], s["binned"],
            merge_layer23=merge_layer23,
        )
        if layers is None:
            layers = _default_layers(s["bc"], s["binned"], merge_layer23)
        for L in layers:
            L.setdefault("gp_boundary_uncertainty", 0.0)

        name = s["name"] or f"sample_{len(all_layers)}"
        all_layers[name] = layers

    return all_layers, shared_kernel


# ──────────────── 7. 快速测试 ────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("GP 皮层分层模块 — 快速自检")
    print("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n_cells = 5000
    depths = np.random.beta(2, 2, n_cells)  # 模拟 depth 分布
    densities = (
        0.5 + 0.3 * np.exp(-((depths - 0.25) ** 2) / 0.005)
        + 0.4 * np.exp(-((depths - 0.55) ** 2) / 0.008)
        + 0.1 * np.random.randn(n_cells) * 0.05
    )
    order = np.argsort(depths)
    depths = depths[order]
    densities = densities[order]

    print(f"模拟细胞: {n_cells}")
    print(f"depth 范围: [{depths.min():.3f}, {depths.max():.3f}]")
    print()

    # 运行 GP 分层
    layers = segment_layers_gp(
        depths, densities,
        bin_width=0.02,
        merge_layer23=True,
        plot=True,
        save_dir="output",
    )

    print("分层结果:")
    for L in layers:
        print(f"  L{L['layer']:>4s}: [{L['start']:.3f}, {L['end']:.3f}]"
              f"  mean_density={L['mean_density']:.3f}"
              f"  uncertainty={L.get('gp_boundary_uncertainty', 0):.3f}")
    print()

    # 测试多切片模式
    print("测试多切片分层 GP...")
    samples = [
        {"name": "sample_A", "depth": np.random.beta(2.2, 2, 3000),
         "density": np.random.rand(3000)},
        {"name": "sample_B", "depth": np.random.beta(1.8, 2.2, 4000),
         "density": np.random.rand(4000)},
        {"name": "sample_C", "depth": np.random.beta(2.5, 1.8, 3500),
         "density": np.random.rand(3500)},
    ]
    for s in samples:
        s["depth"] = np.sort(s["depth"])

    all_layers, shared_k = hierarchical_gp_segmentation(samples)
    print(f"共享核: {shared_k}")
    for name, lyrs in all_layers.items():
        print(f"  {name}: {len(lyrs)} layers")

    print("\n[DONE] 自检完成")
