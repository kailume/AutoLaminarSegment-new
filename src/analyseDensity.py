"""
精简版密度分层算法模块。

本文件只保留后续分层流程真正需要的核心算法：

1. 读取并规范化输入数据：
   - WM/GM 边界点：必须包含 x, y 坐标列；
   - 细胞中心点：兼容 X/Y、x/y、centroid_x/centroid_y。
2. 对每个细胞计算归一化深度：
   - legacy: depth = dist_to_GM / (dist_to_GM + dist_to_WM)
   - harmonic: 在 GM=0、WM=1 的边界条件下求解调和深度场
   - GM 边界附近 depth 接近 0，WM 边界附近 depth 接近 1。
3. 用 KDE 估计每个细胞位置的局部密度。
4. 按 depth 分箱，得到 depth-density 曲线。
5. 使用 peak-based 方法自动推断皮层分层边界。
6. 保存算法诊断图 depth_density_layers_peak_based.png。

主要对外函数：
    analyze()
        输入 WM/GM 边界和细胞坐标，输出按深度排序的 depth 与 density。

    computeAverage()
        输入细胞级 depth/density，输出分箱后的平均密度曲线。

    segmentLayer_peak_based()
        输入 depth-density 曲线，输出最终层边界列表。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde


# run_pipeline.py 会在运行时覆盖该模块级变量，用于控制算法图保存目录。
output_dir = "output"
DEPTH_METHOD = "legacy"      # "legacy" = 最近边界距离；"harmonic" = GM/WM 之间的调和深度场
HARMONIC_MAX_DIM = 1024      # harmonic 深度场网格最长边，越大越精细但越慢


def _ensure_xy_dataframe(data):
    """
    将细胞坐标数据统一整理为包含 X、Y 两列的 DataFrame。

    参数:
        data:
            可以是 CSV 文件路径，也可以是 pandas.DataFrame。
            支持的列名包括：
            - X, Y
            - x, y
            - centroid_x, centroid_y

    返回:
        pandas.DataFrame
            只包含两列：X, Y。

    说明:
        后续所有几何计算都使用 X/Y 表示细胞中心点在 DAPI/40x 图像中的像素坐标。
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x") or cols_lower.get("centroid_x")
    y_col = cols_lower.get("y") or cols_lower.get("centroid_y")
    if x_col is None or y_col is None:
        if "X" in df.columns and "Y" in df.columns:
            x_col, y_col = "X", "Y"
        else:
            raise ValueError(f"Cannot find cell coordinate columns in {list(df.columns)}")

    out = df[[x_col, y_col]].copy()
    out.columns = ["X", "Y"]
    return out


def _ensure_boundary_dataframe(data):
    """
    将边界坐标数据统一整理为包含 x、y 两列的 DataFrame。

    参数:
        data:
            可以是 CSV 文件路径，也可以是 pandas.DataFrame。
            必须包含 x/y 坐标列，不接受 centroid_x/centroid_y。

    返回:
        pandas.DataFrame
            只包含两列：x, y。

    说明:
        GM 和 WM 边界都应已经处于 DAPI/40x 图像坐标系。
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x")
    y_col = cols_lower.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"Boundary data must contain x/y columns, got {list(df.columns)}")

    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    return out


def _nearest_boundary_depths(wm, gm, cells):
    """
    基于最近边界距离计算每个细胞的归一化深度。

    参数:
        wm:
            白质边界点 DataFrame，列为 x, y。
        gm:
            灰质外边界点 DataFrame，列为 x, y。
        cells:
            细胞中心点 DataFrame，列为 X, Y。

    返回:
        numpy.ndarray
            每个细胞的 depth，范围大致为 0 到 1。

    算法:
        1. 分别为 WM 和 GM 边界点建立 KDTree；
        2. 查询每个细胞到最近 WM 点和最近 GM 点的距离；
        3. 用 dist_to_GM / (dist_to_GM + dist_to_WM) 归一化。

    depth 含义:
        - 接近 GM 边界：dist_to_GM 小，depth 接近 0；
        - 接近 WM 边界：dist_to_WM 小，depth 接近 1。
    """
    wm_points = wm[["x", "y"]].to_numpy(dtype=float)
    gm_points = gm[["x", "y"]].to_numpy(dtype=float)
    cell_points = cells[["X", "Y"]].to_numpy(dtype=float)

    dist_to_wm, _ = cKDTree(wm_points).query(cell_points, k=1)
    dist_to_gm, _ = cKDTree(gm_points).query(cell_points, k=1)
    return dist_to_gm / (dist_to_gm + dist_to_wm + 1e-8)


def _build_boundary_lookup(points):
    """Build a robust y=f(x) lookup from boundary points for harmonic depth."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)

    df = pd.DataFrame(pts, columns=["x", "y"])
    grouped = df.groupby("x", as_index=False)["y"].median().sort_values("x")
    x_vals = grouped["x"].to_numpy(dtype=float)
    y_vals = grouped["y"].to_numpy(dtype=float)

    if len(x_vals) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), y_vals[0], dtype=float)

    def _lookup(x_new):
        x_arr = np.asarray(x_new, dtype=float)
        clipped = np.clip(x_arr, x_vals[0], x_vals[-1])
        return np.interp(clipped, x_vals, y_vals)

    return _lookup


def _choose_harmonic_scale(width, height, max_dim=1024):
    longest = max(width, height)
    return max(1, int(np.ceil(longest / max_dim)))


def _harmonic_boundary_depths(wm, gm, cells, max_dim=1024):
    """
    Compute normalized cortical depth by solving a harmonic field between GM and WM.

    GM boundary is fixed to 0, WM boundary is fixed to 1. Cell depth is sampled
    from the solved Laplace field. Points outside the field fall back to legacy.
    """
    wm_points = wm[["x", "y"]].to_numpy(dtype=float)
    gm_points = gm[["x", "y"]].to_numpy(dtype=float)
    cell_points = cells[["X", "Y"]].to_numpy(dtype=float)

    all_points = np.vstack([wm_points, gm_points, cell_points])
    min_x = int(np.floor(all_points[:, 0].min()))
    max_x = int(np.ceil(all_points[:, 0].max()))
    min_y = int(np.floor(all_points[:, 1].min()))
    max_y = int(np.ceil(all_points[:, 1].max()))

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    scale = _choose_harmonic_scale(width, height, max_dim=max_dim)

    xs_full = np.arange(min_x, max_x + 1, scale, dtype=float)
    ys_full = np.arange(min_y, max_y + 1, scale, dtype=float)
    if xs_full[-1] != max_x:
        xs_full = np.append(xs_full, float(max_x))
    if ys_full[-1] != max_y:
        ys_full = np.append(ys_full, float(max_y))

    gm_y = _build_boundary_lookup(gm_points)(xs_full)
    wm_y = _build_boundary_lookup(wm_points)(xs_full)

    h = len(ys_full)
    w = len(xs_full)
    domain_mask = np.zeros((h, w), dtype=bool)
    gm_mask = np.zeros((h, w), dtype=bool)
    wm_mask = np.zeros((h, w), dtype=bool)

    for ix in range(w):
        gy = float(gm_y[ix])
        wy = float(wm_y[ix])
        if not np.isfinite(gy) or not np.isfinite(wy):
            continue

        top_y = min(gy, wy)
        bottom_y = max(gy, wy)
        y_start = int(np.searchsorted(ys_full, top_y, side="left"))
        y_end = int(np.searchsorted(ys_full, bottom_y, side="right")) - 1
        if y_start > y_end or y_start >= h or y_end < 0:
            continue

        y_start = max(0, y_start)
        y_end = min(h - 1, y_end)
        domain_mask[y_start : y_end + 1, ix] = True

        gm_idx = int(np.argmin(np.abs(ys_full - gy)))
        wm_idx = int(np.argmin(np.abs(ys_full - wy)))
        gm_mask[gm_idx, ix] = True
        wm_mask[wm_idx, ix] = True
        domain_mask[gm_idx, ix] = True
        domain_mask[wm_idx, ix] = True

    if not np.any(domain_mask):
        raise RuntimeError("Failed to build a valid cortical ribbon for harmonic depth computation.")

    known_mask = gm_mask | wm_mask
    unknown_mask = domain_mask & (~known_mask)
    unknown_indices = np.argwhere(unknown_mask)

    field = np.full((h, w), np.nan, dtype=np.float64)
    field[gm_mask] = 0.0
    field[wm_mask] = 1.0

    if len(unknown_indices) > 0:
        index_map = -np.ones((h, w), dtype=np.int32)
        for idx, (iy, ix) in enumerate(unknown_indices):
            index_map[iy, ix] = idx

        matrix = lil_matrix((len(unknown_indices), len(unknown_indices)), dtype=np.float64)
        rhs = np.zeros(len(unknown_indices), dtype=np.float64)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row, (iy, ix) in enumerate(unknown_indices):
            degree = 0
            for dy, dx in neighbors:
                ny = iy + dy
                nx = ix + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w or not domain_mask[ny, nx]:
                    continue
                degree += 1
                if unknown_mask[ny, nx]:
                    matrix[row, index_map[ny, nx]] = -1.0
                else:
                    rhs[row] += field[ny, nx]
            matrix[row, row] = max(degree, 1)

        field[unknown_mask] = spsolve(matrix.tocsr(), rhs)

    field = np.clip(field, 0.0, 1.0)
    domain_values = field[domain_mask]
    if np.any(np.isfinite(domain_values)):
        dmin = float(np.nanmin(domain_values))
        dmax = float(np.nanmax(domain_values))
        if dmax > dmin:
            field[domain_mask] = (field[domain_mask] - dmin) / (dmax - dmin)

    interpolator = RegularGridInterpolator(
        (ys_full, xs_full),
        field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    sample_points = np.column_stack([cell_points[:, 1], cell_points[:, 0]])
    depth = interpolator(sample_points)

    invalid = ~np.isfinite(depth)
    if np.any(invalid):
        legacy_depth = _nearest_boundary_depths(wm, gm, cells)
        depth[invalid] = legacy_depth[invalid]

    return depth.astype(float)


def _kde_density(cells, bandwidth="scott"):
    """
    使用二维高斯核密度估计计算每个细胞位置的局部密度。

    参数:
        cells:
            细胞中心点 DataFrame，列为 X, Y。
        bandwidth:
            scipy.stats.gaussian_kde 的 bw_method 参数。
            常用取值：
            - "scott"：默认，按 Scott 规则自动估计带宽；
            - "silverman"：按 Silverman 规则估计带宽；
            - float：手动指定相对带宽因子。

    返回:
        numpy.ndarray
            与细胞一一对应的密度值。

    说明:
        gaussian_kde 输出的是概率密度量级。这里乘以 sqrt(n) 做简单缩放，
        使数值更接近“局部细胞密度强度”，方便后续曲线比较。
    """
    coords = cells[["X", "Y"]].to_numpy(dtype=float).T
    n = coords.shape[1]
    if n < 2:
        return np.ones(n, dtype=float)

    kde = gaussian_kde(coords, bw_method=bandwidth)
    density = kde(coords)
    scale_factor = n / np.sqrt(n)
    return np.asarray(density * scale_factor, dtype=float)


def analyze(wm, gm, cell, kde_bandwidth="scott", depth_method=None, harmonic_max_dim=None, **_ignored):
    """
    计算细胞级深度和密度，并按深度从浅到深排序。

    参数:
        wm:
            WM 边界数据，CSV 路径或 DataFrame，包含 x/y。
        gm:
            GM 边界数据，CSV 路径或 DataFrame，包含 x/y。
        cell:
            细胞中心点数据，CSV 路径或 DataFrame。
        kde_bandwidth:
            KDE 带宽参数，传给 _kde_density()。
        depth_method:
            深度计算方式；"legacy" 使用最近 GM/WM 边界距离，"harmonic" 使用调和深度场。
            None 时使用模块级 DEPTH_METHOD。
        harmonic_max_dim:
            harmonic 深度场网格最长边，None 时使用模块级 HARMONIC_MAX_DIM。
        **_ignored:
            兼容旧 run_pipeline.py 调用保留的冗余参数入口。

    返回:
        tuple[numpy.ndarray, numpy.ndarray]
            depth_sorted:
                按深度升序排列的细胞 depth。
            density_sorted:
                与 depth_sorted 对齐的细胞密度。
    """
    # 统一输入列名，避免下游几何计算反复判断列名。
    boundary_wm = _ensure_boundary_dataframe(wm)
    boundary_gm = _ensure_boundary_dataframe(gm)
    cells = _ensure_xy_dataframe(cell)

    if depth_method is None:
        depth_method = DEPTH_METHOD
    depth_method = str(depth_method).lower()
    harmonic_max_dim = HARMONIC_MAX_DIM if harmonic_max_dim is None else harmonic_max_dim

    # 先计算细胞深度，再计算细胞位置的 KDE 密度。
    if depth_method == "legacy":
        depth = _nearest_boundary_depths(boundary_wm, boundary_gm, cells)
    elif depth_method == "harmonic":
        depth = _harmonic_boundary_depths(
            boundary_wm,
            boundary_gm,
            cells,
            max_dim=harmonic_max_dim,
        )
    else:
        raise ValueError(f"Unknown depth_method: {depth_method}. Use 'legacy' or 'harmonic'.")
    print(f"Depth method: {depth_method}")
    density = _kde_density(cells, bandwidth=kde_bandwidth)

    # 后续分箱和分层都假设 depth 从 0 到 1 单调排列。
    order = np.argsort(depth)
    return depth[order], density[order]


def computeAverage(depth, density, mode="average", bin_width=0.02, **_ignored):
    """
    将细胞级 density 按 depth 分箱，生成 depth-density 曲线。

    参数:
        depth:
            每个细胞的归一化深度，一维数组。
        density:
            每个细胞的局部密度，一维数组，与 depth 等长。
        mode:
            每个 depth bin 内的统计方式：
            - "average"：取平均值；
            - "median"：取中位数。
        bin_width:
            depth 分箱宽度。默认 0.02，即 0~1 共 50 个 bin。
        **_ignored:
            兼容旧版本 isshow、issave、issmooth 等参数；精简版不使用。

    返回:
        tuple[numpy.ndarray, numpy.ndarray]
            avg_density:
                每个 depth bin 内的平均/中位密度。
            bin_centers:
                每个 bin 的中心 depth 值。

    说明:
        这一步把离散细胞点转成连续深度方向上的密度曲线，
        peak-based 分层就是基于这条曲线完成的。
    """
    depth = np.asarray(depth, dtype=float)
    density = np.asarray(density, dtype=float)

    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    avg_density = np.zeros_like(bin_centers, dtype=float)

    for i in range(len(bin_centers)):
        # 最后一个 bin 包含右端点 1.0，避免 depth == 1 的细胞被漏掉。
        in_bin = (depth >= bins[i]) & (depth < bins[i + 1])
        if i == len(bin_centers) - 1:
            in_bin = (depth >= bins[i]) & (depth <= bins[i + 1])
        if not np.any(in_bin):
            continue
        if mode == "median":
            avg_density[i] = float(np.median(density[in_bin]))
        else:
            avg_density[i] = float(np.mean(density[in_bin]))

    return avg_density, bin_centers


def _zero_crossing_depths(values, depth_sorted):
    """
    计算一条曲线的过零点对应的 depth 坐标。

    参数:
        values:
            一维曲线值，例如二阶导数。
        depth_sorted:
            与 values 对齐的 depth 坐标。

    返回:
        numpy.ndarray
            通过线性插值得到的过零 depth 列表。

    在 peak-based 算法中的作用:
        对二阶导数求过零点，用来近似“密度变化率最大的位置”，
        这些位置会作为候选层边界。
    """
    crossings = []
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] < 0:
            t = abs(values[i]) / (abs(values[i]) + abs(values[i + 1]) + 1e-8)
            crossings.append(i + t)
    if not crossings:
        return np.asarray([], dtype=float)
    return np.interp(crossings, np.arange(len(depth_sorted)), depth_sorted)


def _mean_density_for_range(depth, density, start, end):
    """
    计算指定 depth 区间内的平均密度。

    参数:
        depth:
            depth 坐标数组。
        density:
            density 数组，与 depth 对齐。
        start:
            区间起点。
        end:
            区间终点。

    返回:
        float
            区间内 density 均值；如果区间内没有点，则返回 0。
    """
    in_range = (depth >= start) & (depth <= end)
    return float(np.mean(density[in_range])) if np.any(in_range) else 0.0


def _default_layers(depth, density, merge_layer23):
    """
    当无法可靠检测到两个密度峰时，返回保底分层边界。

    参数:
        depth:
            分箱后的 depth 坐标。
        density:
            分箱后的 density 曲线。
        merge_layer23:
            True 时输出 4 层：L1、L2/3、L4、L5/6；
            False 时输出 5 层：L1、L2、L3、L4、L5/6。

    返回:
        list[dict]
            每个 dict 包含 layer、start、end、mean_density。
    """
    if merge_layer23:
        boundaries = [0.0, 0.07, 0.35, 0.5, 1.0]
        names = ["1", "2/3", "4", "5/6"]
    else:
        boundaries = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        names = ["1", "2", "3", "4", "5/6"]

    return [
        {
            "layer": names[i],
            "start": float(boundaries[i]),
            "end": float(boundaries[i + 1]),
            "mean_density": _mean_density_for_range(depth, density, boundaries[i], boundaries[i + 1]),
        }
        for i in range(len(names))
    ]


def _build_layers(boundaries, names, depth, density):
    """
    根据边界数组和层名构造标准分层结果。

    参数:
        boundaries:
            单调递增的边界列表，长度应为层数 + 1。
        names:
            层名列表，例如 ["1", "2/3", "4", "5/6"]。
        depth:
            分箱后的 depth 坐标。
        density:
            分箱后的 density 曲线。

    返回:
        list[dict]
            可直接写入 segmented_layers.csv 的结构。
    """
    layers = []
    for i, name in enumerate(names):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])
        layers.append(
            {
                "layer": name,
                "start": start,
                "end": end,
                "mean_density": _mean_density_for_range(depth, density, start, end),
            }
        )
    return layers


def _save_peak_plot(
    layers,
    depth,
    density,
    smooth_density,
    first_derivative,
    second_derivative,
    peak_indices,
    zero_crossings,
    merge_layer23,
    save_path,
):
    """
    保存 peak-based 分层算法诊断图。

    参数:
        layers:
            最终分层结果列表。
        depth:
            分箱后的 depth 坐标。
        density:
            原始分箱密度曲线。
        smooth_density:
            高斯平滑后的密度曲线。
        first_derivative:
            平滑密度曲线的一阶导数。
        second_derivative:
            平滑密度曲线的二阶导数。
        peak_indices:
            两个主峰在 depth/density 数组中的索引：
            第一个视为 L2 或 L2/3 peak，第二个视为 L4 peak。
            如果未找到峰，则对应位置为 None。
        zero_crossings:
            二阶导数过零点对应的 depth 值。
        merge_layer23:
            是否合并 L2/L3，用于标题和 peak 标签。
        save_path:
            输出 PNG 路径。

    输出图结构:
        上图:
            原始密度曲线、平滑密度曲线、分层色带、主峰位置。
        中图:
            一阶导数，用于观察密度变化趋势。
        下图:
            二阶导数和过零点，用于展示候选分层边界来源。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cmap = plt.get_cmap("tab10")
    ymax = float(max(np.max(density), np.max(smooth_density), 1e-12))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1) 上图：depth-density 曲线、平滑曲线、主峰和最终层边界。
    ax1 = axes[0]
    for i, layer in enumerate(layers):
        color = cmap(i % 10)
        ax1.axvspan(layer["start"], layer["end"], color=color, alpha=0.18)
        mid = (layer["start"] + layer["end"]) / 2.0
        ax1.text(mid, ymax * 0.95, f"L{layer['layer']}", ha="center", va="top", fontsize=10, color=color, fontweight="bold")

    ax1.plot(depth, density, "o-", color="C1", alpha=0.5, label="Raw Density", markersize=4)
    ax1.plot(depth, smooth_density, "-", color="C0", linewidth=2, label="Smoothed Density")

    for peak_idx, color, label in zip(
        peak_indices,
        ["red", "purple"],
        ["L2/3 Peak" if merge_layer23 else "L2 Peak", "L4 Peak"],
    ):
        if peak_idx is None:
            continue
        ax1.axvline(depth[peak_idx], color=color, linestyle="--", alpha=0.8, label=f"{label} ({depth[peak_idx]:.2f})")
        ax1.scatter([depth[peak_idx]], [smooth_density[peak_idx]], color="red", s=100, zorder=5, marker="*")

    if merge_layer23:
        title = "Peak-Based Layer Segmentation (4 Layers: L1, L2/3, L4, L5/6)"
    else:
        title = "Peak-Based Layer Segmentation (5 Layers: L1, L2, L3, L4, L5/6)"
    ax1.set_xlabel("Depth (GM -> WM)", fontsize=11)
    ax1.set_ylabel("Cell Density", fontsize=11)
    ax1.set_title(title, fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2) 中图：一阶导数。峰附近的导数变化反映密度上升/下降趋势。
    ax2 = axes[1]
    ax2.plot(depth, first_derivative, "-", color="green", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    for peak_idx, color in zip(peak_indices, ["red", "purple"]):
        if peak_idx is not None:
            ax2.axvline(depth[peak_idx], color=color, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Depth", fontsize=11)
    ax2.set_ylabel("1st Derivative (Gradient)", fontsize=11)
    ax2.set_title("First Derivative of Density", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3) 下图：二阶导数。过零点是候选层边界的重要依据。
    ax3 = axes[2]
    ax3.plot(depth, second_derivative, "-", color="orange", linewidth=1.5)
    ax3.axhline(0, color="gray", linestyle="-", alpha=0.5)
    for crossing in zero_crossings:
        ax3.axvline(crossing, color="blue", linestyle=":", alpha=0.6)
    for peak_idx, color in zip(peak_indices, ["red", "purple"]):
        if peak_idx is not None:
            ax3.axvline(depth[peak_idx], color=color, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Depth", fontsize=11)
    ax3.set_ylabel("2nd Derivative", fontsize=11)
    ax3.set_title("Second Derivative (Zero-crossings = Layer Boundaries)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def segmentLayer_peak_based(avg_density, depth_list, sigma=2, merge_layer23=True, issave=True, **_ignored):
    """
    基于 depth-density 曲线的 peak-based 自动分层算法。

    参数:
        avg_density:
            computeAverage() 输出的平均密度曲线。
        depth_list:
            computeAverage() 输出的 bin 中心 depth。
        sigma:
            高斯平滑参数。值越大，密度曲线越平滑，但可能抹掉细节峰。
        merge_layer23:
            True:
                输出 4 层：L1、L2/3、L4、L5/6。
                这是当前 pipeline 默认设置。
            False:
                输出 5 层：L1、L2、L3、L4、L5/6。
        issave:
            是否保存 depth_density_layers_peak_based.png。
        **_ignored:
            兼容旧版本 isshow 等参数；精简版不使用。

    返回:
        list[dict]
            每个元素对应一层，包含：
            - layer: 层名；
            - start: 该层起始 depth；
            - end: 该层结束 depth；
            - mean_density: 该层区间内平均密度。

    核心逻辑:
        1. 按 depth 排序密度曲线；
        2. 对密度曲线做高斯平滑；
        3. 计算一阶和二阶导数；
        4. 在 depth 0.05~0.8 范围内寻找两个主要密度峰；
        5. 将较浅的峰视为 L2/L2-3 相关峰，较深的峰视为 L4 峰；
        6. 用二阶导数过零点确定峰之间和峰两侧的层边界；
        7. 根据 merge_layer23 决定是否合并 L2/L3；
        8. 找不到足够峰值时使用默认分层边界。
    """
    # 输入曲线可能不是严格按 depth 排列，先排序保证后续求导和找峰稳定。
    depth = np.asarray(depth_list, dtype=float)
    density = np.asarray(avg_density, dtype=float)
    order = np.argsort(depth)
    depth = depth[order]
    density = density[order]

    # 平滑曲线用于找峰和求导，避免单个 bin 的噪声导致层边界抖动。
    smooth_density = gaussian_filter1d(density, sigma=sigma)
    first_derivative = np.gradient(smooth_density, depth)
    second_derivative = np.gradient(first_derivative, depth)

    # 只在中间主要皮层范围找峰，避开最靠近 GM/WM 端点的边界效应。
    search_mask = (depth >= 0.05) & (depth <= 0.8)
    search_indices = np.where(search_mask)[0]
    if len(search_indices) < 5:
        search_indices = np.arange(len(depth))

    # 先用 prominence 约束找明显峰；如果不足两个，再放宽为仅按距离找峰。
    density_search = smooth_density[search_indices]
    min_distance = max(3, len(search_indices) // 10)
    prominence = np.ptp(density_search) * 0.1
    peaks_local, _ = find_peaks(density_search, distance=min_distance, prominence=prominence)
    if len(peaks_local) < 2:
        peaks_local, _ = find_peaks(density_search, distance=min_distance)

    peaks_global = search_indices[peaks_local]
    peak_indices = [None, None]
    zero_depths = np.asarray([], dtype=float)
    if len(peaks_global) >= 2:
        # 取高度最高的两个峰，并按 depth 从浅到深排序。
        top_two = np.argsort(smooth_density[peaks_global])[-2:]
        peak2_idx, peak4_idx = np.sort(peaks_global[top_two])
        peak_indices = [int(peak2_idx), int(peak4_idx)]

        # 二阶导数过零点作为候选边界。
        zero_depths = _zero_crossing_depths(second_derivative, depth)
        peak2_depth = depth[peak2_idx]
        peak4_depth = depth[peak4_idx]

        # L1/L2 边界：取 L2 峰左侧、最靠近峰的二阶导数过零点。
        left_of_peak2 = [d for d in zero_depths if d < peak2_depth]
        boundary_1_2 = max(left_of_peak2) if left_of_peak2 else peak2_depth * 0.5

        # L2/L3 与 L3/L4 边界：优先使用两个峰之间的二阶导数过零点。
        between_peaks = [d for d in zero_depths if peak2_depth < d < peak4_depth]
        if between_peaks:
            mid_point = (peak2_depth + peak4_depth) / 2.0
            boundary_2_3 = min(between_peaks, key=lambda x: abs(x - mid_point * 0.7))
            remaining = [d for d in between_peaks if d > boundary_2_3]
            boundary_3_4 = min(remaining, key=lambda x: abs(x - mid_point * 1.3)) if remaining else (boundary_2_3 + peak4_depth) / 2.0
        else:
            boundary_2_3 = peak2_depth + (peak4_depth - peak2_depth) * 0.33
            boundary_3_4 = peak2_depth + (peak4_depth - peak2_depth) * 0.67

        # L4/L5-6 边界：取 L4 峰右侧、最靠近峰的二阶导数过零点。
        right_of_peak4 = [d for d in zero_depths if d > peak4_depth]
        boundary_4_56 = min(right_of_peak4) if right_of_peak4 else peak4_depth + (1.0 - peak4_depth) * 0.4

        # 保证边界单调递增，避免异常曲线导致后续区间为空或倒置。
        boundaries = [0.0, boundary_1_2, boundary_2_3, boundary_3_4, boundary_4_56, 1.0]
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 0.02
        boundaries = np.clip(boundaries, 0.0, 1.0).tolist()

        # 根据设置生成最终层结构。merge_layer23=True 时跳过 boundary_2_3。
        if merge_layer23:
            final_boundaries = [boundaries[0], boundaries[1], boundaries[3], boundaries[4], boundaries[5]]
            layer_names = ["1", "2/3", "4", "5/6"]
        else:
            final_boundaries = boundaries
            layer_names = ["1", "2", "3", "4", "5/6"]

        layers = _build_layers(final_boundaries, layer_names, depth, density)
    else:
        # 如果找不到两个可靠峰，使用经验默认边界，保证 pipeline 有稳定输出。
        layers = _default_layers(depth, density, merge_layer23)

    if issave:
        _save_peak_plot(
            layers,
            depth,
            density,
            smooth_density,
            first_derivative,
            second_derivative,
            peak_indices,
            zero_depths,
            merge_layer23,
            os.path.join(output_dir, "depth_density_layers_peak_based.png"),
        )

    return layers


def export_cell_features(
    wm, gm, cell,
    depth_method=None,
    kde_bandwidth="scott",
    harmonic_max_dim=None,
    groundtruth_png=None,
):
    """
    为每个细胞导出完整特征集，用于 SVM 精细化分层训练。

    参数:
        wm, gm, cell: 与 analyze() 相同的 WM/GM 边界和细胞数据。
        depth_method: 深度计算方法 ("legacy" / "harmonic")。
        kde_bandwidth: KDE 带宽参数。
        harmonic_max_dim: harmonic 深度场网格最长边。
        groundtruth_png: ground truth 层掩码 PNG 路径。若提供，为每个细胞提取真实层标签。

    返回:
        pd.DataFrame，每个细胞一行，包含:
            cell_id, centroid_x, centroid_y, area_px, area_um2,
            depth, density, dist_to_gm, dist_to_wm,
            local_cell_count, depth_squared, depth_x_centroid,
            layer_label (若 groundtruth_png 提供), coarse_layer.
    """
    # -- 1. 加载原始细胞数据（保留所有列） --
    if isinstance(cell, str):
        cell_df = pd.read_csv(cell)
    else:
        cell_df = cell.copy()

    # 统一列名
    cols_lower = {c.lower(): c for c in cell_df.columns}

    x_col = cols_lower.get("x") or cols_lower.get("centroid_x")
    y_col = cols_lower.get("y") or cols_lower.get("centroid_y")
    if x_col is None or y_col is None:
        if "X" in cell_df.columns and "Y" in cell_df.columns:
            x_col, y_col = "X", "Y"
        else:
            raise ValueError(f"Cannot find coordinate columns in {list(cell_df.columns)}")

    # 坐标列重命名
    rename_map = {x_col: "centroid_x", y_col: "centroid_y"}
    for col in ["cell_id", "area_px", "area_um2"]:
        if col in cols_lower and cols_lower[col] != col:
            rename_map[cols_lower[col]] = col
    cell_df = cell_df.rename(columns=rename_map)

    # 确保必要的列存在
    for col in ["centroid_x", "centroid_y"]:
        if col not in cell_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 构建 XY DataFrame 供下游函数使用
    cells_xy = cell_df[["centroid_x", "centroid_y"]].copy()
    cells_xy.columns = ["X", "Y"]

    # -- 2. 边界数据 --
    boundary_wm = _ensure_boundary_dataframe(wm)
    boundary_gm = _ensure_boundary_dataframe(gm)

    if depth_method is None:
        depth_method = DEPTH_METHOD
    depth_method = str(depth_method).lower()
    hmax = HARMONIC_MAX_DIM if harmonic_max_dim is None else harmonic_max_dim

    # -- 3. 计算深度 --
    if depth_method == "legacy":
        depth = _nearest_boundary_depths(boundary_wm, boundary_gm, cells_xy)
    elif depth_method == "harmonic":
        depth = _harmonic_boundary_depths(boundary_wm, boundary_gm, cells_xy, max_dim=hmax)
    else:
        raise ValueError(f"Unknown depth_method: {depth_method}")

    # -- 4. 计算 KDE 密度 --
    density = _kde_density(cells_xy, bandwidth=kde_bandwidth)

    # -- 5. 计算到 GM/WM 边界的距离 --
    cell_points = cells_xy[["X", "Y"]].to_numpy(dtype=float)
    wm_points = boundary_wm[["x", "y"]].to_numpy(dtype=float)
    gm_points = boundary_gm[["x", "y"]].to_numpy(dtype=float)
    dist_to_wm, _ = cKDTree(wm_points).query(cell_points, k=1)
    dist_to_gm, _ = cKDTree(gm_points).query(cell_points, k=1)

    # -- 6. 计算局部细胞计数 (半径 100 um, 约 200 像素) --
    # 使用 return_length=True 高效获取所有点的邻域计数
    LOCAL_RADIUS = 200.0  # 像素, ~100 um at 0.5 um/px
    tree = cKDTree(cell_points)
    local_cell_count = tree.query_ball_point(cell_points, LOCAL_RADIUS, return_length=True)

    # -- 7. 构建特征 DataFrame --
    feature_df = pd.DataFrame({
        "cell_id": cell_df.get("cell_id", np.arange(len(cell_df)) + 1),
        "centroid_x": cell_df["centroid_x"].values,
        "centroid_y": cell_df["centroid_y"].values,
        "area_px": cell_df.get("area_px", 0),
        "area_um2": cell_df.get("area_um2", 0.0),
        "depth": depth,
        "density": density,
        "dist_to_gm": dist_to_gm,
        "dist_to_wm": dist_to_wm,
        "local_cell_count": local_cell_count,
    })

    # 衍生特征
    feature_df["depth_squared"] = feature_df["depth"] ** 2
    feature_df["depth_x_centroid"] = feature_df["depth"] * feature_df["centroid_x"]

    # -- 8. 从 groundtruth.png 提取真实层标签（若提供）--
    feature_df["layer_label"] = ""
    if groundtruth_png and os.path.isfile(str(groundtruth_png)):
        try:
            import cv2 as _cv2
            gt_img = _cv2.imread(str(groundtruth_png))
            if gt_img is None:
                print(f"[export_cell_features] 警告: 无法读取 {groundtruth_png}")
            else:
                print(f"[export_cell_features] 从 {groundtruth_png} 提取层标签...")
                # GT 颜色映射 (BGR)
                # 从 output/harmonic_depth_validation/ 各样本 GT 图中提取
                # 按像素数排序: L5/6(最多) > L2/3 > L4 > L1(最少)
                gt_color_map = {
                    (0, 0, 255): "1",       # Red -> L1 (最薄层, 最少像素)
                    (127, 127, 0): "2/3",    # Teal -> L2/3
                    (0, 255, 255): "4",      # Yellow -> L4
                    (255, 127, 127): "5/6",  # Pink -> L5/6 (最厚层, 最多像素)
                }

                # 批量颜色匹配 (向量化)
                xs = np.round(feature_df["centroid_x"].values).astype(int)
                ys = np.round(feature_df["centroid_y"].values).astype(int)
                h_img, w_img = gt_img.shape[:2]
                in_bounds = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)

                # 只处理图内细胞
                valid_idx = np.where(in_bounds)[0]
                valid_x = xs[valid_idx]
                valid_y = ys[valid_idx]
                pixel_bgr = gt_img[valid_y, valid_x, :].astype(int)  # (N, 3)

                # 预分配标签数组
                labels = np.full(len(feature_df), "", dtype=object)
                assigned = np.zeros(len(valid_idx), dtype=bool)

                for (ref_b, ref_g, ref_r), lyr in gt_color_map.items():
                    match = (
                        (np.abs(pixel_bgr[:, 0] - ref_b) <= 15) &
                        (np.abs(pixel_bgr[:, 1] - ref_g) <= 15) &
                        (np.abs(pixel_bgr[:, 2] - ref_r) <= 15) &
                        ~assigned
                    )
                    labels[valid_idx[match]] = lyr
                    assigned |= match

                feature_df["layer_label"] = labels
                n_labeled = int(assigned.sum())
                print(f"  -> {n_labeled}/{len(feature_df)} 个细胞获得真实层标签")
        except ImportError:
            print("[export_cell_features] 警告: 需要 opencv-python 读取 ground truth PNG")
        except Exception as e:
            print(f"[export_cell_features] 提取层标签失败: {e}")

    # -- 9. coarse_layer 暂设为空，后续由 pipeline 填充 --
    feature_df["coarse_layer"] = ""

    return feature_df
