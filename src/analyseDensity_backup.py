import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import os

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 默认输出路径
output_dir = "output"


def _ensure_xy_dataframe(data, x_candidates, y_candidates):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    cols_lower = {c.lower(): c for c in df.columns}
    x_col = next((cols_lower[c] for c in x_candidates if c in cols_lower), None)
    y_col = next((cols_lower[c] for c in y_candidates if c in cols_lower), None)
    if x_col is None or y_col is None:
        raise ValueError(f"Cannot find coordinate columns in {list(df.columns)}")

    out = df[[x_col, y_col]].copy()
    out.columns = ["X", "Y"]
    return out


def _ensure_boundary_dataframe(data):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    cols_lower = {c.lower(): c for c in df.columns}
    x_col = cols_lower.get("x")
    y_col = cols_lower.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"Boundary file must contain x/y columns, got {list(df.columns)}")

    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    return out


def _build_boundary_lookup(pts):
    pts = np.asarray(pts, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return lambda x: np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)

    df = pd.DataFrame(pts, columns=["x", "y"])
    grouped = df.groupby("x", as_index=False)["y"].median().sort_values("x")
    x_vals = grouped["x"].to_numpy(dtype=float)
    y_vals = grouped["y"].to_numpy(dtype=float)

    if len(x_vals) == 1:
        y0 = float(y_vals[0])

        def _constant_lookup(x_new):
            x_arr = np.asarray(x_new, dtype=float)
            return np.full_like(x_arr, y0, dtype=float)

        return _constant_lookup

    def _lookup(x_new):
        x_arr = np.asarray(x_new, dtype=float)
        clipped = np.clip(x_arr, x_vals[0], x_vals[-1])
        return np.interp(clipped, x_vals, y_vals)

    return _lookup


def _nearest_boundary_distances(cell_points, boundary_points):
    tree = cKDTree(np.asarray(boundary_points, dtype=float))
    distances, _ = tree.query(np.asarray(cell_points, dtype=float), k=1)
    return distances.astype(float)


def compute_legacy_depths(wm, gm, cell):
    boundary_W = _ensure_boundary_dataframe(wm)
    boundary_G = _ensure_boundary_dataframe(gm)
    cells = _ensure_xy_dataframe(cell, ["x", "centroid_x"], ["y", "centroid_y"])

    cell_points = cells[["X", "Y"]].to_numpy(dtype=float)
    wm_points = boundary_W[["x", "y"]].to_numpy(dtype=float)
    gm_points = boundary_G[["x", "y"]].to_numpy(dtype=float)

    dist_to_W = _nearest_boundary_distances(cell_points, wm_points)
    dist_to_G = _nearest_boundary_distances(cell_points, gm_points)
    depth = dist_to_G / (dist_to_W + dist_to_G + 1e-8)
    return depth, {
        "dist_to_W": dist_to_W,
        "dist_to_G": dist_to_G,
        "cell_points": cell_points,
        "wm_points": wm_points,
        "gm_points": gm_points,
    }


def _choose_harmonic_scale(width, height, max_dim=1024):
    longest = max(width, height)
    return max(1, int(np.ceil(longest / max_dim)))


def compute_harmonic_depths(wm, gm, cell, max_dim=1024):
    boundary_W = _ensure_boundary_dataframe(wm)
    boundary_G = _ensure_boundary_dataframe(gm)
    cells = _ensure_xy_dataframe(cell, ["x", "centroid_x"], ["y", "centroid_y"])

    wm_points = boundary_W[["x", "y"]].to_numpy(dtype=float)
    gm_points = boundary_G[["x", "y"]].to_numpy(dtype=float)
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

    gm_lookup = _build_boundary_lookup(gm_points)
    wm_lookup = _build_boundary_lookup(wm_points)
    gm_y = gm_lookup(xs_full)
    wm_y = wm_lookup(xs_full)

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

        A = lil_matrix((len(unknown_indices), len(unknown_indices)), dtype=np.float64)
        b = np.zeros(len(unknown_indices), dtype=np.float64)
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
                    A[row, index_map[ny, nx]] = -1.0
                else:
                    b[row] += field[ny, nx]
            A[row, row] = max(degree, 1)

        field[unknown_mask] = spsolve(A.tocsr(), b)

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
        legacy_depth, _ = compute_legacy_depths(boundary_W, boundary_G, cells)
        depth[invalid] = legacy_depth[invalid]

    return depth.astype(float), {
        "depth_field": field,
        "domain_mask": domain_mask,
        "gm_mask": gm_mask,
        "wm_mask": wm_mask,
        "xs": xs_full,
        "ys": ys_full,
        "scale": scale,
        "bbox": (min_x, min_y, max_x, max_y),
        "cell_points": cell_points,
        "wm_points": wm_points,
        "gm_points": gm_points,
    }

def calculate_cell_density(cells, radius=100):
    coords = cells[['X', 'Y']].values

    # 使用NearestNeighbors计算密度
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    densities = []

    for coord in coords:
        # 找到半径内的所有邻居
        indices = nbrs.radius_neighbors([coord], return_distance=False)[0]
        # 密度为邻居数量除以圆形区域面积
        # density = len(indices) / (np.pi * radius**2)
        density = len(indices)
        densities.append(density)

    return np.array(densities)


def calculate_cell_density_kde(cells, bw_method='scott', scale_factor=None):
    """
    使用高斯核密度估计（KDE）计算细胞密度，代替固定半径的计数法。

    KDE 将每个细胞视为一个平滑核的中心，叠加得到连续的密度场，
    对局部密度变化更敏感，且不依赖半径参数的硬阈值。

    Parameters
    ----------
    cells : pd.DataFrame
        包含 'X', 'Y' 列的细胞坐标数据。
    bw_method : str or float
        'scott' → Scott's rule（默认，适用于近似高斯分布的数据）
        'silverman' → Silverman's rule（对尖峰分布更敏感）
        float → 手动指定带宽（像素单位）
    scale_factor : float or None
        最终密度的缩放因子。若为 None，则自动设为 N / (N^(1/d))
        使密度值量级与原始半径法可比。

    Returns
    -------
    densities : np.ndarray
        每个细胞位置的核密度估计值。
    """
    from scipy.stats import gaussian_kde

    coords = cells[['X', 'Y']].values.T  # gaussian_kde 要求 (d, n) 格式
    n = coords.shape[1]

    # 如果 bandwidth 是数值（像素单位），转为相对协方差因子
    if isinstance(bw_method, (int, float)):
        # 计算数据协方差，再缩放为标准差以匹配 bw_method
        cov = np.cov(coords)
        # 目标：使 KDE 的带宽 ≈ bw_method 像素
        # gaussian_kde 的 bw_method 是协方差缩放因子，
        # 设为 bw_method / data_stdev 使有效带宽约为 bw_method
        data_stdev = np.sqrt(np.trace(cov) / 2) if np.trace(cov) > 0 else 1.0
        bw_effective = bw_method / data_stdev if data_stdev > 0 else 1.0
        bw_method = bw_effective

    kde = gaussian_kde(coords, bw_method=bw_method)

    # 在每个细胞位置评估密度
    densities = kde(coords)

    # 缩放为强度（不是概率密度），使其量级与半径法可比
    if scale_factor is None:
        scale_factor = n / n ** (1.0 / 2.0) if n > 0 else 1.0
    densities = densities * scale_factor

    return np.asarray(densities).flatten()

def radiustocorrelation(cells, depth, isshow=True):
    # 测试不同半径对相关性的影响
    multi_density = {}
    multi_correlation = {}
    radius_list = [10, 20, 30, 50, 100, 150, 200, 300, 500]
    best_radius = 100
    for radius in range(10, 201, 10):
    # for radius in radius_list:
        density = calculate_cell_density(cells, radius=radius)
        correlation = np.corrcoef(depth, density)[0, 1]
        multi_density[radius] = density
        multi_correlation[radius] = correlation
        if abs(correlation) > abs(multi_correlation.get(best_radius, 0)):
            best_radius = radius
    if isshow:
        plt.figure(figsize=(10, 6))
        plt.plot(list(multi_correlation.keys()), list(multi_correlation.values()), marker='o')
        plt.xlabel('Radius for Density Calculation', fontsize=12)
        plt.ylabel('Correlation between Depth and Density', fontsize=12)
        plt.title('Correlation vs Radius for Density Calculation', fontsize=14)
        plt.grid()
        plt.show()
    
    print(f"best radius: {best_radius}, correlation: {multi_correlation[best_radius]:.3f}")
    return best_radius

def analyze(wm, gm, cell, depth_method='harmonic', radius=300, return_details=False, harmonic_max_dim=1024,
            density_method='radius', kde_bandwidth='scott'):
    boundary_W = _ensure_boundary_dataframe(wm)
    boundary_G = _ensure_boundary_dataframe(gm)
    cells = _ensure_xy_dataframe(cell, ["x", "centroid_x"], ["y", "centroid_y"])

    print(f"Boundary W points: {len(boundary_W)}")
    print(f"Boundary G points: {len(boundary_G)}")
    print(f"Number of cells: {len(cells)}")
    print(f"Depth method: {depth_method}")
    print(f"Density method: {density_method}")

    legacy_depth, legacy_details = compute_legacy_depths(boundary_W, boundary_G, cells)
    harmonic_details = None

    if depth_method == 'harmonic':
        depth, harmonic_details = compute_harmonic_depths(
            boundary_W, boundary_G, cells, max_dim=harmonic_max_dim
        )
    elif depth_method == 'legacy':
        depth = legacy_depth
    else:
        raise ValueError(f"Unknown depth method: {depth_method}")

    if density_method == 'kde':
        density = calculate_cell_density_kde(cells, bw_method=kde_bandwidth)
    elif density_method == 'radius':
        density = calculate_cell_density(cells, radius=radius)
    else:
        raise ValueError(f"Unknown density method: {density_method}")

    sorted_indices = np.argsort(depth)
    depth_sorted = depth[sorted_indices]
    density_sorted = density[sorted_indices]

    correlation = np.corrcoef(depth_sorted, density_sorted)[0, 1]

    print("\nStatistics:")
    print(f"Depth range: {depth_sorted.min():.3f} - {depth_sorted.max():.3f}")
    print(f"Depth mean: {depth_sorted.mean():.3f}")
    print(f"Density range: {density_sorted.min():.6f} - {density_sorted.max():.6f}")
    print(f"Density mean: {density_sorted.mean():.6f}")
    print(f"Depth-density correlation: {correlation:.3f}")

    if depth_method == 'harmonic':
        delta = np.abs(depth - legacy_depth)
        print(f"Legacy vs harmonic depth mean abs diff: {delta.mean():.4f}")
        print(f"Legacy vs harmonic depth max abs diff: {delta.max():.4f}")

    if return_details:
        return depth_sorted, density_sorted, {
            'depth_unsorted': depth,
            'density_unsorted': density,
            'sorted_indices': sorted_indices,
            'cells': cells,
            'legacy_depth_unsorted': legacy_depth,
            'legacy_details': legacy_details,
            'harmonic_details': harmonic_details,
            'depth_method': depth_method,
            'radius': radius,
        }

    return depth_sorted, density_sorted

def visualize(depth, density, issave=True):
    # 创建二维直方图
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ax_main = axes[0, 0]
    # h = ax_main.hist2d(depth, density, bins=50, cmap='viridis', alpha=0.8)
    # ax_main.set_xlabel('Depth to outer boundary', fontsize=12)
    # ax_main.set_ylabel('Density', fontsize=12)
    # ax_main.set_title('Depth vs Density', fontsize=14)
    # plt.colorbar(h[3], ax=ax_main, label='Cell Count')
    
    # # 深度的边际分布
    # ax_depth = axes[0, 1]
    # ax_depth.hist(depth, bins=50, alpha=0.7, color='blue', edgecolor='black')
    # ax_depth.set_xlabel('Cell Depth', fontsize=12)
    # ax_depth.set_ylabel('Frequency', fontsize=12)
    # ax_depth.set_title('Depth Distribution', fontsize=14)
    
    # # 密度的边际分布
    # ax_density = axes[1, 0]
    # ax_density.hist(density, bins=50, alpha=0.7, color='red', edgecolor='black')
    # ax_density.set_xlabel('Cell Density', fontsize=12)
    # ax_density.set_ylabel('Frequency', fontsize=12)
    # ax_density.set_title('Density Distribution', fontsize=14)
    
    # 散点图，尺寸为8x6
    plt.figure(figsize=(8, 5))
    plt.scatter(depth, density, c=density, alpha=0.6, s=10)
    # scatter = plt.scatter(depth, density, c=density, cmap='plasma', alpha=0.6, s=10)
    plt.xlabel('Depth', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Depth-Density Scatter Plot', fontsize=14)
    # plt.colorbar(scatter, label='Density')
    
    plt.tight_layout()
    if issave: plt.savefig(os.path.join(output_dir, 'depth_density_scatter.png'))
    plt.show(block=True)
    
    # # 使用seaborn创建更美观的联合分布图
    # # 创建数据框
    # data_df = pd.DataFrame({
    #     'depth': depth,
    #     'density': density
    # })
    # # 绘制联合分布图
    # g = sns.JointGrid(data=data_df, x='depth', y='density', height=8)
    # # 主图：二维直方图
    # g.plot_joint(plt.hexbin, gridsize=30, cmap='Blues')
    # # 边际图：直方图
    # g.plot_marginals(sns.histplot, kde=True)
    # # 设置标签
    # g.set_axis_labels('Cell Depth (Relative to Boundary B)', 'Cell Density', fontsize=12)
    # plt.suptitle('Joint Distribution of Cell Depth and Density', fontsize=14, y=1.02)
    # plt.show()

def computeAverage(depth, density, isshow=True, issave=True, mode='average', issmooth=True):
    bins = np.arange(0, 1.01, 0.02)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    avg_density = []
    for i in range(len(bins)-1):
        mask = (depth >= bins[i]) & (depth < bins[i+1])
        if np.sum(mask) > 0:
            if mode == 'average':
                avg_density.append(np.mean(density[mask]))
            elif mode == 'median':
                avg_density.append(np.median(density[mask]))
        else:
            avg_density.append(0)
    avg_density = np.array(avg_density)
    if issmooth: 
        avg_density = gaussian_filter1d(avg_density, sigma=2)
    if isshow:
        # 把每个点连接起来
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, avg_density, marker='o')
        plt.xlabel('Depth (pial to white)', fontsize=12)
        plt.ylabel('Average Cell Density', fontsize=12)
        plt.title('Average Cell Density vs Depth', fontsize=14)
        plt.grid()
        if issave: plt.savefig(os.path.join(output_dir, f'depth-{mode}-density.png'))
        plt.show(block=True)

        # # 把平均密度曲线叠加在原始散点图上
        # plt.figure(figsize=(8, 5))
        # plt.scatter(depth, density, c=density, alpha=0.6, s=10)
        # plt.plot(bin_centers, avg_density, color='red')
        # plt.xlabel('Depth', fontsize=12)
        # plt.ylabel('Density', fontsize=12)
        # plt.title('Depth-Density Scatter Plot with Average Curve', fontsize=14)
        # plt.tight_layout()
        # if issave: plt.savefig(f'IO\\OUTPUT\\depth_density_scatter_with_avg.png')
        # plt.show()

    return avg_density, bin_centers

def _labels_to_layers(labels, depth_list, avg_density, n_clusters):
    """
    将聚类标签转换为分层结果的辅助函数
    """
    # 将簇按深度中心排序，保证层号与深度呈单调关系（从浅到深）
    cluster_mean_depth = []
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            cluster_mean_depth.append((k, depth_list[mask].mean()))
        else:
            cluster_mean_depth.append((k, np.inf))
    # 按深度排序
    ordered = [k for k, _ in sorted(cluster_mean_depth, key=lambda x: x[1])]

    # 重新映射标签为 0..(n_clusters-1)，并按深度顺序编号
    label_map = {old: new for new, old in enumerate(ordered)}
    ordered_labels = np.array([label_map[l] for l in labels])

    # 计算每个层的起止深度（扩展半个 bin 宽度以覆盖区间）
    if len(depth_list) > 1:
        bin_width = np.median(np.diff(np.sort(depth_list)))
    else:
        bin_width = 0.02
    
    layers = []
    for layer_idx in range(n_clusters):
        mask = ordered_labels == layer_idx
        if not np.any(mask):
            continue
        start = depth_list[mask].min() - bin_width/2
        end = depth_list[mask].max() + bin_width/2
        start = max(0.0, start)
        end = min(1.0, end)
        mean_d = float(np.mean(avg_density[mask]))
        layers.append({'layer': layer_idx+1, 'start': float(start), 'end': float(end), 'mean_density': mean_d})
    
    return layers, ordered_labels


def _visualize_layers(layers, depth_list, avg_density, title_suffix='', issave=True, savename='depth_density_layers.png'):
    """
    可视化分层结果的辅助函数
    """
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(8, 5))
    # 背景色带
    ymax = max(avg_density) if np.any(avg_density) else 1.0
    for i, L in enumerate(layers):
        # 支持字符串类型的layer名称（如 "5/6"）
        layer_idx = i if isinstance(L['layer'], str) else L['layer'] - 1
        color = cmap(layer_idx % 10)
        plt.axvspan(L['start'], L['end'], color=color, alpha=0.18)
        # 在区间中间标注层号
        mid = (L['start'] + L['end']) / 2
        plt.text(mid, ymax*0.95, f"L{L['layer']}", ha='center', va='top', fontsize=9, color=color)

    plt.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
    plt.xlabel('Cell Depth (to GM)', fontsize=12)
    plt.ylabel('Average Cell Density', fontsize=12)
    plt.title(f'Segmented Layers (n={len(layers)}) {title_suffix}', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if issave: 
        plt.savefig(os.path.join(output_dir, savename))
    plt.show(block=True)

    # # 竖向绘制密度-深度曲线和分层结果
    # plt.figure(figsize=(5, 8))
    # xmax = max(avg_density) if np.any(avg_density) else 1.0
    
    # for L in layers:
    #     color = cmap((L['layer']-1) % 10)
    #     plt.axhspan(L['start'], L['end'], color=color, alpha=0.18)
    #     # 在区间中间标注层号
    #     mid = (L['start'] + L['end']) / 2
    #     plt.text(xmax*0.9, mid, f"Layer {L['layer']}", ha='right', va='center', fontsize=9, color=color)

    # plt.plot(avg_density, depth_list, marker='o', linestyle='-', color='C1')
    # plt.gca().invert_yaxis() # 翻转Y轴，使0(GM)在上方
    
    # plt.ylabel('Cell Depth (to GM)', fontsize=12)
    # plt.xlabel('Average Cell Density', fontsize=12)
    # plt.title(f'Segmented Layers (Vertical) {title_suffix}', fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    if issave:
        vert_savename = savename.replace('.png', '_vertical.png')
        if vert_savename == savename: 
             vert_savename = 'vertical_' + savename

def _compute_layer_density_diff(layers, method_name=''):
    """
    计算层间密度差异作为层间差异验证的辅助函数
    
    参数:
        layers: 分层结果列表，每个元素包含 'layer', 'start', 'end', 'mean_density'
        method_name: 方法名称，用于打印
    
    返回:
        diff_stats: 包含层间差异统计信息的字典
    """
    if len(layers) < 2:
        print(f"\n[{method_name}] 层间密度差异验证: 层数不足，无法计算层间差异")
        return None
    
    densities = [L['mean_density'] for L in layers]
    
    # 计算相邻层之间的密度差异
    inter_layer_diffs = []
    for i in range(len(densities) - 1):
        diff = abs(densities[i+1] - densities[i])
        inter_layer_diffs.append(diff)
    
    # 计算统计指标
    mean_diff = np.mean(inter_layer_diffs)
    max_diff = np.max(inter_layer_diffs)
    min_diff = np.min(inter_layer_diffs)
    std_diff = np.std(inter_layer_diffs)
    total_range = max(densities) - min(densities)
    
    # 计算层间差异比率（相邻层差异占总密度范围的比例）
    if total_range > 0:
        diff_ratio = mean_diff / total_range
    else:
        diff_ratio = 0
    
    # 计算层间差异变异系数 (CV)
    if mean_diff > 0:
        cv = std_diff / mean_diff
    else:
        cv = 0
    
    # 打印层间差异信息
    # print(f"\n[{method_name}] 层间密度差异验证:")
    # print(f"  各层平均密度: {[f'{d:.2f}' for d in densities]}")
    print(f"  相邻层间差异: {[f'{d:.2f}' for d in inter_layer_diffs]}")
    print(f"  平均层间差异: {mean_diff:.4f}")
    # print(f"  最大层间差异: {max_diff:.4f} (Layer {inter_layer_diffs.index(max_diff)+1} -> {inter_layer_diffs.index(max_diff)+2})")
    # print(f"  最小层间差异: {min_diff:.4f}")
    # print(f"  层间差异标准差: {std_diff:.4f}")
    # print(f"  层间差异比率 (平均差异/总范围): {diff_ratio:.4f}")
    print(f"  层间差异变异系数 (CV): {cv:.4f}")
    
    diff_stats = {
        'densities': densities,
        'inter_layer_diffs': inter_layer_diffs,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'min_diff': min_diff,
        'std_diff': std_diff,
        'total_range': total_range,
        'diff_ratio': diff_ratio,
        'cv': cv
    }
    
    return diff_stats


def segmentLayer_peak_based(avg_density, depth_list, sigma=2, merge_layer23=False, isshow=True, issave=True):
    """
    基于峰值的分层算法
    
    算法逻辑：
    1. 在深度0.1~0.8范围内搜索两个最明显的密度峰值
    2. 左边峰值作为第2层中心，右边峰值作为第4层中心
    3. 在峰值两侧搜索一阶梯度为0（二阶导数过零点，即密度变化率最大的拐点）作为层边界
    4. 第5层和第6层合并为"5/6"层
    5. 可选：第2层和第3层合并为"2/3"层（当merge_layer23=True时）
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组  
        sigma: 高斯平滑参数
        merge_layer23: 是否合并第2层和第3层（忽略第一个峰值右侧的分层线）
        isshow: 是否显示可视化
        issave: 是否保存图像
    
    返回:
        layers: 分层结果列表，每层包含 layer, start, end, mean_density
               - merge_layer23=False: 5层 (L1, L2, L3, L4, L5/6)
               - merge_layer23=True:  4层 (L1, L2/3, L4, L5/6)
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)
    
    # 按深度排序
    sort_idx = np.argsort(depth_list)
    depth_sorted = depth_list[sort_idx]
    density_sorted = avg_density[sort_idx]
    
    # 高斯平滑
    density_smooth = gaussian_filter1d(density_sorted, sigma=sigma)
    
    # 计算一阶和二阶导数
    first_deriv = np.gradient(density_smooth, depth_sorted)
    second_deriv = np.gradient(first_deriv, depth_sorted)
    
    # ===== 步骤1: 在0.1~0.8深度范围内找两个最明显的峰值 =====
    search_mask = (depth_sorted >= 0.05) & (depth_sorted <= 0.8)
    search_indices = np.where(search_mask)[0]
    
    if len(search_indices) < 5:
        print("警告: 0.1~0.8深度范围内数据点不足，使用全范围搜索")
        search_indices = np.arange(len(depth_sorted))
    
    # 在搜索范围内找峰值
    density_search = density_smooth[search_indices]
    # 使用相对高度和距离约束找峰值
    min_distance = max(3, len(search_indices) // 10)
    peaks_local, properties = find_peaks(
        density_search, 
        distance=min_distance,
        prominence=np.ptp(density_search) * 0.1  # 峰值突出度至少为范围的10%
    )
    
    # 映射回原始索引
    peaks_global = search_indices[peaks_local]
    
    if len(peaks_global) < 2:
        # 如果峰值不足2个，降低阈值重试
        peaks_local, properties = find_peaks(density_search, distance=min_distance)
        peaks_global = search_indices[peaks_local]
    
    found_peaks = len(peaks_global) >= 2

    if found_peaks:
        # 选择最高的两个峰值
        peak_heights = density_smooth[peaks_global]
        top2_idx = np.argsort(peak_heights)[-2:]
        two_peaks = np.sort(peaks_global[top2_idx])  # 按深度排序

        peak_layer2_idx = two_peaks[0]  # 左峰 -> Layer 2中心
        peak_layer4_idx = two_peaks[1]  # 右峰 -> Layer 4中心

        peak_layer2_depth = depth_sorted[peak_layer2_idx]
        peak_layer4_depth = depth_sorted[peak_layer4_idx]

        print(f"  找到Layer 2中心: depth={peak_layer2_depth:.3f}, density={density_smooth[peak_layer2_idx]:.2f}")
        print(f"  找到Layer 4中心: depth={peak_layer4_depth:.3f}, density={density_smooth[peak_layer4_idx]:.2f}")

        # ===== 步骤2: 寻找二阶导数过零点作为层边界 =====
        # 二阶导数过零点 = 一阶导数的极值点 = 密度变化率最大的位置

        def find_zero_crossings(arr):
            """找到数组中符号变化的位置（过零点）"""
            crossings = []
            for i in range(len(arr) - 1):
                if arr[i] * arr[i+1] < 0:
                    # 线性插值找更精确的位置
                    crossings.append(i + abs(arr[i]) / (abs(arr[i]) + abs(arr[i+1]) + 1e-8))
            return crossings

        zero_crossings = find_zero_crossings(second_deriv)
        zero_crossing_depths = np.interp(zero_crossings, np.arange(len(depth_sorted)), depth_sorted)

        print(f"  二阶导数过零点数量: {len(zero_crossings)}")

        # ===== 步骤3: 根据峰值位置确定层边界 =====
        # Layer 1: 0 ~ boundary_1_2
        # Layer 2: boundary_1_2 ~ boundary_2_3 (中心在peak_layer2)
        # Layer 3: boundary_2_3 ~ boundary_3_4
        # Layer 4: boundary_3_4 ~ boundary_4_56 (中心在peak_layer4)
        # Layer 5/6: boundary_4_56 ~ 1.0

        boundaries = [0.0]  # 起始边界

        # 边界1-2: Layer2峰值左侧最近的过零点
        left_of_peak2 = [d for d in zero_crossing_depths if d < peak_layer2_depth]
        if left_of_peak2:
            boundary_1_2 = max(left_of_peak2)  # 最靠近峰值的
        else:
            boundary_1_2 = peak_layer2_depth * 0.5  # 默认
        boundaries.append(boundary_1_2)

        # 边界2-3: Layer2峰值和Layer4峰值之间的过零点
        between_peaks = [d for d in zero_crossing_depths
                        if peak_layer2_depth < d < peak_layer4_depth]
        if between_peaks:
            # 取中间的过零点，或者最接近两峰中点的
            mid_point = (peak_layer2_depth + peak_layer4_depth) / 2
            boundary_2_3 = min(between_peaks, key=lambda x: abs(x - mid_point * 0.7))  # 偏向Layer2

            # 如果有多个过零点，找Layer3的边界
            remaining = [d for d in between_peaks if d > boundary_2_3]
            if remaining:
                boundary_3_4 = min(remaining, key=lambda x: abs(x - mid_point * 1.3))  # 偏向Layer4
            else:
                boundary_3_4 = (boundary_2_3 + peak_layer4_depth) / 2
        else:
            # 没有过零点，均分
            boundary_2_3 = peak_layer2_depth + (peak_layer4_depth - peak_layer2_depth) * 0.33
            boundary_3_4 = peak_layer2_depth + (peak_layer4_depth - peak_layer2_depth) * 0.67

        boundaries.append(boundary_2_3)
        boundaries.append(boundary_3_4)

        # 边界4-5/6: Layer4峰值右侧的过零点
        right_of_peak4 = [d for d in zero_crossing_depths if d > peak_layer4_depth]
        if right_of_peak4:
            boundary_4_56 = min(right_of_peak4)  # 最靠近峰值的
        else:
            boundary_4_56 = peak_layer4_depth + (1.0 - peak_layer4_depth) * 0.4
        boundaries.append(boundary_4_56)

        boundaries.append(1.0)  # 结束边界

        # 确保边界递增
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i-1]:
                boundaries[i] = boundaries[i-1] + 0.02
        boundaries = np.clip(boundaries, 0, 1)

        print(f"  计算出的层边界(原始): {[f'{b:.3f}' for b in boundaries]}")

        # ===== 步骤4: 根据merge_layer23参数构建分层结果 =====
        if merge_layer23:
            # 合并第2层和第3层：忽略boundary_2_3（第一个峰值右侧的分层线）
            # 原始边界: [0, boundary_1_2, boundary_2_3, boundary_3_4, boundary_4_56, 1.0]
            # 合并后:   [0, boundary_1_2, boundary_3_4, boundary_4_56, 1.0]
            merged_boundaries = [boundaries[0], boundaries[1], boundaries[3], boundaries[4], boundaries[5]]
            layer_names = ['1', '2/3', '4', '5/6']
            n_layers = 4
            print(f"  合并L2/L3后的边界: {[f'{b:.3f}' for b in merged_boundaries]}")
        else:
            # 保持5层
            merged_boundaries = boundaries
            layer_names = ['1', '2', '3', '4', '5/6']
            n_layers = 5
    else:
        print("警告: 未能找到足够的峰值，使用默认分层")
        # 生成默认均匀分割的4层结果，但仍保留中间分析图
        defaults = _create_default_4_layers(depth_sorted, density_sorted)
        merged_boundaries = [0.0]
        for L in defaults:
            merged_boundaries.append(L['end'])
        layer_names = [L['layer'] for L in defaults]
        n_layers = len(defaults)
        # 占位值：后续可视化中不绘制峰值/过零点
        peak_layer2_idx = None
        peak_layer4_idx = None
        zero_crossing_depths = []

    layers = []
    for i in range(n_layers):
        start = merged_boundaries[i]
        end = merged_boundaries[i + 1]

        # 计算该层的平均密度
        mask = (depth_sorted >= start) & (depth_sorted <= end)
        if np.any(mask):
            mean_density = float(np.mean(density_sorted[mask]))
        else:
            mean_density = 0.0

        layers.append({
            'layer': layer_names[i],
            'start': float(start),
            'end': float(end),
            'mean_density': mean_density
        })

    # ===== 可视化 =====
    print(f"  [调试] isshow={isshow}, issave={issave}, merge_layer23={merge_layer23}")
    if isshow:
        print("  [调试] 正在调用 _visualize_peak_based_layers...")
        _visualize_peak_based_layers(
            layers, depth_sorted, density_sorted, density_smooth,
            first_deriv, second_deriv,
            peak_layer2_idx, peak_layer4_idx, zero_crossing_depths,
            issave, merge_layer23,
            found_peaks=found_peaks,
        )
        print("  [调试] _visualize_peak_based_layers 调用完成")

    # 计算层间密度差异
    method_name = 'Peak-Based (L2/3 merged)' if merge_layer23 else 'Peak-Based'
    _compute_layer_density_diff(layers, method_name)

    return layers


def _create_default_5_layers(depth_sorted, density_sorted):
    """创建默认的5层分割"""
    boundaries = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    layer_names = ['1', '2', '3', '4', '5/6']
    layers = []
    
    for i in range(5):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (depth_sorted >= start) & (depth_sorted <= end)
        mean_density = float(np.mean(density_sorted[mask])) if np.any(mask) else 0.0
        layers.append({
            'layer': layer_names[i],
            'start': float(start),
            'end': float(end),
            'mean_density': mean_density
        })
    return layers

def _create_default_4_layers(depth_sorted, density_sorted):
    """创建默认的4层分割"""
    boundaries = [0.0, 0.07, 0.35, 0.5, 1.0]
    layer_names = ['1', '2/3', '4', '5/6']
    layers = []
    
    for i in range(4):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (depth_sorted >= start) & (depth_sorted <= end)
        mean_density = float(np.mean(density_sorted[mask])) if np.any(mask) else 0.0
        layers.append({
            'layer': layer_names[i],
            'start': float(start),
            'end': float(end),
            'mean_density': mean_density
        })
    return layers


def _visualize_peak_based_layers(layers, depth_sorted, density_sorted, density_smooth,
                                  first_deriv, second_deriv,
                                  peak2_idx, peak4_idx, zero_crossings,
                                  issave=True, merge_layer23=False,
                                  found_peaks=True):
    """可视化基于峰值的分层结果"""

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 上图：密度曲线、峰值和分层
    ax1 = axes[0]
    cmap = plt.get_cmap('tab10')
    ymax = max(density_sorted) if np.any(density_sorted) else 1.0

    # 绘制分层背景色带
    for i, L in enumerate(layers):
        color = cmap(i % 10)
        ax1.axvspan(L['start'], L['end'], color=color, alpha=0.18)
        mid = (L['start'] + L['end']) / 2
        ax1.text(mid, ymax*0.95, f"L{L['layer']}", ha='center', va='top', fontsize=10, color=color, fontweight='bold')

    # 绘制密度曲线
    ax1.plot(depth_sorted, density_sorted, 'o-', color='C1', alpha=0.5, label='Raw Density', markersize=4)
    ax1.plot(depth_sorted, density_smooth, '-', color='C0', linewidth=2, label='Smoothed Density')

    # 标记峰值（仅在找到时）
    if found_peaks and peak2_idx is not None and peak4_idx is not None:
        peak_label2 = 'L2/3 Peak' if merge_layer23 else 'L2 Peak'
        ax1.axvline(depth_sorted[peak2_idx], color='red', linestyle='--', alpha=0.8, label=f'{peak_label2} ({depth_sorted[peak2_idx]:.2f})')
        ax1.axvline(depth_sorted[peak4_idx], color='purple', linestyle='--', alpha=0.8, label=f'L4 Peak ({depth_sorted[peak4_idx]:.2f})')
        ax1.scatter([depth_sorted[peak2_idx], depth_sorted[peak4_idx]],
                    [density_smooth[peak2_idx], density_smooth[peak4_idx]],
                    color='red', s=100, zorder=5, marker='*')

    ax1.set_xlabel('Depth (GM → WM)', fontsize=11)
    ax1.set_ylabel('Cell Density', fontsize=11)
    # 根据是否合并调整标题
    if found_peaks:
        if merge_layer23:
            title = 'Peak-Based Layer Segmentation (4 Layers: L1, L2/3, L4, L5/6)'
        else:
            title = 'Peak-Based Layer Segmentation (5 Layers: L1, L2, L3, L4, L5/6)'
    else:
        title = 'Peak-Based Layer Segmentation [DEFAULT — no peaks found]'
    ax1.set_title(title, fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 中图：一阶导数
    ax2 = axes[1]
    ax2.plot(depth_sorted, first_deriv, '-', color='green', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    if found_peaks and peak2_idx is not None:
        ax2.axvline(depth_sorted[peak2_idx], color='red', linestyle='--', alpha=0.5)
    if found_peaks and peak4_idx is not None:
        ax2.axvline(depth_sorted[peak4_idx], color='purple', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Depth', fontsize=11)
    ax2.set_ylabel('1st Derivative (Gradient)', fontsize=11)
    ax2.set_title('First Derivative of Density', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 下图：二阶导数和过零点
    ax3 = axes[2]
    ax3.plot(depth_sorted, second_deriv, '-', color='orange', linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # 标记过零点（仅在找到时）
    for zc in zero_crossings:
        ax3.axvline(zc, color='blue', linestyle=':', alpha=0.6)

    if found_peaks and peak2_idx is not None:
        ax3.axvline(depth_sorted[peak2_idx], color='red', linestyle='--', alpha=0.5)
    if found_peaks and peak4_idx is not None:
        ax3.axvline(depth_sorted[peak4_idx], color='purple', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Depth', fontsize=11)
    ax3.set_ylabel('2nd Derivative', fontsize=11)
    ax3.set_title('Second Derivative (Zero-crossings = Layer Boundaries)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if issave:
        try:
            save_path = os.path.join(output_dir, 'depth_density_layers_peak_based.png')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"  分层结果图已保存: {save_path}")
        except Exception as e:
            print(f"  保存分层结果图失败: {e}")
    plt.show(block=True)
    print("  [调试] plt.show() 已执行")


def segmentLayer_kmeans(avg_density, depth_list, n_clusters=6, offset=False, isshow=True, issave=True):
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)

    # 特征：[归一化深度, 归一化平均密度]；做min-max归一化以平衡尺度
    d_min, d_ptp = depth_list.min(), np.ptp(depth_list) + 1e-8
    rho_min, rho_ptp = avg_density.min(), np.ptp(avg_density) + 1e-8
    depth_scaled = (depth_list - d_min) / d_ptp
    dens_scaled = (avg_density - rho_min) / rho_ptp
    X_norm = np.vstack([depth_scaled, dens_scaled]).T

    if offset:
        # 定义目标深度初始点
        target_depths = np.array([0.05, 0.18, 0.35, 0.55, 0.75, 0.92])
        # (插值)获取这些深度对应的密度值
        sort_idx = np.argsort(depth_list)
        target_densities = np.interp(target_depths, depth_list[sort_idx], avg_density[sort_idx])

        # 将初始点归一化到与 X_norm 相同的空间
        init_d_norm = (target_depths - d_min) / d_ptp
        init_rho_norm = (target_densities - rho_min) / rho_ptp
        init_centers = np.vstack([init_d_norm, init_rho_norm]).T

        kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1, random_state=42)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels = kmeans.fit_predict(X_norm)
    layers, _ = _labels_to_layers(labels, depth_list, avg_density, n_clusters)

    if isshow:
        _visualize_layers(layers, depth_list, avg_density, '[KMeans]', issave, 'depth_density_layers_kmeans.png')

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'KMeans')

    return layers


def segmentLayer_gradient(avg_density, depth_list, n_layers=6, sigma=2, isshow=True, issave=True):
    """
    方法1: 密度梯度分析法
    通过计算密度曲线的一阶导数（梯度），在梯度变化剧烈处（极值点）确定层边界
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组
        n_layers: 目标分层数量
        sigma: 高斯平滑参数
        isshow: 是否显示可视化
        issave: 是否保存图像
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)
    
    # 按深度排序
    sort_idx = np.argsort(depth_list)
    depth_sorted = depth_list[sort_idx]
    density_sorted = avg_density[sort_idx]
    
    # 高斯平滑以减少噪声
    density_smooth = gaussian_filter1d(density_sorted, sigma=sigma)
    
    # 计算一阶导数（梯度）
    gradient = np.gradient(density_smooth, depth_sorted)
    
    # 找到梯度的极值点（正负转换处）作为潜在边界
    # 使用梯度绝对值的峰值
    gradient_abs = np.abs(gradient)
    peaks, properties = find_peaks(gradient_abs, height=np.percentile(gradient_abs, 50))
    
    # 如果找到的边界点太少，降低阈值
    if len(peaks) < n_layers - 1:
        peaks, properties = find_peaks(gradient_abs, height=np.percentile(gradient_abs, 25))
    
    # 选择最显著的 n_layers-1 个边界点
    if len(peaks) >= n_layers - 1:
        # 按峰值高度排序，选择最高的
        peak_heights = properties['peak_heights']
        top_indices = np.argsort(peak_heights)[::-1][:n_layers-1]
        boundary_indices = np.sort(peaks[top_indices])
    else:
        # 如果边界点不够，均匀分割
        boundary_indices = np.linspace(0, len(depth_sorted)-1, n_layers+1, dtype=int)[1:-1]
    
    # 构建边界深度
    boundaries = [0.0] + list(depth_sorted[boundary_indices]) + [1.0]
    boundaries = sorted(set(boundaries))
    
    # 构建layers
    layers = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (depth_sorted >= start) & (depth_sorted < end)
        if i == len(boundaries) - 2:  # 最后一层包含右边界
            mask = (depth_sorted >= start) & (depth_sorted <= end)
        if np.any(mask):
            mean_d = float(np.mean(density_sorted[mask]))
        else:
            mean_d = 0.0
        layers.append({'layer': i+1, 'start': float(start), 'end': float(end), 'mean_density': mean_d})

    if isshow:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 上图：密度曲线和梯度
        ax1 = axes[0]
        ax1.plot(depth_sorted, density_smooth, 'b-', label='Smoothed Density', linewidth=2)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Density', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(depth_sorted, gradient, 'r--', label='Gradient', alpha=0.7)
        ax1_twin.set_ylabel('Gradient', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 标记边界点
        for idx in boundary_indices:
            ax1.axvline(depth_sorted[idx], color='green', linestyle=':', alpha=0.8)
        ax1.set_title('Density Gradient Analysis')
        ax1.legend(loc='upper left')
        
        # 下图：分层结果
        ax2 = axes[1]
        cmap = plt.get_cmap('tab10')
        ymax = max(avg_density) if np.any(avg_density) else 1.0
        for L in layers:
            color = cmap((L['layer']-1) % 10)
            ax2.axvspan(L['start'], L['end'], color=color, alpha=0.18)
            mid = (L['start'] + L['end']) / 2
            ax2.text(mid, ymax*0.95, f"Layer {L['layer']}", ha='center', va='top', fontsize=9, color=color)
        ax2.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Cell Depth (to GM)')
        ax2.set_ylabel('Average Cell Density')
        ax2.set_title(f'Segmented Layers (n={len(layers)}) [Gradient Method]')
        ax2.grid(True)
        
        plt.tight_layout()
        if issave:
            plt.savefig('depth_density_layers_gradient.png')
        plt.show()

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'Gradient')

    return layers


def segmentLayer_second_derivative(avg_density, depth_list, n_layers=6, sigma=3, isshow=True, issave=True):
    """
    方法2: 二阶导数方法
    通过计算密度曲线的二阶导数，在拐点处（二阶导数过零点）确定层边界
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组
        n_layers: 目标分层数量
        sigma: 高斯平滑参数
        isshow: 是否显示可视化
        issave: 是否保存图像
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)
    
    # 按深度排序
    sort_idx = np.argsort(depth_list)
    depth_sorted = depth_list[sort_idx]
    density_sorted = avg_density[sort_idx]
    
    # 高斯平滑
    density_smooth = gaussian_filter1d(density_sorted, sigma=sigma)
    
    # 计算一阶和二阶导数
    first_deriv = np.gradient(density_smooth, depth_sorted)
    second_deriv = np.gradient(first_deriv, depth_sorted)
    
    # 找到二阶导数的过零点（拐点）
    zero_crossings = []
    for i in range(len(second_deriv) - 1):
        if second_deriv[i] * second_deriv[i+1] < 0:  # 符号变化
            # 线性插值找到精确的过零位置
            zero_crossings.append(i)
    
    # 同时找到二阶导数的极值点
    local_max = argrelextrema(second_deriv, np.greater, order=2)[0]
    local_min = argrelextrema(second_deriv, np.less, order=2)[0]
    extrema = np.sort(np.concatenate([local_max, local_min]))
    
    # 合并过零点和极值点作为候选边界
    candidates = np.unique(np.concatenate([zero_crossings, extrema]))
    
    # 选择最显著的边界点
    if len(candidates) >= n_layers - 1:
        # 按二阶导数绝对值排序
        significance = np.abs(second_deriv[candidates])
        top_indices = np.argsort(significance)[::-1][:n_layers-1]
        boundary_indices = np.sort(candidates[top_indices])
    else:
        # 如果边界点不够，均匀分割
        boundary_indices = np.linspace(0, len(depth_sorted)-1, n_layers+1, dtype=int)[1:-1]
    
    # 构建边界深度
    boundaries = [0.0] + list(depth_sorted[boundary_indices]) + [1.0]
    boundaries = sorted(set(boundaries))
    
    # 构建layers
    layers = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (depth_sorted >= start) & (depth_sorted < end)
        if i == len(boundaries) - 2:
            mask = (depth_sorted >= start) & (depth_sorted <= end)
        if np.any(mask):
            mean_d = float(np.mean(density_sorted[mask]))
        else:
            mean_d = 0.0
        layers.append({'layer': i+1, 'start': float(start), 'end': float(end), 'mean_density': mean_d})

    if isshow:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 上图：密度曲线和二阶导数
        ax1 = axes[0]
        ax1.plot(depth_sorted, density_smooth, 'b-', label='Smoothed Density', linewidth=2)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Density', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(depth_sorted, second_deriv, 'r--', label='2nd Derivative', alpha=0.7)
        ax1_twin.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax1_twin.set_ylabel('2nd Derivative', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 标记边界点
        for idx in boundary_indices:
            ax1.axvline(depth_sorted[idx], color='green', linestyle=':', alpha=0.8)
        ax1.set_title('Second Derivative Analysis (Inflection Points)')
        ax1.legend(loc='upper left')
        
        # 下图：分层结果
        ax2 = axes[1]
        cmap = plt.get_cmap('tab10')
        ymax = max(avg_density) if np.any(avg_density) else 1.0
        for L in layers:
            color = cmap((L['layer']-1) % 10)
            ax2.axvspan(L['start'], L['end'], color=color, alpha=0.18)
            mid = (L['start'] + L['end']) / 2
            ax2.text(mid, ymax*0.95, f"Layer {L['layer']}", ha='center', va='top', fontsize=9, color=color)
        ax2.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Cell Depth (to GM)')
        ax2.set_ylabel('Average Cell Density')
        ax2.set_title(f'Segmented Layers (n={len(layers)}) [2nd Derivative Method]')
        ax2.grid(True)
        
        plt.tight_layout()
        if issave:
            plt.savefig('depth_density_layers_2nd_derivative.png')
        plt.show()

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, '2nd Derivative')

    return layers


def segmentLayer_gmm(avg_density, depth_list, n_clusters=6, isshow=True, issave=True):
    """
    方法4: 高斯混合模型 (GMM)
    使用GMM对深度-密度二维数据进行概率聚类
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组
        n_clusters: 聚类数量
        isshow: 是否显示可视化
        issave: 是否保存图像
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)

    # 特征归一化
    d_min, d_ptp = depth_list.min(), np.ptp(depth_list) + 1e-8
    rho_min, rho_ptp = avg_density.min(), np.ptp(avg_density) + 1e-8
    depth_scaled = (depth_list - d_min) / d_ptp
    dens_scaled = (avg_density - rho_min) / rho_ptp
    X_norm = np.vstack([depth_scaled, dens_scaled]).T

    # 使用GMM进行聚类
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', 
                          random_state=42, n_init=10)
    labels = gmm.fit_predict(X_norm)
    
    # 获取每个点属于各簇的概率
    probas = gmm.predict_proba(X_norm)
    
    layers, ordered_labels = _labels_to_layers(labels, depth_list, avg_density, n_clusters)

    if isshow:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：概率分布热图
        ax1 = axes[0]
        sort_idx = np.argsort(depth_list)
        im = ax1.imshow(probas[sort_idx].T, aspect='auto', cmap='YlOrRd',
                        extent=[depth_list.min(), depth_list.max(), n_clusters-0.5, -0.5])
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Cluster')
        ax1.set_title('GMM Cluster Probabilities')
        plt.colorbar(im, ax=ax1, label='Probability')
        
        # 右图：分层结果
        ax2 = axes[1]
        cmap = plt.get_cmap('tab10')
        ymax = max(avg_density) if np.any(avg_density) else 1.0
        for L in layers:
            color = cmap((L['layer']-1) % 10)
            ax2.axvspan(L['start'], L['end'], color=color, alpha=0.18)
            mid = (L['start'] + L['end']) / 2
            ax2.text(mid, ymax*0.95, f"Layer {L['layer']}", ha='center', va='top', fontsize=9, color=color)
        ax2.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Cell Depth (to GM)')
        ax2.set_ylabel('Average Cell Density')
        ax2.set_title(f'Segmented Layers (n={len(layers)}) [GMM]')
        ax2.grid(True)
        
        plt.tight_layout()
        if issave:
            plt.savefig('depth_density_layers_gmm.png')
        plt.show()

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'GMM')

    return layers


def segmentLayer_multi_threshold(avg_density, depth_list, n_layers=6, isshow=True, issave=True):
    """
    方法5: 多阈值分割法
    基于密度值的多阈值分割，使用Otsu多阈值或均匀分位数方法
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组
        n_layers: 目标分层数量
        isshow: 是否显示可视化
        issave: 是否保存图像
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)
    
    # 按深度排序
    sort_idx = np.argsort(depth_list)
    depth_sorted = depth_list[sort_idx]
    density_sorted = avg_density[sort_idx]
    
    # 方法：基于密度变化率的自适应阈值
    # 计算密度的变化
    density_diff = np.abs(np.diff(density_sorted))
    density_diff = np.concatenate([[0], density_diff])  # 补齐长度
    
    # 计算累积密度变化作为分割依据
    cumsum_diff = np.cumsum(density_diff)
    total_diff = cumsum_diff[-1] + 1e-8
    normalized_cumsum = cumsum_diff / total_diff
    
    # 根据累积变化量均匀分割
    thresholds = np.linspace(0, 1, n_layers + 1)[1:-1]
    boundary_indices = []
    for thresh in thresholds:
        idx = np.argmin(np.abs(normalized_cumsum - thresh))
        boundary_indices.append(idx)
    boundary_indices = np.unique(boundary_indices)
    
    # 如果边界点不够，补充基于密度分位数的边界
    if len(boundary_indices) < n_layers - 1:
        # 使用密度分位数作为补充
        density_percentiles = np.percentile(density_sorted, np.linspace(0, 100, n_layers + 1)[1:-1])
        for perc in density_percentiles:
            idx = np.argmin(np.abs(density_sorted - perc))
            boundary_indices = np.append(boundary_indices, idx)
        boundary_indices = np.unique(boundary_indices)[:n_layers-1]
    
    boundary_indices = np.sort(boundary_indices)
    
    # 构建边界深度
    boundaries = [0.0] + list(depth_sorted[boundary_indices]) + [1.0]
    boundaries = sorted(set(boundaries))
    
    # 构建layers
    layers = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mask = (depth_sorted >= start) & (depth_sorted < end)
        if i == len(boundaries) - 2:
            mask = (depth_sorted >= start) & (depth_sorted <= end)
        if np.any(mask):
            mean_d = float(np.mean(density_sorted[mask]))
        else:
            mean_d = 0.0
        layers.append({'layer': i+1, 'start': float(start), 'end': float(end), 'mean_density': mean_d})

    if isshow:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 上图：密度曲线和累积变化
        ax1 = axes[0]
        ax1.plot(depth_sorted, density_sorted, 'b-', label='Density', linewidth=2)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Density', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(depth_sorted, normalized_cumsum, 'g--', label='Cumulative Change', alpha=0.7)
        ax1_twin.set_ylabel('Normalized Cumulative Change', color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        
        # 标记阈值线
        for thresh in thresholds:
            ax1_twin.axhline(thresh, color='red', linestyle=':', alpha=0.5)
        
        # 标记边界点
        for idx in boundary_indices:
            ax1.axvline(depth_sorted[idx], color='purple', linestyle=':', alpha=0.8)
        ax1.set_title('Multi-Threshold Segmentation (Density Change Based)')
        ax1.legend(loc='upper left')
        
        # 下图：分层结果
        ax2 = axes[1]
        cmap = plt.get_cmap('tab10')
        ymax = max(avg_density) if np.any(avg_density) else 1.0
        for L in layers:
            color = cmap((L['layer']-1) % 10)
            ax2.axvspan(L['start'], L['end'], color=color, alpha=0.18)
            mid = (L['start'] + L['end']) / 2
            ax2.text(mid, ymax*0.95, f"Layer {L['layer']}", ha='center', va='top', fontsize=9, color=color)
        ax2.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Cell Depth (to GM)')
        ax2.set_ylabel('Average Cell Density')
        ax2.set_title(f'Segmented Layers (n={len(layers)}) [Multi-Threshold]')
        ax2.grid(True)
        
        plt.tight_layout()
        if issave:
            plt.savefig('depth_density_layers_multi_threshold.png')
        plt.show()

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'Multi-Threshold')

    return layers


def segmentLayer_dbscan(avg_density, depth_list, eps=0.15, min_samples=2, isshow=True, issave=True):
    """
    方法6: DBSCAN密度聚类
    使用DBSCAN对深度-密度二维数据进行基于密度的聚类，能够自动发现任意形状的簇
    
    参数:
        avg_density: 平均密度数组
        depth_list: 深度数组
        eps: DBSCAN的邻域半径参数，控制簇的紧密程度
        min_samples: 形成核心点所需的最小样本数
        isshow: 是否显示可视化
        issave: 是否保存图像
    """
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)

    # 特征归一化
    d_min, d_ptp = depth_list.min(), np.ptp(depth_list) + 1e-8
    rho_min, rho_ptp = avg_density.min(), np.ptp(avg_density) + 1e-8
    depth_scaled = (depth_list - d_min) / d_ptp
    dens_scaled = (avg_density - rho_min) / rho_ptp
    X_norm = np.vstack([depth_scaled, dens_scaled]).T

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_norm)
    
    # 处理噪声点（标签为-1的点）
    # 将噪声点分配给最近的非噪声点
    unique_labels = set(labels)
    if -1 in unique_labels:
        noise_mask = labels == -1
        non_noise_mask = ~noise_mask
        
        if np.any(non_noise_mask):
            # 对于每个噪声点，找到最近的非噪声点并分配相同的标签
            from scipy.spatial.distance import cdist
            noise_points = X_norm[noise_mask]
            non_noise_points = X_norm[non_noise_mask]
            non_noise_labels = labels[non_noise_mask]
            
            distances = cdist(noise_points, non_noise_points)
            nearest_indices = np.argmin(distances, axis=1)
            labels[noise_mask] = non_noise_labels[nearest_indices]
    
    # 获取实际的簇数量
    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 0:
        print("警告: DBSCAN未找到任何簇，尝试调整eps和min_samples参数")
        return []
    
    # 使用辅助函数转换标签为层
    layers, ordered_labels = _labels_to_layers(labels, depth_list, avg_density, n_clusters)

    if isshow:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：聚类散点图
        ax1 = axes[0]
        cmap = plt.get_cmap('tab10')
        for label in np.unique(ordered_labels):
            mask = ordered_labels == label
            color = cmap(label % 10)
            ax1.scatter(depth_list[mask], avg_density[mask], c=[color], 
                       label=f'Cluster {label+1}', s=50, alpha=0.7)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Average Density')
        ax1.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
        ax1.legend()
        ax1.grid(True)
        
        # 右图：分层结果
        ax2 = axes[1]
        ymax = max(avg_density) if np.any(avg_density) else 1.0
        for L in layers:
            color = cmap((L['layer']-1) % 10)
            ax2.axvspan(L['start'], L['end'], color=color, alpha=0.18)
            mid = (L['start'] + L['end']) / 2
            ax2.text(mid, ymax*0.95, f"Layer {L['layer']}", ha='center', va='top', fontsize=9, color=color)
        ax2.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
        ax2.set_xlabel('Cell Depth (to GM)')
        ax2.set_ylabel('Average Cell Density')
        ax2.set_title(f'Segmented Layers (n={len(layers)}) [DBSCAN]')
        ax2.grid(True)
        
        plt.tight_layout()
        if issave:
            plt.savefig('depth_density_layers_dbscan.png')
        plt.show()

    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'DBSCAN')

    return layers


def customSegmentPipeline(avg_density, depth_list, isshow=True, issave=True):
    # 分步分层
    # 初始化Layers列表
    depth_list = np.asarray(depth_list)
    avg_density = np.asarray(avg_density)
    layers = []
    
    L1_scale = depth_list <= 0.2
    depth_l1 = depth_list[L1_scale]
    density_l1 = avg_density[L1_scale]
    # 寻找density_l1的波谷位置作为L1/L2边界
    if len(density_l1) >= 3:
        valleys, _ = find_peaks(-density_l1, distance=5)
        if len(valleys) > 0:
            L1_edge = depth_l1[valleys[0]]
        else:
            L1_edge = 0.1

    # 计算0-0.1深度范围内的平均密度
    mask_l1 = depth_list < L1_edge
    mean_density_l1 = np.mean(avg_density[mask_l1]) if np.any(mask_l1) else 0
    L1 = {'layer': 1, 'start': 0.0, 'end': L1_edge, 'mean_density': mean_density_l1}
    layers.append(L1)

    # L2-L6: GMM 3聚类
    mask_rest = depth_list >= L1_edge
    depth_rest = depth_list[mask_rest]
    density_rest = avg_density[mask_rest]
    
    # 特征归一化
    d_min, d_ptp = depth_rest.min(), np.ptp(depth_rest) + 1e-8
    rho_min, rho_ptp = density_rest.min(), np.ptp(density_rest) + 1e-8
    depth_scaled = (depth_rest - d_min) / d_ptp
    dens_scaled = (density_rest - rho_min) / rho_ptp
    X_norm = np.vstack([depth_scaled, dens_scaled]).T

    # 在density_rest中寻找2个波峰位置作为L2/3和L4的初始中心
    peaks, _ = find_peaks(avg_density[(L1_edge <= depth_list) & (depth_list <= 0.8)], distance=5)
    if len(peaks) >= 3:
        left_peak = peaks[0]
        right_peak = peaks[-1]
        peak_scale = right_peak - left_peak
        left_scale = [x for x in peaks if x - left_peak < peak_scale / 2]
        right_scale = [x for x in peaks if right_peak - x < peak_scale / 2]
        # 找到left_scale中的最高峰作为L2/3中点
        L23_mid = depth_rest[left_scale][np.argmax(density_rest[left_scale])]
        # 找到right_scale中的最高峰作为L4中点
        L4_mid = depth_rest[right_scale][np.argmax(density_rest[right_scale])]
        L56_mid = 1 - (1 - L4_mid) / 2
    elif len(peaks) == 2:
        # 左边的波峰作为L2/3中点，右边的波峰作为L4中点
        peak_depths = depth_rest[peaks]
        L23_mid = peak_depths.min()
        L4_mid = peak_depths.max()
        L56_mid = 1 - (1 - L4_mid) / 2
    else:
        L23_mid = 0.18
        L4_mid = 0.5
        L56_mid = 1 - (1 - L4_mid) / 2

    # 使用GMM进行3分类
    n_clusters = 3
    # # 定义初始聚类中心：[归一化深度, 归一化密度]
    # 通过插值获取对应深度的密度值作为初始中心
    target_depths_norm = np.array([L23_mid, L4_mid, L56_mid])
    sort_idx_rest = np.argsort(depth_scaled)
    target_dens_norm = np.interp(target_depths_norm, depth_scaled[sort_idx_rest], dens_scaled[sort_idx_rest])
    init_means = np.column_stack([target_depths_norm, target_dens_norm])
    
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', 
                            random_state=42, n_init=1, reg_covar=1e-2, max_iter=100, means_init=init_means)
    labels = gmm.fit_predict(X_norm)

    # # 使用KMeans进行3分类
    # n_clusters = 3
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(X_norm)
    
    # 将簇按深度中心排序，保证层号与深度呈单调关系
    cluster_mean_depth = []
    for k in range(n_clusters):
        cluster_mask = labels == k
        if np.any(cluster_mask):
            cluster_mean_depth.append((k, depth_rest[cluster_mask].mean()))
        else:
            cluster_mean_depth.append((k, np.inf))
    # 按深度排序
    ordered = [k for k, _ in sorted(cluster_mean_depth, key=lambda x: x[1])]

    # 重新映射标签
    label_map = {old: new for new, old in enumerate(ordered)}
    ordered_labels = np.array([label_map[l] for l in labels])

    prob = gmm.predict_proba(X_norm)
    # 重新排列概率矩阵的列顺序
    prob = prob[:, ordered]
    print(prob)
    # 重新分配概率模糊的点到第1和第3簇
    max_proba = prob.max(axis=1)
    fuzzy_threshold = 0.59
    fuzzy_mask = max_proba < fuzzy_threshold
    for i in np.where(fuzzy_mask)[0]:
        if prob[i, 0] > prob[i, 2]:
            ordered_labels[i] = 0
        else:
            ordered_labels[i] = 2

    # 计算每个层的边界（确保不重叠）
    # 按深度排序数据
    sort_idx = np.argsort(depth_rest)
    depth_sorted = depth_rest[sort_idx]
    labels_sorted = ordered_labels[sort_idx]
    density_sorted = density_rest[sort_idx]
    
    # 找到每个簇的边界，确保不重叠
    prev_end = L1_edge  # L1的结束位置
    layerlist = [2, 4, 5]
    for cluster_idx in range(n_clusters):
        cluster_mask = labels_sorted == cluster_idx
        if not np.any(cluster_mask):
            continue
        
        cluster_depths = depth_sorted[cluster_mask]
        cluster_densities = density_sorted[cluster_mask]
        
        # 该簇的起始位置为前一层的结束位置
        start = prev_end
        # 该簇的结束位置为簇内最大深度
        end = cluster_depths.max()
        
        # 如果是最后一个簇，结束位置设为1.0
        if cluster_idx == n_clusters - 1:
            end = 1.0
        else:
            # 找到下一个簇的最小深度，取中点作为边界
            next_cluster_mask = labels_sorted == cluster_idx + 1
            if np.any(next_cluster_mask):
                next_min = depth_sorted[next_cluster_mask].min()
                # 边界设为当前簇最大值和下一簇最小值的中点
                end = (cluster_depths.max() + next_min) / 2
        
        # 计算该层的平均密度（使用原始深度范围内的数据）
        layer_mask = (depth_rest >= start) & (depth_rest < end)
        if cluster_idx == n_clusters - 1:
            layer_mask = (depth_rest >= start) & (depth_rest <= end)
        mean_d = float(np.mean(density_rest[layer_mask])) if np.any(layer_mask) else 0
        
        layer = {
            'layer': layerlist[cluster_idx],
            'start': float(start),
            'end': float(end),
            'mean_density': mean_d
        }
        layers.append(layer)
        prev_end = end
    
    # 细分L2为L2和L3
    if len(layers) >= 2:
        L2 = layers[1]
        # 提取L2范围内的数据
        mask_l2 = (depth_list >= L2['start']) & (depth_list < L2['end'])
        depth_l2 = depth_list[mask_l2]
        density_l2 = avg_density[mask_l2]
        
        if len(depth_l2) > 0:
            # 使用梯度法细分L2为L2和L3
            # sub_layers = segmentLayer_gradient(density_l2, depth_l2, n_layers=2, 
            #                                    sigma=1, isshow=False, issave=False)
            # 使用kmeans法细分L2为L2和L3
            sub_layers = segmentLayer_kmeans(density_l2, depth_l2, n_clusters=2, isshow=False, issave=False)
            if len(sub_layers) == 2:
                # 更新L2和添加L3
                L2_updated = sub_layers[0]
                L3 = sub_layers[1]
                L3['end'] = L2['end']  # 保持L3的结束位置为原L2的结束位置
                
                L2['end'] = L2_updated['end']
                L2['mean_density'] = L2_updated['mean_density']
                
                L3_dict = {
                    'layer': 3,
                    'start': L2['end'],
                    'end': L3['end'],
                    'mean_density': L3['mean_density']
                }
                layers.insert(2, L3_dict)  # 插入到L2后面
    # 细分L5为L5和L6
    if len(layers) >= 5:
        L5 = layers[4]
        # 提取L5范围内的数据
        mask_l5 = (depth_list >= L5['start']) & (depth_list < L5['end'])
        depth_l5 = depth_list[mask_l5]
        density_l5 = avg_density[mask_l5]
        
        if len(depth_l5) > 0:
            # 使用梯度法细分L5为L5和L6
            # sub_layers = segmentLayer_gradient(density_l5, depth_l5, n_layers=2, 
            # #                                    sigma=1, isshow=False, issave=False)
            sub_layers = segmentLayer_kmeans(density_l5, depth_l5, n_clusters=2, isshow=False, issave=False)
            if len(sub_layers) == 2:
                # 更新L5和添加L6
                L5_updated = sub_layers[0]
                L6 = sub_layers[1]
                
                L5['end'] = L5_updated['end']
                L5['mean_density'] = L5_updated['mean_density']
                
                L6_dict = {
                    'layer': 6,
                    'start': L5['end'],
                    'end': L6['end'],
                    'mean_density': L6['mean_density']
                }
                layers.insert(5, L6_dict)  # 插入到L5后面
    
    # for L in layers:
    #     if L['layer'] == 4:
    #         L['end'] = 0.71;
    #     if L['layer'] == 5:
    #         L['start'] = 0.71;
    

    if isshow:
        _visualize_layers(layers, depth_list, avg_density, 'Multi-Method', 
                          issave, 'depth_density_layers_custom.png')
    
    # 计算层间密度差异验证
    _compute_layer_density_diff(layers, 'Custom')

    return layers


def segmentLayer(avg_density, depth_list, offset=True, isshow=True, issave=True, method='kmeans', n_clusters=6, **kwargs):
    method = method.lower()
    
    if method == 'kmeans':
        return segmentLayer_kmeans(avg_density, depth_list, n_clusters=n_clusters, 
                                   offset=offset, isshow=isshow, issave=issave)
    elif method == 'gradient':
        sigma = kwargs.get('sigma', 2)
        return segmentLayer_gradient(avg_density, depth_list, n_layers=n_clusters,
                                     sigma=sigma, isshow=isshow, issave=issave)
    elif method == 'second_derivative' or method == '2nd_derivative':
        sigma = kwargs.get('sigma', 3)
        return segmentLayer_second_derivative(avg_density, depth_list, n_layers=n_clusters,
                                              sigma=sigma, isshow=isshow, issave=issave)
    elif method == 'gmm':
        return segmentLayer_gmm(avg_density, depth_list, n_clusters=n_clusters,
                                isshow=isshow, issave=issave)
    elif method == 'multi_threshold':
        return segmentLayer_multi_threshold(avg_density, depth_list, n_layers=n_clusters,
                                            isshow=isshow, issave=issave)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.15)
        min_samples = kwargs.get('min_samples', 2)
        return segmentLayer_dbscan(avg_density, depth_list, eps=eps, min_samples=min_samples,
                                   isshow=isshow, issave=issave)
    elif method == 'custom':
        return customSegmentPipeline(avg_density, depth_list, isshow=isshow, issave=issave)
    else:
        raise ValueError(f"未知的分层方法: {method}. 可选方法: 'kmeans', 'gradient', 'second_derivative', 'gmm', 'multi_threshold', 'dbscan'")
    


if __name__ == "__main__":
    wm = "input/WM_40x.csv"
    gm = "input/GM_40x.csv"
    cell = "output/nuclei_centroids.csv"
    depth, density = analyze(wm, gm, cell)

    visualize(depth, density, issave=True)
    avg_density, depth_list = computeAverage(depth, density, mode='median', isshow=True, issave=True, issmooth=False)

    # avg_density = gaussian_filter1d(avg_density, sigma=2)

    # 可视化平均密度曲线
    plt.figure(figsize=(8, 5))
    plt.plot(depth_list, avg_density, marker='o', linestyle='-', color='C1')
    plt.xlabel('Cell Depth (to GM)')
    plt.ylabel('Average Cell Density')
    plt.title('Average Cell Density vs. Depth')
    plt.grid(True)
    plt.show()
    
    
    # 1. 'gradient': 密度梯度分析法
    # 2. 'second_derivative': 二阶导数方法
    # 3. 'kmeans': KMeans聚类（默认）
    # 4. 'gmm': 高斯混合模型
    # 5. 'multi_threshold': 多阈值分割法
    # 6. 'DBSCAN': 密度聚类
    # 7. 'custom': 自定义
    
    method = 'custom'  # 修改此处以切换方法
    layers = segmentLayer(avg_density, depth_list, offset=False, isshow=True, issave=True, 
                          method=method, n_clusters=4)

    # layers = customSegmentPipeline(avg_density, depth_list, isshow=True, issave=False)
    
    # 保存分层结果为 CSV 文件
    layers_df = pd.DataFrame(layers)
    layers_df.to_csv(os.path.join(output_dir, 'segmented_layers.csv'), index=False)
    
    # # 对比所有方法
    # methods = ['kmeans', 'gradient', 'second_derivative', 'gmm', 'multi_threshold']
    # for m in methods:
    #     print(f"\n=== 使用方法: {m} ===")
    #     layers = segmentLayer(avg_density, depth_list, method=m, isshow=True, issave=False, n_clusters=4, offset=False)
