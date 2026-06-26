#!/usr/bin/env python
"""
从 4x 明场/组织图像中自动分割灰质 GM、白质 WM 和 gray mask，并映射到 40x/DAPI 图像坐标系。

默认输入文件位于 input 目录：
  - 4x.png   ：低倍 4x 图像，用于组织、白质、灰质区域分割
  - 4x.json  ：4x 扫描元数据，用于坐标映射
  - dapi.png ：40x/DAPI 图像，用于确定最终画布尺寸
  - 40x.json ：40x 扫描元数据，用于坐标映射

主要流程：
  1. 可选地对 4x 图像降采样，加速组织/白质/灰质分割。
  2. 在 4x 图像上提取 tissue mask、white mask、gray mask。
  3. 基于 gray contour 到 white contour 的距离，将 gray contour 分成 GM 外边界和 WM 内边界。
  4. 如果使用了降采样，将 mask 恢复到原始 4x 尺寸，并将 GM/WM 边界按比例放回原坐标。
  5. 对恢复后的边界做插值和平滑，补齐缩放导致的点间空隙。
  6. 根据 4x/40x JSON 元数据，将 GM/WM 和 gray mask 映射到 40x/DAPI 坐标。

默认输出：
  - GM_4x.csv / WM_4x.csv                 ：原始 4x 坐标边界
  - GM_40x.csv / WM_40x.csv               ：映射到 40x 坐标后的完整边界
  - GM_40x_clipped.csv / WM_40x_clipped.csv：裁剪到 40x 图像范围内的边界
  - GM.csv / WM.csv / mask.png            ：下游 run_pipeline.py 默认使用的别名文件
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------- 4x 分割参数窗口 ----------------
# 几何/速度参数。1.0 表示沿用原始 4x 全分辨率分割；
# 0.5 / 0.25 表示先将 4x 图像宽高缩小到 1/2 或 1/4 后再分割，
# 最后再把 mask 和 GM/WM 边界恢复到原始 4x 坐标。
SEGMENT_4X_DOWNSAMPLE_RATE = 0.2

# Tissue mask 提取参数。
# TISSUE_OTSU_MULTIPLIER 越大，阈值越高，倾向于保留更深/更暗的组织区域。
# TISSUE_MORPH_KERNEL 控制开闭运算尺度，用于去除小噪声、填补组织区域内的小孔洞。
# TISSUE_BORDER_CLEAR: 形态学操作前在阈值图边界清除的像素宽度（原始4x坐标），
#   防止组织区域离图像边界太近时被大核操作"拉"到边界上形成突出。
#   0 表示不清除。
TISSUE_GAUSSIAN_KERNEL = 5
TISSUE_OTSU_MULTIPLIER = 1.0
TISSUE_MORPH_KERNEL = 51
TISSUE_BORDER_CLEAR = 50
TISSUE_OPEN_ITERATIONS = 2
TISSUE_CLOSE_ITERATIONS = 2

# White matter mask 提取参数。
# WHITE_OTSU_MULTIPLIER 控制白质阈值相对 Otsu 阈值的比例。
# WHITE_MORPH_KERNEL 控制白质区域平滑和去噪尺度。
# WHITE_DILATE_KERNEL 用于适当扩张白质区域，使后续 WM 边界更贴近灰白交界。
WHITE_OTSU_MULTIPLIER = 0.35
WHITE_MORPH_KERNEL = 51
WHITE_OPEN_ITERATIONS = 2
WHITE_CLOSE_ITERATIONS = 2
WHITE_DILATE_KERNEL = 51
WHITE_DILATE_ITERATIONS = 1

# Gray matter mask 提取参数。
# gray mask 由 tissue mask 减去 white mask 得到，再用形态学操作清理边缘。
GRAY_MORPH_KERNEL = 91
GRAY_CLOSE_ITERATIONS = 2
GRAY_OPEN_ITERATIONS = 2

# 形态学操作 padding 参数。
# 在 open/close/dilate 前先给 mask 四周补一圈 0，再在操作后裁回原图尺寸。
# 这样可以减少大核形态学操作在图像边缘处产生的粘连/贴边伪影。
MORPH_PADDING = 128

# GM/WM 边界拆分和降采样恢复参数。
# BOUNDARY_DISTANCE_THRESHOLD：gray contour 上距离 white contour 足够远的点归为 GM，
# 距离 white contour 较近的点归为 WM。
# BOUNDARY_INTERPOLATION_STEP：边界恢复到原图后，相邻采样点的目标间距，越小越密。
# BOUNDARY_SMOOTH_KERNEL：边界坐标移动平均窗口，用于减少降采样恢复后的锯齿。
BOUNDARY_DISTANCE_THRESHOLD = 300
BOUNDARY_INTERPOLATION_STEP = 1.0
BOUNDARY_SMOOTH_KERNEL = 601
BOUNDARY_CHAIKIN_ITERATIONS = 3
BOUNDARY_GAUSSIAN_SIGMA = 15.0
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)


def _scaled_odd(value: int, scale: float, minimum: int = 3) -> int:
    """按降采样比例缩放形态学/平滑核大小，并保证核大小为奇数。"""
    size = max(minimum, int(round(float(value) * float(scale))))
    if size % 2 == 0:
        size += 1
    return size


def _scaled_distance(value: float, scale: float) -> float:
    """按降采样比例缩放距离阈值，避免在小图上仍使用原图尺度。"""
    return max(1.0, float(value) * float(scale))


def _scaled_padding(value: int, scale: float) -> int:
    """按降采样比例缩放形态学 padding 大小。"""
    return max(0, int(round(float(value) * float(scale))))


def _validate_rate(rate: float) -> float:
    """检查降采样比例是否合法。rate 必须在 (0, 1] 范围内。"""
    rate = float(rate)
    if rate <= 0 or rate > 1:
        raise ValueError(f"4x downsample rate must be in (0, 1], got {rate}")
    return rate


def _operation_padding(kernel: np.ndarray, iterations: int, param_scale: float) -> int:
    """
    计算一次形态学操作实际使用的 padding。

    padding 至少覆盖核半径乘以迭代次数；否则大核 close/dilate 仍可能在裁剪边缘处
    受到边界条件影响。
    """
    configured = _scaled_padding(MORPH_PADDING, param_scale)
    radius = max(kernel.shape[:2]) // 2
    required = radius * max(int(iterations), 1) + 2
    return max(configured, required)


def morph_with_padding(
    mask: np.ndarray,
    op: int,
    kernel: np.ndarray,
    iterations: int = 1,
    param_scale: float = 1.0,
) -> np.ndarray:
    """
    带 0-padding 的 morphologyEx。

    形态学操作前先扩展画布，操作完成后裁回原始尺寸，降低组织/白质/灰质区域在
    图像边缘被形态学核错误拉伸或粘连到边界的概率。
    """
    pad = _operation_padding(kernel, iterations, param_scale)
    if pad <= 0:
        return cv2.morphologyEx(
            mask,
            op,
            kernel,
            iterations=iterations,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    morphed = cv2.morphologyEx(
        padded,
        op,
        kernel,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return morphed[pad:-pad, pad:-pad]


def dilate_with_padding(
    mask: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 1,
    param_scale: float = 1.0,
) -> np.ndarray:
    """
    带 0-padding 的 dilation。

    白质膨胀最容易把区域推到图像边界上，因此这里和 open/close 一样先 padding
    再裁回原图尺寸。
    """
    pad = _operation_padding(kernel, iterations, param_scale)
    if pad <= 0:
        return cv2.dilate(
            mask,
            kernel,
            iterations=iterations,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    dilated = cv2.dilate(
        padded,
        kernel,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dilated[pad:-pad, pad:-pad]


def read_gray_image(path: Path) -> np.ndarray:
    """以灰度方式读取图像；4x 图像走这个入口。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def read_color_image(path: Path) -> np.ndarray:
    """以彩色方式读取图像；40x/DAPI 图像用于可视化和确定输出尺寸。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def resolve_image_path(input_dir: Path, filename: str) -> Path:
    """按精确路径优先，否则用同名 stem 自动匹配 tif/png/jpg 等常见图像格式。"""
    raw = Path(filename)
    base = raw if raw.is_absolute() else input_dir / raw
    if base.exists():
        return base

    stem = base.stem if base.suffix else base.name
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = base.parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    matches = [p for p in base.parent.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        priority = {ext: idx for idx, ext in enumerate(IMAGE_EXTENSIONS)}
        return sorted(matches, key=lambda p: priority.get(p.suffix.lower(), 999))[0]

    return base


def largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留二值 mask 中面积最大的连通域，去掉离散噪声区域。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    if num_labels <= 1:
        return out
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out[labels == largest_label] = 255
    return out


def crop_tissue(image: np.ndarray, param_scale: float = 1.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    从 4x 图像中提取整体组织区域 tissue mask。

    输入:
        image: 4x 灰度图，可能是原图，也可能是降采样后的工作图。
        param_scale: 当前工作图相对原图的比例，用于同步缩放核大小。

    输出:
        tissue_mask: 二值组织 mask，白色区域表示组织。
        contours: tissue mask 的外轮廓。
    """
    # 先用高斯模糊降低局部噪声，再用 Otsu 自动估计组织/背景阈值。
    blur_kernel = _scaled_odd(TISSUE_GAUSSIAN_KERNEL, param_scale, minimum=3)
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold *= TISSUE_OTSU_MULTIPLIER
    _, thresholded = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

    # 清除图像边界区域，防止组织与图像边界粘连形成突出。
    border = max(1, int(round(TISSUE_BORDER_CLEAR * param_scale)))
    if border > 0:
        thresholded[:border, :] = 0
        thresholded[-border:, :] = 0
        thresholded[:, :border] = 0
        thresholded[:, -border:] = 0

    # 大核开闭运算用于去掉背景小噪点，并填补组织内部空洞。
    k = _scaled_odd(TISSUE_MORPH_KERNEL, param_scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = morph_with_padding(
        thresholded,
        cv2.MORPH_OPEN,
        kernel,
        iterations=TISSUE_OPEN_ITERATIONS,
        param_scale=param_scale,
    )
    closed = morph_with_padding(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=TISSUE_CLOSE_ITERATIONS,
        param_scale=param_scale,
    )

    tissue_mask = largest_component(closed)
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tissue_mask, contours, -1, 255, thickness=cv2.FILLED)
    return tissue_mask, contours


def crop_white(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    tissue_image: np.ndarray,
    param_scale: float = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    在 tissue mask 内提取白质区域 white mask。

    输入:
        image: 4x 灰度图。
        tissue_mask: 上一步得到的整体组织 mask。
        tissue_image: image 与 tissue_mask 相乘后的组织图。
        param_scale: 当前工作图相对原图的比例。

    输出:
        white_mask: 二值白质 mask。
        contours: 白质区域外轮廓。
    """
    # 只在 tissue mask 内统计灰度直方图，避免背景像素影响白质阈值。
    hist = cv2.calcHist([image], [0], tissue_mask, [256], [0, 256]).flatten()
    total_pixels = float(hist.sum())
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    weight_b = 0.0
    maximum = 0.0
    threshold = 0

    for i in range(256):
        weight_b += float(hist[i])
        if weight_b == 0:
            continue
        weight_f = total_pixels - weight_b
        if weight_f == 0:
            break
        sum_b += i * float(hist[i])
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > maximum:
            maximum = between
            threshold = i

    # 白质阈值使用 Otsu 阈值的倍率，倍率是一个关键可调参数。
    threshold *= WHITE_OTSU_MULTIPLIER
    _, result = cv2.threshold(tissue_image, threshold, 255, cv2.THRESH_BINARY_INV)
    white_mask = cv2.bitwise_and(result, tissue_mask)

    # 对白质 mask 做开闭运算，并只保留最大连通域。
    k = _scaled_odd(WHITE_MORPH_KERNEL, param_scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    white_mask = morph_with_padding(
        white_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=WHITE_OPEN_ITERATIONS,
        param_scale=param_scale,
    )
    white_mask = morph_with_padding(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=WHITE_CLOSE_ITERATIONS,
        param_scale=param_scale,
    )
    white_mask = largest_component(white_mask)

    # 轻微膨胀白质区域，使灰白交界处的 WM 边界更稳定。
    dk = _scaled_odd(WHITE_DILATE_KERNEL, param_scale)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    white_mask = dilate_with_padding(
        white_mask,
        dilate_kernel,
        iterations=WHITE_DILATE_ITERATIONS,
        param_scale=param_scale,
    )

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(white_mask, contours, -1, 255, thickness=cv2.FILLED)
    return white_mask, contours


def extract_gray(
    tissue_mask: np.ndarray,
    white_mask: np.ndarray,
    param_scale: float = 1.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    从 tissue mask 中扣除 white mask，得到灰质区域 gray mask。

    输出:
        gray_mask: 二值灰质 mask。
        contours: 灰质区域外轮廓。后续会根据它与白质轮廓的距离拆分 GM/WM。
    """
    gray_mask = cv2.bitwise_and(tissue_mask, cv2.bitwise_not(white_mask))
    k = _scaled_odd(GRAY_MORPH_KERNEL, param_scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray_mask = morph_with_padding(
        gray_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=GRAY_CLOSE_ITERATIONS,
        param_scale=param_scale,
    )
    gray_mask = morph_with_padding(
        gray_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=GRAY_OPEN_ITERATIONS,
        param_scale=param_scale,
    )
    gray_mask = largest_component(gray_mask)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return gray_mask, contours


def _append_boundary_run(
    out: list[np.ndarray],
    run: list[list[list[int]]],
    min_points: int,
) -> None:
    """Append one continuous GM/WM boundary run as an independent open polyline."""
    if len(run) >= min_points:
        out.append(np.asarray(run, dtype=np.int32))


def compute_boundaries(
    white_contours: list[np.ndarray],
    gray_contours: list[np.ndarray],
    param_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    将 gray contour 拆分为 GM 外边界和 WM 内边界。

    算法:
        对 gray contour 上的每个点，计算它到 white contour 的最短距离。
        - 距离大于 BOUNDARY_DISTANCE_THRESHOLD：认为是远离白质的灰质外边界 GM。
        - 距离小于等于阈值：认为靠近灰白交界，归为 WM 边界。

    注意:
        这里是 4x 分割里最耗时的步骤之一；降采样会显著减少 contour 点数。
    """
    distance_threshold = _scaled_distance(BOUNDARY_DISTANCE_THRESHOLD, param_scale)
    gm_contours: list[np.ndarray] = []
    wm_contours: list[np.ndarray] = []

    if not white_contours:
        return gray_contours, wm_contours

    for cnt in gray_contours:
        current_run: list[list[list[int]]] = []
        current_is_gm: bool | None = None

        for pt in cnt:
            x, y = int(pt[0][0]), int(pt[0][1])
            min_dist = float("inf")
            # pointPolygonTest(..., measureDist=True) 返回点到轮廓的有符号距离；
            # 这里取绝对值，只关心与白质轮廓的距离大小。
            for white_cnt in white_contours:
                dist = abs(cv2.pointPolygonTest(white_cnt, (x, y), True))
                min_dist = min(min_dist, dist)
                if min_dist <= 0:
                    break

            is_gm = min_dist > distance_threshold
            if current_is_gm is None:
                current_is_gm = is_gm
                current_run = [[[x, y]]]
                continue

            if is_gm == current_is_gm:
                current_run.append([[x, y]])
                continue

            # 边界类别发生切换时，先结束当前连续片段。
            # 不能把所有 GM 点或 WM 点收进同一个列表，否则绘制 OuterInnerPoints.png 时
            # 不连续片段会被 cv2.polylines 硬连起来，看起来像边界闭合或乱连。
            if current_is_gm:
                _append_boundary_run(gm_contours, current_run, min_points=3)
            else:
                _append_boundary_run(wm_contours, current_run, min_points=2)
            current_is_gm = is_gm
            current_run = [[[x, y]]]

        if current_run:
            if current_is_gm:
                _append_boundary_run(gm_contours, current_run, min_points=3)
            else:
                _append_boundary_run(wm_contours, current_run, min_points=2)

    return gm_contours, wm_contours


def contours_to_points(contours: list[np.ndarray]) -> list[tuple[int, int]]:
    """将 OpenCV contour 列表展开为 [(x, y), ...]，便于保存 CSV 和坐标变换。"""
    points: list[tuple[int, int]] = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        points.extend((int(x), int(y)) for x, y in pts)
    return points


def _resample_polyline(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    """
    沿折线弧长重新采样边界点。

    降采样后再放大回原分辨率时，原始 contour 点之间会出现较大间隔。
    这里按固定步长插值补点，使 GM/WM 边界在原始 4x 尺寸上更连续。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return pts

    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    keep = np.concatenate([[True], seg_len > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        return pts

    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    distance = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(distance[-1])
    if total <= 1e-6:
        return pts

    step = max(float(step), 0.25)
    sample_count = max(int(np.ceil(total / step)) + 1, len(pts))
    sample_distance = np.linspace(0.0, total, sample_count)
    x = np.interp(sample_distance, distance, pts[:, 0])
    y = np.interp(sample_distance, distance, pts[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def _chaikin_smooth_polyline(points: np.ndarray, iterations: int) -> np.ndarray:
    """
    用 Chaikin corner cutting 平滑折线。

    该方法会把每个尖锐折角切成两段更缓和的线段，特别适合降低从低分辨率
    contour 放大回原图后产生的阶梯感。这里按开放折线处理，保留首尾点。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    iterations = max(int(iterations), 0)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = np.asarray(new_pts, dtype=np.float32)
    return pts


def _gaussian_kernel1d(sigma: float, max_kernel_size: int) -> np.ndarray:
    """构建一维高斯核，长度受 BOUNDARY_SMOOTH_KERNEL 限制。"""
    sigma = float(sigma)
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)

    radius = max(1, int(round(sigma * 3.0)))
    max_kernel_size = max(3, int(max_kernel_size))
    if max_kernel_size % 2 == 0:
        max_kernel_size -= 1
    radius = min(radius, max_kernel_size // 2)

    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _smooth_polyline(points: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    对边界坐标做高斯平滑。

    相比简单移动平均，高斯平滑对局部阶梯和噪声更稳定，不容易把整段边界拉成
    过度平直的线。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if kernel_size <= 1 or len(pts) < 3 or BOUNDARY_GAUSSIAN_SIGMA <= 0:
        return pts

    kernel = _gaussian_kernel1d(BOUNDARY_GAUSSIAN_SIGMA, kernel_size)
    if len(kernel) <= 1:
        return pts

    pad = len(kernel) // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    smoothed = np.column_stack([x, y]).astype(np.float32)
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]
    return smoothed


def restore_boundary_contours(
    contours: list[np.ndarray],
    rate: float,
    output_shape: tuple[int, int],
    interpolation_step: float = BOUNDARY_INTERPOLATION_STEP,
    smooth_kernel: int = BOUNDARY_SMOOTH_KERNEL,
) -> list[np.ndarray]:
    """
    将低分辨率 contour 恢复到原始 4x 坐标，并补齐放缩带来的边界空隙。

    输入:
        contours: 降采样图像中得到的 contour。
        rate: 降采样比例。例如 0.25 表示当前 contour 坐标是原图的 1/4。
        output_shape: 原始 4x 图像尺寸 (height, width)。
        interpolation_step: 插值后相邻点的目标距离。
        smooth_kernel: 移动平均平滑窗口。

    输出:
        restored: OpenCV contour 格式，坐标已经回到原始 4x 图像。
    """
    rate = _validate_rate(rate)
    out_h, out_w = output_shape
    scale = 1.0 / rate
    restored: list[np.ndarray] = []

    for contour in contours:
        # 先按 1/rate 放大回原始 4x 坐标，再沿边界插值补点和平滑。
        pts = contour.reshape(-1, 2).astype(np.float32) * scale
        pts = _resample_polyline(pts, step=interpolation_step)
        pts = _chaikin_smooth_polyline(pts, BOUNDARY_CHAIKIN_ITERATIONS)
        pts = _smooth_polyline(pts, smooth_kernel)
        pts = _resample_polyline(pts, step=interpolation_step)
        if len(pts) < 2:
            continue

        pts[:, 0] = np.clip(pts[:, 0], 0, out_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, out_h - 1)
        pts_i = np.rint(pts).astype(np.int32)

        # 平滑和取整后可能产生重复点，去重可减少后续保存和绘制开销。
        dedup = [pts_i[0]]
        for pt in pts_i[1:]:
            if pt[0] != dedup[-1][0] or pt[1] != dedup[-1][1]:
                dedup.append(pt)
        if len(dedup) >= 2:
            restored.append(np.asarray(dedup, dtype=np.int32).reshape(-1, 1, 2))

    return restored


def restore_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """将低分辨率二值 mask 用最近邻插值恢复到原始 4x 尺寸。"""
    out_h, out_w = output_shape
    if mask.shape[:2] == (out_h, out_w):
        return mask
    return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def segment_4x_image(
    image: np.ndarray,
    downsample_rate: float | None = None,
) -> dict[str, object]:
    """
    统一执行 4x 图像中的 tissue/white/gray/GM/WM 分割。

    输入:
        image: 原始 4x 灰度图。
        downsample_rate:
            None 时使用文件顶部 SEGMENT_4X_DOWNSAMPLE_RATE；
            1.0 表示原图分割；
            0.5/0.25 等表示先降采样分割，再恢复结果。

    输出:
        dict，包含:
            tissue_mask / white_mask / gray_mask:
                尺寸均为原始 4x 图像尺寸。
            tissue_contours / white_contours / gray_contours:
                恢复到原始 4x 坐标的区域轮廓。
            gm_contours / wm_contours:
                恢复到原始 4x 坐标的 GM/WM 边界。
    """
    rate = SEGMENT_4X_DOWNSAMPLE_RATE if downsample_rate is None else downsample_rate
    rate = _validate_rate(rate)
    full_h, full_w = image.shape[:2]

    if rate < 1:
        # 只在分割阶段降采样；最终输出的 mask 和边界仍回到原始 4x 分辨率。
        small_w = max(1, int(round(full_w * rate)))
        small_h = max(1, int(round(full_h * rate)))
        work_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        print(f"  4x segmentation downsample: {full_w}x{full_h} -> {small_w}x{small_h} (rate={rate})")
    else:
        work_image = image

    # 分割顺序：整体组织 -> 白质 -> 灰质 -> 根据灰质/白质距离拆分 GM/WM。
    tissue_mask_small, tissue_contours_small = crop_tissue(work_image, param_scale=rate)
    tissue_image_small = cv2.bitwise_and(work_image, work_image, mask=tissue_mask_small)
    white_mask_small, white_contours_small = crop_white(
        work_image,
        tissue_mask_small,
        tissue_image_small,
        param_scale=rate,
    )
    gray_mask_small, gray_contours_small = extract_gray(
        tissue_mask_small,
        white_mask_small,
        param_scale=rate,
    )
    gm_small, wm_small = compute_boundaries(white_contours_small, gray_contours_small, param_scale=rate)

    if rate < 1:
        # 降采样分割后，所有 mask 和 contour 都恢复到原始 4x 图像坐标。
        tissue_contours = restore_boundary_contours(tissue_contours_small, rate, image.shape[:2])
        white_contours = restore_boundary_contours(white_contours_small, rate, image.shape[:2])
        gray_contours = restore_boundary_contours(gray_contours_small, rate, image.shape[:2])
        gm_contours = restore_boundary_contours(gm_small, rate, image.shape[:2])
        wm_contours = restore_boundary_contours(wm_small, rate, image.shape[:2])

        tissue_mask = restore_mask(tissue_mask_small, image.shape[:2])
        white_mask = restore_mask(white_mask_small, image.shape[:2])
        gray_mask = restore_mask(gray_mask_small, image.shape[:2])
    else:
        # rate=1 时不做额外插值和平滑，保持原始全分辨率分割行为。
        tissue_contours = tissue_contours_small
        white_contours = white_contours_small
        gray_contours = gray_contours_small
        gm_contours = gm_small
        wm_contours = wm_small
        tissue_mask = tissue_mask_small
        white_mask = white_mask_small
        gray_mask = gray_mask_small

    return {
        "rate": rate,
        "tissue_mask": tissue_mask,
        "white_mask": white_mask,
        "gray_mask": gray_mask,
        "tissue_contours": tissue_contours,
        "white_contours": white_contours,
        "gray_contours": gray_contours,
        "gm_contours": gm_contours,
        "wm_contours": wm_contours,
    }


def save_points_csv(points: list[tuple[int, int]], path: Path) -> None:
    """将边界点保存为 CSV，列名固定为 x,y，供后续 pipeline 直接读取。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points)


def read_scan_metadata(path: Path, image_shape: tuple[int, int]) -> dict:
    """
    读取扫描元数据，并换算出图像在物理坐标系中的覆盖范围。

    JSON 中的 pixel_scale 表示每个像素对应的物理尺寸；
    scan_area 或 positions 用于确定图像左上角和右下角的物理坐标。
    这些信息后续用于建立 4x 像素坐标到 40x 像素坐标的仿射变换。
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scan = data.get("scan_info", {})
    scale = float(scan["pixel_scale"])
    camera_w, camera_h = scan.get("camera_resolution", [image_shape[1], image_shape[0]])
    scan_area = scan.get("scan_area")
    if not scan_area:
        positions = data.get("positions", [])
        if not positions:
            raise ValueError(f"No scan_area or positions in {path}")
        xs = [float(p["x"]) for p in positions]
        ys = [float(p["y"]) for p in positions]
        scan_area = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}

    min_x = float(scan_area["min_x"]) - (float(camera_w) * scale / 2.0)
    min_y = float(scan_area["min_y"]) - (float(camera_h) * scale / 2.0)
    width_phys = image_shape[1] * scale
    height_phys = image_shape[0] * scale
    return {
        "scale": scale,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": min_x + width_phys,
        "max_y": min_y + height_phys,
        "shape": image_shape,
    }


def read_objective_offset(path: Path) -> dict[str, float] | None:
    """
    读取倍镜偏移配置文件。

    JSON 格式:
        {"Obj4X": {"x_um": 59.8, "y_um": -517.7}}

    表示 4x 物镜相对于 40x 物镜的物理偏移量（微米）。
    当显微镜切换倍镜时，4x 镜头的光心与 40x 镜头的光心不重合，
    此偏移用于修正坐标映射中的系统误差。
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    obj4x = data.get("Obj4X", {})
    offset = {
        "x_um": float(obj4x["x_um"]),
        "y_um": float(obj4x["y_um"]),
    }
    print(f"  Objective offset (4x -> 40x): x={offset['x_um']:+.3f} um, y={offset['y_um']:+.3f} um")
    return offset


def affine_4x_to_40x(
    meta_4x: dict,
    meta_40x: dict,
    objective_offset: dict[str, float] | None = None,
) -> np.ndarray:
    """
    构建 4x 像素坐标到 40x 像素坐标的仿射矩阵。

    除扫描范围偏移外，还支持倍镜硬件偏移修正：
        x40 = scale_ratio * x4 + tx
        y40 = scale_ratio * y4 + ty

    Parameters
    ----------
    meta_4x, meta_40x : dict
        通过 read_scan_metadata() 读取的扫描元数据。
    objective_offset : dict or None
        可选的倍镜偏移，格式为 {"x_um": ..., "y_um": ...}。
    """
    scale_ratio = meta_4x["scale"] / meta_40x["scale"]
    tx = (meta_4x["min_x"] - meta_40x["min_x"]) / meta_40x["scale"]
    ty = (meta_4x["min_y"] - meta_40x["min_y"]) / meta_40x["scale"]

    if objective_offset is not None:
        # 4x 镜头的光心偏移在物理坐标系中是固定值，
        # 将其从微米转换到 40x 像素坐标后累加到平移项。
        tx += objective_offset["x_um"] / meta_40x["scale"]
        ty += objective_offset["y_um"] / meta_40x["scale"]

    return np.asarray([[scale_ratio, 0.0, tx], [0.0, scale_ratio, ty]], dtype=np.float32)


def transform_points(points: list[tuple[int, int]], matrix: np.ndarray) -> list[tuple[int, int]]:
    """用 2x3 仿射矩阵批量转换边界点坐标，并四舍五入到像素坐标。"""
    if not points:
        return []
    pts = np.asarray(points, dtype=np.float32)
    x = matrix[0, 0] * pts[:, 0] + matrix[0, 2]
    y = matrix[1, 1] * pts[:, 1] + matrix[1, 2]
    out = np.column_stack([np.rint(x), np.rint(y)]).astype(np.int64)
    return [(int(px), int(py)) for px, py in out]


def clip_points(points: list[tuple[int, int]], width: int, height: int) -> list[tuple[int, int]]:
    """只保留落在目标图像画布内的点，避免后续可视化/分层使用图外边界。"""
    return [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]


def transform_mask(mask: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """将 4x gray mask 映射到 40x/DAPI 图像尺寸，使用最近邻保持二值 mask。"""
    out_h, out_w = output_shape
    return cv2.warpAffine(mask, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)


def split_in_bounds_segments(points: list[tuple[int, int]], width: int, height: int) -> list[np.ndarray]:
    """
    将点序列按是否在图像范围内拆成连续段。

    这样绘制 40x 边界时不会把图外点和图内点硬连成跨越整张图的线。
    """
    segments = []
    current = []
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            current.append((x, y))
        else:
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.int32).reshape(-1, 1, 2))
            current = []
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=np.int32).reshape(-1, 1, 2))
    return segments


def draw_dashed_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int,
    gap_len: int,
) -> None:
    """按照点序列绘制虚线折线，用于 GM/WM 边界可视化。"""
    pts = points.reshape(-1, 2)
    start = 0
    draw = True
    while start < len(pts) - 1:
        end = min(start + (dash_len if draw else gap_len), len(pts))
        if draw and end - start >= 2:
            seg = pts[start:end].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(image, [seg], False, color, thickness, lineType=cv2.LINE_AA)
        start = end
        draw = not draw


def draw_boundary_visualization(
    base_image: np.ndarray,
    gm_segments: list[np.ndarray],
    wm_segments: list[np.ndarray],
    out_path: Path,
    thickness: int,
) -> None:
    """
    在原图上绘制 GM/WM 虚线边界。

    GM 使用绿色虚线，WM 使用红色虚线；线宽由调用方根据图像尺寸传入。
    """
    if base_image.ndim == 2:
        vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_image.copy()

    dash_len = max(18, thickness * 4)
    gap_len = max(12, thickness * 3)
    gm_color = (0, 255, 0)      # green in BGR
    wm_color = (0, 0, 255)      # red in BGR

    for seg in gm_segments:
        draw_dashed_polyline(vis, seg, gm_color, thickness, dash_len, gap_len)
    for seg in wm_segments:
        draw_dashed_polyline(vis, seg, wm_color, thickness, dash_len, gap_len)

    cv2.imwrite(str(out_path), vis)


def collect_parameters(args: argparse.Namespace, image_shape: tuple[int, int]) -> dict:
    """收集当前运行的所有分割参数，保存为 JSON 供后续调试对比。"""
    rate = _validate_rate(args.boundary_downsample_rate if args.boundary_downsample_rate is not None else SEGMENT_4X_DOWNSAMPLE_RATE)
    return {
        "image_shape_4x": {"height": image_shape[0], "width": image_shape[1]},
        "downsample": {
            "SEGMENT_4X_DOWNSAMPLE_RATE": SEGMENT_4X_DOWNSAMPLE_RATE,
            "effective_rate": rate,
        },
        "tissue": {
            "TISSUE_GAUSSIAN_KERNEL": TISSUE_GAUSSIAN_KERNEL,
            "TISSUE_OTSU_MULTIPLIER": TISSUE_OTSU_MULTIPLIER,
            "TISSUE_MORPH_KERNEL": TISSUE_MORPH_KERNEL,
            "TISSUE_BORDER_CLEAR": TISSUE_BORDER_CLEAR,
            "TISSUE_OPEN_ITERATIONS": TISSUE_OPEN_ITERATIONS,
            "TISSUE_CLOSE_ITERATIONS": TISSUE_CLOSE_ITERATIONS,
        },
        "white_matter": {
            "WHITE_OTSU_MULTIPLIER": WHITE_OTSU_MULTIPLIER,
            "WHITE_MORPH_KERNEL": WHITE_MORPH_KERNEL,
            "WHITE_OPEN_ITERATIONS": WHITE_OPEN_ITERATIONS,
            "WHITE_CLOSE_ITERATIONS": WHITE_CLOSE_ITERATIONS,
            "WHITE_DILATE_KERNEL": WHITE_DILATE_KERNEL,
            "WHITE_DILATE_ITERATIONS": WHITE_DILATE_ITERATIONS,
        },
        "gray_matter": {
            "GRAY_MORPH_KERNEL": GRAY_MORPH_KERNEL,
            "GRAY_CLOSE_ITERATIONS": GRAY_CLOSE_ITERATIONS,
            "GRAY_OPEN_ITERATIONS": GRAY_OPEN_ITERATIONS,
        },
        "morph_padding": {
            "MORPH_PADDING": MORPH_PADDING,
        },
        "boundary": {
            "BOUNDARY_DISTANCE_THRESHOLD": BOUNDARY_DISTANCE_THRESHOLD,
            "BOUNDARY_INTERPOLATION_STEP": BOUNDARY_INTERPOLATION_STEP,
            "BOUNDARY_SMOOTH_KERNEL": BOUNDARY_SMOOTH_KERNEL,
            "BOUNDARY_CHAIKIN_ITERATIONS": BOUNDARY_CHAIKIN_ITERATIONS,
            "BOUNDARY_GAUSSIAN_SIGMA": BOUNDARY_GAUSSIAN_SIGMA,
        },
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。CLI 参数优先级高于文件顶部默认参数。"""
    parser = argparse.ArgumentParser(description="Segment 4x GM/WM boundaries and map them to DAPI coordinates.")
    parser.add_argument("--input-dir", default="input", help="Input directory containing 4x.png, 4x.json, dapi.png, 40x.json")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to input-dir.")
    parser.add_argument("--image-4x", default="4x.png", help="4x image filename")
    parser.add_argument("--image-40x", default="dapi.png", help="40x/DAPI image filename")
    parser.add_argument("--json-4x", default="4x.json", help="4x scan metadata filename")
    parser.add_argument("--json-40x", default="40x.json", help="40x scan metadata filename")
    parser.add_argument(
        "--4x-downsample-rate",
        "--boundary-downsample-rate",
        dest="boundary_downsample_rate",
        type=float,
        default=None,
        help=f"4x GM/WM segmentation scale. 1=no downsample, 0.5=half, 0.25=quarter. Default: SEGMENT_4X_DOWNSAMPLE_RATE={SEGMENT_4X_DOWNSAMPLE_RATE}.",
    )
    parser.add_argument("--objective-offset-config", default="ObjectiveOffsetConfig.json", help="Objective offset calibration JSON file")
    parser.add_argument("--no-aliases", action="store_true", help="Do not write GM.csv, WM.csv, and mask.png pipeline aliases")
    return parser.parse_args()


def main() -> None:
    """命令行入口：完成 4x 分割、4x->40x 映射、CSV/mask/可视化输出。"""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_4x_path = resolve_image_path(input_dir, args.image_4x)
    image_40x_path = resolve_image_path(input_dir, args.image_40x)
    json_4x_path = input_dir / args.json_4x
    json_40x_path = input_dir / args.json_40x

    image_4x = read_gray_image(image_4x_path)
    image_40x = read_color_image(image_40x_path)
    h4, w4 = image_4x.shape[:2]
    h40, w40 = image_40x.shape[:2]
    print(f"4x image: {w4} x {h4}")
    print(f"40x/DAPI image: {w40} x {h40}")

    print("Segmenting tissue, white matter, and gray matter masks from 4x image...")
    # 这一步内部会根据 --4x-downsample-rate 决定是否降采样分割；
    # 返回的 mask 和 contour 坐标始终已经恢复到原始 4x 尺寸。
    seg = segment_4x_image(image_4x, downsample_rate=args.boundary_downsample_rate)
    tissue_mask = seg["tissue_mask"]
    white_mask = seg["white_mask"]
    gray_mask = seg["gray_mask"]
    gm_contours = seg["gm_contours"]
    wm_contours = seg["wm_contours"]

    gm_4x = contours_to_points(gm_contours)
    wm_4x = contours_to_points(wm_contours)
    print(f"4x boundaries: GM={len(gm_4x)} points, WM={len(wm_4x)} points")

    cv2.imwrite(str(output_dir / "tissueMask.png"), tissue_mask)
    # cv2.imwrite(str(output_dir / "tissueImage.png"), tissue_image)
    cv2.imwrite(str(output_dir / "whiteMask.png"), white_mask)
    cv2.imwrite(str(output_dir / "grayMask.png"), gray_mask)
    cv2.imwrite(str(output_dir / "grayImage.png"), cv2.bitwise_and(image_4x, image_4x, mask=gray_mask))

    save_points_csv(gm_4x, output_dir / "GM_4x.csv")
    save_points_csv(wm_4x, output_dir / "WM_4x.csv")

    # 4x 原图上的边界检查图，方便观察低倍分割是否合理。
    draw_boundary_visualization(
        image_4x,
        gm_contours,
        wm_contours,
        output_dir / "OuterInnerPoints.png",
        thickness=max(6, min(h4, w4) // 350),
    )
    
    meta_4x = read_scan_metadata(json_4x_path, image_4x.shape[:2])
    meta_40x = read_scan_metadata(json_40x_path, image_40x.shape[:2])

    offset_path = input_dir / args.objective_offset_config
    objective_offset = read_objective_offset(offset_path)

    matrix = affine_4x_to_40x(meta_4x, meta_40x, objective_offset=objective_offset)
    print(f"4x->40x affine matrix:\n{matrix}")

    # 将 4x 边界点映射到 40x/DAPI 坐标；clipped 版本用于后续分层主流程。
    gm_40x = transform_points(gm_4x, matrix)
    wm_40x = transform_points(wm_4x, matrix)
    gm_40x_clipped = clip_points(gm_40x, w40, h40)
    wm_40x_clipped = clip_points(wm_40x, w40, h40)
    print(f"40x converted: GM={len(gm_40x)}, WM={len(wm_40x)}")
    print(f"40x clipped:   GM={len(gm_40x_clipped)}, WM={len(wm_40x_clipped)}")

    save_points_csv(gm_40x, output_dir / "GM_40x.csv")
    save_points_csv(wm_40x, output_dir / "WM_40x.csv")
    save_points_csv(gm_40x_clipped, output_dir / "GM_40x_clipped.csv")
    save_points_csv(wm_40x_clipped, output_dir / "WM_40x_clipped.csv")

    if not args.no_aliases:
        # 下游 run_pipeline.py 默认读取 GM.csv、WM.csv 和 mask.png。
        save_points_csv(gm_40x_clipped, output_dir / "GM.csv")
        save_points_csv(wm_40x_clipped, output_dir / "WM.csv")

    # gray mask 同样从 4x 坐标映射到 40x/DAPI 坐标，作为后续细胞过滤和可视化 ROI。
    gray_mask_40x = transform_mask(gray_mask, matrix, image_40x.shape[:2])
    cv2.imwrite(str(output_dir / "grayMask_40x.png"), gray_mask_40x)
    if not args.no_aliases:
        cv2.imwrite(str(output_dir / "mask.png"), gray_mask_40x)

    gm_40x_segments = split_in_bounds_segments(gm_40x, w40, h40)
    wm_40x_segments = split_in_bounds_segments(wm_40x, w40, h40)
    # draw_boundary_visualization(
    #     image_40x,
    #     gm_40x_segments,
    #     wm_40x_segments,
    #     output_dir / "OuterInnerPoints_40x_clipped.png",
    #     thickness=max(10, min(h40, w40) // 500),
    # )

    # roi_gray = cv2.bitwise_and(image_40x, image_40x, mask=gray_mask_40x)
    # cv2.imwrite(str(output_dir / "roiImage.png"), roi_gray)

    print(f"Done. Outputs saved to: {output_dir.resolve()}")

    params = collect_parameters(args, image_4x.shape[:2])
    params_path = output_dir / "parameters.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"Parameters saved to: {params_path.resolve()}")


if __name__ == "__main__":
    main()
