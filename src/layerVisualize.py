"""
Layer segmentation visualization — per-pixel depth-based color assignment.

Computes depth for each pixel via Euclidean distance transform from GM/WM
boundaries, then assigns layer colors directly from depth thresholds.
No column-by-column interpolation or contour-based boundary detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import pandas as pd
import os

# 默认输入输出路径（可通过参数覆盖）
input_dir = "input"
output_dir = "output"
COMPACT_RATE = 1.0  # 1.0 = 原图可视化；0.1 = 约等于原来的 10:1 压缩
IMAGE_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jp2", ".j2k"
)


def resolve_image_path(directory, filename):
    """Resolve filename in directory, falling back to same-stem common image formats."""
    base = filename if os.path.isabs(filename) else os.path.join(directory, filename)
    if os.path.exists(base):
        return base

    parent = os.path.dirname(base)
    name = os.path.basename(base)
    stem, suffix = os.path.splitext(name)
    if not suffix:
        stem = name

    for ext in IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = os.path.join(parent, stem + candidate_ext)
            if os.path.exists(candidate):
                return candidate
    return base


def load_visualization_image(image_path):
    """Load an image for visualization, supporting tif/tiff/png/jpg/jpeg and similar formats."""
    image_path = str(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    reader = None

    if ext in {".tif", ".tiff"}:
        try:
            import tifffile
            img = tifffile.imread(image_path)
            reader = "tifffile"
            # tifffile may warn about OME discontiguous storage but still return data;
            # validate that the result is a sane image array (2D or 3D)
            if not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
                raise ValueError("unexpected tifffile output")
        except Exception:
            img = io.imread(image_path)
            reader = "skimage"
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        reader = "cv2"
        if img is None:
            img = io.imread(image_path)
            reader = "skimage"

    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img = np.asarray(img)

    # Drop singleton dimensions such as (H, W, 1).
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Convert unsupported channel layouts to a displayable form.
    if img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :3]

    # Normalize non-uint8 images for stable OpenCV visualization/saving.
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # `io.imread` returns RGB/RGBA while OpenCV uses BGR/BGRA.
    if reader == "skimage":
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    return img


def get_coords(df):
    """从DataFrame中获取坐标，兼容大小写列名"""
    if 'x' in df.columns and 'y' in df.columns:
        return df[['x', 'y']].values
    elif 'X' in df.columns and 'Y' in df.columns:
        return df[['X', 'Y']].values
    else:
        return df.iloc[:, :2].values


def _clip_points_to_image(points, h, w, name):
    """Keep only finite boundary points inside the image canvas."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"{name} boundary must be an Nx2 coordinate array")

    pts = pts[:, :2]
    finite = np.isfinite(pts).all(axis=1)
    in_bounds = (
        finite
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )
    removed = int(len(pts) - np.count_nonzero(in_bounds))
    if removed:
        print(f"  [边界裁剪] {name}: 移除图外/无效点 {removed}/{len(pts)}")

    clipped = np.rint(pts[in_bounds]).astype(np.int32)
    if len(clipped) == 0:
        raise ValueError(f"{name} boundary has no points inside image bounds ({w}x{h})")
    return clipped


def _compute_depth_map(h, w, gm_pts, wm_pts):
    """
    对全图每个像素计算归一化深度 (0→1)。

    使用欧几里得距离变换：
      depth = dist_to_GM / (dist_to_GM + dist_to_WM)
    灰质边界 → 0，白质边界 → 1，中间连续过渡。
    边界点会被限制在原始图像画布内，避免图外点影响深度场。
    """
    gm_pts = _clip_points_to_image(gm_pts, h, w, "GM(depth)")
    wm_pts = _clip_points_to_image(wm_pts, h, w, "WM(depth)")

    gm_mask = np.zeros((h, w), dtype=np.uint8)
    wm_mask = np.zeros((h, w), dtype=np.uint8)

    for pt in gm_pts:
        px, py = int(pt[0]), int(pt[1])
        gm_mask[py, px] = 255

    for pt in wm_pts:
        px, py = int(pt[0]), int(pt[1])
        wm_mask[py, px] = 255

    dist_gm = cv2.distanceTransform(255 - gm_mask, cv2.DIST_L2, 5)
    dist_wm = cv2.distanceTransform(255 - wm_mask, cv2.DIST_L2, 5)

    total = dist_gm + dist_wm + 1e-8
    depth = dist_gm / total

    return depth


def _draw_dashed_points(img, pts, color, thickness, dash_len=20, gap_len=15):
    """Draw dashed polylines through an Nx2 point array."""
    n = pts.shape[0]
    i = 0
    while i < n:
        end = min(i + dash_len, n)
        if end - i >= 2:
            seg = pts[i:end].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [seg], False, color, thickness)
        i += dash_len + gap_len


def _draw_dashed_contour(img, contour, color, thickness, dash_len=20, gap_len=15, valid_mask=None):
    """沿轮廓绘制虚线（逐段交替画/不画）。"""
    pts = contour.squeeze()
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 2:
        return

    if valid_mask is None:
        _draw_dashed_points(img, pts, color, thickness, dash_len, gap_len)
        return

    h, w = valid_mask.shape[:2]
    current = []
    for x, y in pts.astype(np.int32):
        if 0 <= x < w and 0 <= y < h and valid_mask[y, x]:
            current.append((x, y))
            continue

        if len(current) >= 4:
            _draw_dashed_points(
                img, np.asarray(current, dtype=np.int32), color, thickness, dash_len, gap_len
            )
        current = []

    if len(current) >= 4:
        _draw_dashed_points(
            img, np.asarray(current, dtype=np.int32), color, thickness, dash_len, gap_len
        )


def assign_layers_to_mask(wm_path, gm_path, layers_csv_path, image_path,
                          issave=True, save_dir=None, mask_img=None,
                          compact=False, compact_rate=None):
    """
    在图像上生成分层颜色填充Mask — 基于逐像素深度值直接赋予层颜色。

    Args:
        wm_path: 白质边界CSV文件路径
        gm_path: 灰质边界CSV文件路径
        layers_csv_path: 分层结果CSV文件路径
        image_path: 原始图像路径
        issave: 是否保存结果
        save_dir: 保存目录（如果为None则使用默认output_dir）
        mask_img: 二值掩码（uint8），用于限定有效组织区域
        compact: 兼容旧参数；True 时等同 compact_rate=0.1
        compact_rate: 可视化降采样比例，1 不压缩，0.1 约等于原 10:1 压缩
    """
    global output_dir
    if save_dir is not None:
        output_dir = save_dir

    # -- 读取分层定义 --
    layers_df = pd.read_csv(layers_csv_path)
    layers = layers_df.to_dict('records')
    print(f"已加载分层定义: {len(layers)} 层")

    # -- 读取原始图像 --
    orig_img = load_visualization_image(image_path)
    if orig_img is None:
        raise ValueError(f"无法读取原始图像: {image_path}")
    h, w = orig_img.shape[:2]

    # -- 加载 GM/WM 边界坐标 --
    wm_df = pd.read_csv(wm_path)
    gm_df = pd.read_csv(gm_path)
    wm_pts = get_coords(wm_df)
    gm_pts = get_coords(gm_df)
    print(f"GM边界点数: {len(gm_pts)}, WM边界点数: {len(wm_pts)}")
    gm_pts = _clip_points_to_image(gm_pts, h, w, "GM")
    wm_pts = _clip_points_to_image(wm_pts, h, w, "WM")
    print(f"  图内边界点数: GM={len(gm_pts)}, WM={len(wm_pts)}")

    # -- 确定组织掩码（mask内像素才会被填色） --
    _mask_loaded_from = None
    mask = None
    if mask_img is not None:
        if mask_img.shape[:2] == (h, w):
            mask = mask_img
            _mask_loaded_from = "外部mask（来自 pipeline）"
        else:
            print(f"  外部mask尺寸 {mask_img.shape[:2]} 与图像 ({h},{w}) 不匹配，自动resize")
            mask = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
            _mask_loaded_from = "外部mask（resized）"
        print(f"  使用{_mask_loaded_from}，有效像素: {np.count_nonzero(mask)}")

    # 尝试从 input / output 目录加载默认灰度掩码
    if mask is None:
        for p in [
            resolve_image_path(output_dir, 'grayMask_40x.png'),
            resolve_image_path(input_dir, 'grayMask_40x.png'),
            resolve_image_path(input_dir, 'mask.png'),
        ]:
            # 大图可能超出 OpenCV 像素限制，fallback 到 skimage
            tmp = None
            try:
                tmp = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            except Exception:
                tmp = None
            if tmp is None:
                try:
                    from skimage import io as _io
                    tmp = _io.imread(p)
                    if tmp.ndim == 3:
                        tmp = tmp[:, :, 0]
                    tmp = (tmp > 0).astype(np.uint8) * 255
                except Exception:
                    tmp = None
            if tmp is None:
                try:
                    from PIL import Image as _PIL
                    import PIL
                    _saved = PIL.Image.MAX_IMAGE_PIXELS
                    PIL.Image.MAX_IMAGE_PIXELS = None
                    _pil_img = _PIL.open(p)
                    tmp = np.array(_pil_img.convert("L"))
                    PIL.Image.MAX_IMAGE_PIXELS = _saved
                    tmp = (tmp > 0).astype(np.uint8) * 255
                except Exception:
                    tmp = None
            if tmp is not None and tmp.shape[:2] == (h, w):
                mask = tmp
                _mask_loaded_from = p
                break

    if mask is not None:
        print(f"  有效组织掩码已加载 ({_mask_loaded_from})，非零像素: {np.count_nonzero(mask)}")
    else:
        print(f"  [信息] 未加载掩码，将使用全图深度填色")

    if compact_rate is None:
        compact_rate = 0.1 if compact else COMPACT_RATE
    compact_rate = float(compact_rate)
    if compact_rate <= 0 or compact_rate > 1:
        raise ValueError(f"compact_rate must be in (0, 1], got {compact_rate}")

    # -- Compact 模式：按 compact_rate 对最终可视化降采样 --
    if compact_rate < 1:
        scale = compact_rate
        small_w = max(int(round(w * scale)), 1)
        small_h = max(int(round(h * scale)), 1)
        orig_img = cv2.resize(orig_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        h, w = small_h, small_w

        # 边界坐标同步缩放，并保持在缩略图画布内
        wm_pts = np.rint(wm_pts * scale).astype(np.int32)
        gm_pts = np.rint(gm_pts * scale).astype(np.int32)
        wm_pts[:, 0] = np.clip(wm_pts[:, 0], 0, w - 1)
        wm_pts[:, 1] = np.clip(wm_pts[:, 1], 0, h - 1)
        gm_pts[:, 0] = np.clip(gm_pts[:, 0], 0, w - 1)
        gm_pts[:, 1] = np.clip(gm_pts[:, 1], 0, h - 1)

        # 掩码同步缩放
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        print(f"  Compact模式: rate={compact_rate}, 降采样至 ({h}, {w})")

    # -- 计算全图深度场（每个像素独立计算，与列插值无关） --
    print("正在计算全图深度场（距离变换）...")
    depth_map = _compute_depth_map(h, w, gm_pts, wm_pts)
    print(f"  深度范围: {depth_map.min():.3f} ~ {depth_map.max():.3f}")

    # -- 定义层颜色（BGR） --
    layer_colors = [
        (255, 100, 100),   # Layer 1 - 浅红
        (100, 255, 100),   # Layer 2 - 浅绿
        (100, 100, 255),   # Layer 3 - 浅蓝
        (255, 255, 100),   # Layer 4 - 黄色
        (255, 100, 255),   # Layer 5 - 粉色
        (100, 255, 255),   # Layer 6 - 青色
        (255, 180, 100),   # Layer 7 - 橙色
        (180, 100, 255),   # Layer 8 - 紫色
    ]

    # -- 直接根据深度值给每个像素赋予对应层级的颜色 --
    layer_color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, layer in enumerate(layers):
        layer_start = layer['start']
        layer_end = layer['end']
        color = layer_colors[i % len(layer_colors)]

        # 深度在 [start, end) 范围内的像素赋予对应颜色
        region = (depth_map >= layer_start) & (depth_map < layer_end)
        layer_color_mask[region] = color

    # mask 裁剪：不在掩码内的像素强制置黑（确保 mask 始终生效）
    if mask is not None:
        layer_color_mask[mask == 0] = 0
        filled_px = np.count_nonzero(np.any(layer_color_mask > 0, axis=2))
        print(f"  Mask裁剪后非零像素: {filled_px} / {h * w}")

    # -- 生成 Overlay：mask 叠加到原图 --
    if len(orig_img.shape) == 2:
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    elif orig_img.shape[2] == 4:
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
    else:
        orig_bgr = orig_img.copy()

    alpha = 0.4
    overlay_img = cv2.addWeighted(layer_color_mask, alpha, orig_bgr, 1 - alpha, 0)
    # overlay 仅在 mask 内显示彩色叠加层，mask 外保留原图
    if mask is not None:
        overlay_img[mask == 0] = orig_bgr[mask == 0]

    # -- 生成分层线图像（粗白色虚线）--
    print("正在生成分层线图像...")
    # 先在空白画布上绘制虚线，再合成到原图上（保持 mask 外原图完整）
    line_canvas = np.zeros_like(orig_bgr)
    line_thickness = 10
    line_valid_mask = np.ones((h, w), dtype=bool)
    margin = max(line_thickness + 2, 1)
    line_valid_mask[:margin, :] = False
    line_valid_mask[-margin:, :] = False
    line_valid_mask[:, :margin] = False
    line_valid_mask[:, -margin:] = False
    if mask is not None:
        mask_bin = (mask > 0).astype(np.uint8)
        ksize = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        line_valid_mask &= cv2.erode(mask_bin, kernel, iterations=1) > 0

    boundaries = set()
    for i, layer in enumerate(layers):
        end = layer['end']
        if i < len(layers) - 1:
            boundaries.add(round(end, 6))

    for boundary in sorted(boundaries):
        binary = (depth_map >= boundary).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.arcLength(cnt, True) < 50:
                continue
            _draw_dashed_contour(
                line_canvas, cnt, (255, 255, 255), thickness=line_thickness,
                valid_mask=line_valid_mask
            )
    # 直接按 CSV 中的点序列绘制 GM/WM 边界，避免点图 + findContours 把开放边界闭合。
    _draw_dashed_points(line_canvas, gm_pts, (0, 255, 0), thickness=10)
    _draw_dashed_points(line_canvas, wm_pts, (0, 0, 255), thickness=10)

    # 将虚线合成到原图，仅保留 mask 内的虚线
    layer_lines_img = orig_bgr.copy()
    if mask is not None:
        line_mask = (line_canvas > 0) & (np.tile(mask > 0, (3, 1, 1)).transpose(1, 2, 0))
    else:
        line_mask = line_canvas > 0
    layer_lines_img[line_mask] = line_canvas[line_mask]
    print(f"  绘制了 {len(boundaries)} 条分层线")

    # -- 保存单张图像 --
    if issave:
        cv2.imwrite(f'{output_dir}\\layers_color_mask.png', layer_color_mask)
        print(f"分层颜色Mask已保存为 {output_dir}\\layers_color_mask.png")
        cv2.imwrite(f'{output_dir}\\layers_overlay.png', overlay_img)
        print(f"Overlay图已保存为 {output_dir}\\layers_overlay.png")
        cv2.imwrite(f'{output_dir}\\layer_lines.png', layer_lines_img)
        print(f"分层线图已保存为 {output_dir}\\layer_lines.png")

    # -- 4图综合展示（2x2） --
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(layer_lines_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Layer Lines (Dashed)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(layer_color_mask, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Layer Color Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay (Mask on Original)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    if issave:
        combined_path = os.path.join(output_dir, 'combined_visualization.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"综合可视化图已保存为 {combined_path}")
        print(f"\n  输出文件:")
        print(f"    {output_dir}\\layers_color_mask.png")
        print(f"    {output_dir}\\layers_overlay.png")
        print(f"    {output_dir}\\layer_lines.png")
        print(f"    {output_dir}\\combined_visualization.png")
    plt.show(block=True)


if __name__ == "__main__":
    # 测试入口
    assign_layers_to_mask(
        f"{input_dir}\\WM_40x.csv", f"{input_dir}\\GM_40x.csv",
        f"{output_dir}\\segmented_layers.csv",
        resolve_image_path(input_dir, "40x.png"),
        issave=True,
    )
