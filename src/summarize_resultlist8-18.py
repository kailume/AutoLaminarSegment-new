#!/usr/bin/env python
"""
为 resultlist 中 case 8-18 逐例生成分层结果可视化图。

每张图 1×4 左右排列（同 resultlist_summary 风格）：
  Col 0: 4x_dapi overview + 红色定位框
  Col 1: GT 放大裁剪 + 彩色分层虚线
  Col 2: 4x.png overview + 红色定位框
  Col 3: 算法放大 + 彩色分层虚线
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# 颜色映射 (RGB)
# 参考 resultanalysis.py：GT_LAYER_COLORS / ALGO_LAYER_COLORS（BGR）
# ---------------------------------------------------------------------------

# GT mask 中的实际颜色
GT_MASK_LAYER_RGB = {
    "L1":   (100, 100, 255),
    "L2/3": (100, 255, 100),
    "L4":   (255, 100, 100),
    "L5/6": (255, 100, 255),
}

# Algo mask 中的实际颜色
ALGO_MASK_LAYER_RGB = {
    "L1":   (100, 100, 255),
    "L2/3": (100, 255, 100),
    "L4":   (255, 100, 100),
    "L5/6": (100, 255, 255),   # BGR(255,255,100)
}

# 分层虚线绘制颜色 — 统一白色
GT_BOUNDARY_COLORS_RGB = {
    "L1":   (255, 255, 255),
    "L2/3": (255, 255, 255),
    "L4":   (255, 255, 255),
    "L5/6": (255, 255, 255),
}

ALGO_BOUNDARY_COLORS_RGB = {
    "L1":   (255, 255, 255),
    "L2/3": (255, 255, 255),
    "L4":   (255, 255, 255),
    "L5/6": (255, 255, 255),
}

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

DEFAULT_4X_PIXEL_SCALE = 1.625
DEFAULT_CAMERA_RESOLUTION = (1376, 1024)


def read_json_start(json_path: Path) -> tuple[float, float, float]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    positions = data.get("positions", [])
    if not positions:
        raise ValueError(f"No positions in {json_path}")
    scan_info = data.get("scan_info", {})
    pixel_scale = float(scan_info.get("pixel_scale", DEFAULT_4X_PIXEL_SCALE))
    camera_w, camera_h = scan_info.get("camera_resolution", DEFAULT_CAMERA_RESOLUTION)
    x0 = float(positions[0]["x"]) - float(camera_w) * pixel_scale / 2.0
    y0 = float(positions[0]["y"]) - float(camera_h) * pixel_scale / 2.0
    return x0, y0, pixel_scale


def read_objective_offset(case_dir: Path) -> dict[str, float] | None:
    path = case_dir / "ObjectiveOffsetConfig.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    obj4x = data.get("Obj4X", {})
    if "x_um" not in obj4x or "y_um" not in obj4x:
        return None
    return {"x_um": float(obj4x["x_um"]), "y_um": float(obj4x["y_um"])}


def register_rect_40x_on_4x(
    case_dir: Path,
    dapi_shape: tuple[int, int],
    objective_offset: dict[str, float] | None = None,
) -> tuple[int, int, int, int]:
    start4_x, start4_y, scale4 = read_json_start(case_dir / "4x.json")
    start40_x, start40_y, scale40 = read_json_start(case_dir / "40x.json")
    dapi_h, dapi_w = dapi_shape

    off_x = objective_offset["x_um"] if objective_offset else 0.0
    off_y = objective_offset["y_um"] if objective_offset else 0.0

    x = int(round((start40_x - start4_x - off_x) / scale4))
    y = int(round((start40_y - start4_y - off_y) / scale4))
    ratio = scale40 / scale4
    w = max(1, int(round(dapi_w * ratio)))
    h = max(1, int(round(dapi_h * ratio)))
    return x, y, w, h


def load_image_rgb(path: Path) -> Image.Image:
    """加载图像 → 8-bit RGB，处理 16-bit 等特殊情况。"""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.array(img, dtype=np.float64)
        low, high = np.percentile(arr, [1, 99.5])
        if high > low:
            arr = np.clip((arr - low) / (high - low) * 255, 0, 255)
        else:
            arr = arr / 256.0
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def load_fit_image_rgb(path: Path, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    """加载并缩放到目标尺寸，返回 (image, scale)。"""
    img = Image.open(path)
    orig_w, orig_h = img.width, img.height
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.array(img, dtype=np.float64)
        low, high = np.percentile(arr, [1, 99.5])
        if high > low:
            arr = np.clip((arr - low) / (high - low) * 255, 0, 255)
        else:
            arr = arr / 256.0
        img = Image.fromarray(arr.astype(np.uint8))
        orig_w, orig_h = img.width, img.height
    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    scale = img.width / orig_w if orig_w > 0 else 1.0
    return img.convert("RGB").copy(), scale


def apply_dapi_lut(img: Image.Image) -> Image.Image:
    """给灰度 DAPI 图像加上蓝-青色荧光 LUT（模拟显微镜 DAPI 通道）。"""
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 2] = np.clip(gray, 0, 255).astype(np.uint8)           # Blue: 100%
    rgb[:, :, 1] = np.clip(gray * 0.35, 0, 255).astype(np.uint8)    # Green: 35%
    rgb[:, :, 0] = np.clip(gray * 0.05, 0, 255).astype(np.uint8)    # Red:   5%
    return Image.fromarray(rgb)


_NICE_BAR_LENGTHS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def add_scale_bar(
    img: Image.Image,
    um_per_px: float,
    bar_um: float | None = None,
    bg_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """在图像右下角绘制 scale bar，可指定 bar 长度(µm)和背景色。"""
    if um_per_px <= 0:
        return img

    # 自动选长度：目标占图像宽度约 12%
    if bar_um is None:
        target_um = img.width * 0.12 * um_per_px
        bar_um = min(_NICE_BAR_LENGTHS, key=lambda n: abs(target_um - n))
    bar_px = max(10, min(int(round(bar_um / um_per_px)), img.width - 30))

    margin = 14
    bar_h = max(2, int(img.height * 0.006))
    x2 = img.width - margin
    x1 = x2 - bar_px
    y0 = img.height - margin

    draw = ImageDraw.Draw(img)
    label = f"{bar_um} um"
    bbox = draw.textbbox((0, 0), label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x1 + (bar_px - tw) / 2
    ty = y0 - bar_h - th - 3

    # 可选黑色背景
    if bg_color is not None:
        pad_x = 5
        pad_y = 3
        draw.rectangle(
            [x1 - pad_x, ty - pad_y, x2 + pad_x, y0 + pad_y],
            fill=bg_color,
        )

    draw.rectangle([x1, y0 - bar_h, x2, y0], fill=(255, 255, 255))
    draw.text((tx, ty), label, fill=(255, 255, 255))

    return img


def fit_to_box_with_scale(img: Image.Image, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    scale = min(max_width / img.width, max_height / img.height, 1.0)
    if scale >= 1.0:
        return img, 1.0
    size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    return img.resize(size, Image.Resampling.LANCZOS), scale


def fit_to_box(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    return fit_to_box_with_scale(img, max_width, max_height)[0]


def crop_with_padding(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox
    out = Image.new("RGB", (w, h), (0, 0, 0))
    src_box = (
        max(0, x), max(0, y),
        min(img.width, x + w), min(img.height, y + h),
    )
    if src_box[2] <= src_box[0] or src_box[3] <= src_box[1]:
        return out
    crop = img.crop(src_box)
    paste_x = max(0, -x)
    paste_y = max(0, -y)
    out.paste(crop, (paste_x, paste_y))
    return out


def scale_visible_bbox(
    bbox: tuple[int, int, int, int], scale: float, image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    img_w, img_h = image_size
    x1 = max(0.0, x * scale)
    y1 = max(0.0, y * scale)
    x2 = min(float(img_w - 1), (x + w) * scale)
    y2 = min(float(img_h - 1), (y + h) * scale)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_scaled_registered_box(
    img: Image.Image,
    rect: tuple[float, float, float, float] | None,
    color=(255, 0, 0),
    width: int = 3,
) -> Image.Image:
    """在 overview 图上绘制定位框（红色实线）。"""
    if rect is None:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(rect, outline=color, width=width)
    return out


# ---------------------------------------------------------------------------
# 虚线绘制
# ---------------------------------------------------------------------------

def _draw_dashed_polyline_cv(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 9,
    gap_len: int = 5,
) -> None:
    pts = points.reshape(-1, 2)
    start = 0
    draw_flag = True
    while start < len(pts) - 1:
        end = min(start + (dash_len if draw_flag else gap_len), len(pts))
        if draw_flag and end - start >= 2:
            seg = pts[start:end].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(image, [seg], False, color, thickness, lineType=cv2.LINE_AA)
        start = end
        draw_flag = not draw_flag


# ---------------------------------------------------------------------------
# 彩色分层边界叠加
# ---------------------------------------------------------------------------

COLOR_TOLERANCE = 15


def _process_and_draw_edge(edge: np.ndarray, base_np: np.ndarray,
                           bdr_color: tuple[int, int, int], thickness: int):
    """从布尔边缘图中提取边界点，按列分段 + 高斯平滑后画虚线。"""
    rows, cols = np.where(edge)
    if len(rows) < 5:
        return
    order = np.argsort(cols)
    xs = cols[order].astype(np.float32)
    ys = rows[order].astype(np.float32) + 0.5

    gaps = np.diff(xs) > 2
    seg_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
    seg_ends = np.concatenate([np.where(gaps)[0], [len(xs) - 1]])

    for s, e in zip(seg_starts, seg_ends):
        if e - s < 5:
            continue
        pts = np.column_stack([xs[s:e + 1], ys[s:e + 1]])
        n_pts = len(pts)
        if n_pts >= 6:
            r = int(round(2.5 * 0.8))
            window = 2 * r + 1
            kernel = cv2.getGaussianKernel(window, 0.8).ravel()
            y_pad = np.pad(pts[:, 1], (r, r), mode="edge")
            pts[:, 1] = np.convolve(y_pad, kernel[::-1], mode="valid")
        pts_int = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        _draw_dashed_polyline_cv(base_np, pts_int, bdr_color, thickness=thickness)


def _draw_horizontal_edge(base_np: np.ndarray, cols: np.ndarray, y: float,
                          bdr_color: tuple[int, int, int], thickness: int):
    """画出位于 y 处的水平层边界（层延伸到图像边缘时使用）。"""
    cols = cols.astype(np.float32)
    gaps = np.diff(cols) > 2
    seg_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
    seg_ends = np.concatenate([np.where(gaps)[0], [len(cols) - 1]])
    for s, e in zip(seg_starts, seg_ends):
        if e - s < 3:
            continue
        pts = np.column_stack([cols[s:e + 1], np.full(e - s + 1, y, dtype=np.float32)])
        pts_int = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        _draw_dashed_polyline_cv(base_np, pts_int, bdr_color, thickness=thickness)


def overlay_colored_layer_boundaries(
    base: Image.Image,
    mask_path: Path,
    mask_colors: dict[str, tuple[int, int, int]],
    boundary_colors: dict[str, tuple[int, int, int]],
    thickness: int = 2,
) -> Image.Image:
    """
    只画水平层间分界线（去除左右竖边）。

    每层的"顶边界"（当前层之下）和"底边界"（当前层之上）独立处理，
    避免顶/底边界点混合后高斯平滑拉偏 y 坐标。
    """
    with Image.open(mask_path) as raw:
        mask_img = raw.resize(base.size, Image.Resampling.NEAREST).convert("RGB")
    base_np = np.asarray(base).copy()
    mask_np = np.asarray(mask_img)
    h, w = mask_np.shape[:2]

    # 1. 标签图（层编号从 1 开始，0=背景）
    layer_names = list(mask_colors.keys())
    label_map = np.zeros((h, w), dtype=np.uint8)
    for idx, layer_name in enumerate(layer_names):
        target = np.array(mask_colors[layer_name], dtype=np.int16)
        diff = np.abs(mask_np.astype(np.int16) - target)
        label_map[np.all(diff <= COLOR_TOLERANCE, axis=2)] = idx + 1

    # 2. 错位比较 (y, y+1)
    upper = label_map[:-1, :]  # 上一行
    lower = label_map[1:, :]   # 下一行

    # 3. 逐层处理：顶边界和底边界分开绘制
    for idx in range(1, len(layer_names) + 1):
        bdr_color = boundary_colors.get(layer_names[idx - 1], (255, 255, 255))

        # 底边界：当前层在上 → 其他层/背景在下
        edge_bot = ((upper == idx) & (lower != idx))
        _process_and_draw_edge(edge_bot, base_np, bdr_color, thickness)

        # 顶边界：背景在上 → 当前层在下
        edge_top = (upper == 0) & (lower == idx)
        _process_and_draw_edge(edge_top, base_np, bdr_color, thickness)

    # 4. 补画层延伸到图像边缘时的边界（错位比较无法检测图像第 0 行和第 h-1 行）
    for idx in range(1, len(layer_names) + 1):
        bdr_color = boundary_colors.get(layer_names[idx - 1], (255, 255, 255))

        # 图像顶部
        if np.any(label_map[0, :] == idx):
            _draw_horizontal_edge(base_np, np.where(label_map[0, :] == idx)[0],
                                  0.0, bdr_color, thickness)

        # 图像底部
        if np.any(label_map[h - 1, :] == idx):
            _draw_horizontal_edge(base_np, np.where(label_map[h - 1, :] == idx)[0],
                                  float(h - 1), bdr_color, thickness)

    # 5. 在每层左侧标注层级名称
    base_pil = Image.fromarray(base_np)
    draw_text = ImageDraw.Draw(base_pil)
    try:
        font = ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    left_band = min(80, w)  # 仅看左侧一列定位

    for idx in range(1, len(layer_names) + 1):
        # 只在左侧区域内找层的 y 范围
        left_mask = label_map[:, :left_band] == idx
        ys = np.where(left_mask)[0]
        if len(ys) < 5:
            continue
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        if y_max - y_min < 24:
            continue
        label = layer_names[idx - 1]
        bbox = draw_text.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        tx = 6
        ty = int(np.clip((y_min + y_max - th) / 2, y_min + 4, y_max - th - 4))
        draw_text.text((tx, ty), label, fill=(255, 255, 255), font=font)

    return base_pil


# ---------------------------------------------------------------------------
# GT mask 查找
# ---------------------------------------------------------------------------

def find_gt_mask(case_dir: Path) -> Path | None:
    for name in ["gt_40x_crop_from_gt.png", "GT.png", "4x_GT.png"]:
        p = case_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# 逐例生成 figure
# ---------------------------------------------------------------------------

def make_case_figure(
    case_dir: Path,
    panel_width: int = 620,
    panel_height: int = 660,
    dpi: int = 150,
) -> plt.Figure:
    dapi_img = Image.open(case_dir / "dapi.png")
    dapi_shape = (dapi_img.height, dapi_img.width)
    dapi_img.close()

    image_4x = load_image_rgb(case_dir / "4x.png")
    image_4x_dapi = load_image_rgb(case_dir / "4x_dapi.png")

    gt_mask_path = find_gt_mask(case_dir)
    if gt_mask_path is None:
        raise FileNotFoundError(f"No GT mask found in {case_dir}")

    objective_offset = read_objective_offset(case_dir)

    dapi_x, dapi_y, dapi_w, dapi_h = register_rect_40x_on_4x(
        case_dir, dapi_shape, objective_offset=objective_offset,
    )
    dapi_bbox = (dapi_x, dapi_y, dapi_w, dapi_h)

    # 读取像素比例 (µm/pixel)
    _, _, pixel_scale_4x = read_json_start(case_dir / "4x.json")
    _, _, pixel_scale_40x = read_json_start(case_dir / "40x.json")

    # ---- Col 0: GT overview (4x_dapi) ----
    image_4x_dapi = apply_dapi_lut(image_4x_dapi)
    gt_overview, gt_scale = fit_to_box_with_scale(image_4x_dapi, panel_width, panel_height)
    gt_rect = scale_visible_bbox(dapi_bbox, gt_scale, gt_overview.size)
    gt_overview = draw_scaled_registered_box(gt_overview, gt_rect, color=(255, 0, 0), width=3)
    gt_overview = add_scale_bar(gt_overview, pixel_scale_4x / gt_scale, bar_um=500)

    # ---- Col 1: GT zoom (crop from 4x_dapi) ----
    gt_crop = crop_with_padding(image_4x_dapi, dapi_bbox)
    gt_crop, crop_scale = fit_to_box_with_scale(gt_crop, panel_width, panel_height)
    gt_crop = overlay_colored_layer_boundaries(
        gt_crop, gt_mask_path,
        GT_MASK_LAYER_RGB, GT_BOUNDARY_COLORS_RGB,
    )
    gt_crop = add_scale_bar(gt_crop, pixel_scale_4x / crop_scale, bar_um=300)

    # ---- Col 2: Algo overview (4x) ----
    algo_overview, algo_scale = fit_to_box_with_scale(image_4x, panel_width, panel_height)
    algo_rect = scale_visible_bbox(dapi_bbox, algo_scale, algo_overview.size)
    algo_overview = draw_scaled_registered_box(algo_overview, algo_rect, color=(255, 0, 0), width=3)
    algo_overview = add_scale_bar(algo_overview, pixel_scale_4x / algo_scale,
                                  bar_um=500, bg_color=(0, 0, 0))

    # ---- Col 3: Algo zoom (40x dapi, no LUT) ----
    algo_detail, dapi_scale = load_fit_image_rgb(case_dir / "dapi.png", panel_width, panel_height)
    algo_detail = overlay_colored_layer_boundaries(
        algo_detail, case_dir / "layers_color_mask.png",
        ALGO_MASK_LAYER_RGB, ALGO_BOUNDARY_COLORS_RGB,
    )
    algo_detail = add_scale_bar(algo_detail, pixel_scale_40x / dapi_scale, bar_um=300)

    # ---- 组装 1×4 ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=dpi)

    for ax, im in zip(axes, [gt_overview, gt_crop, algo_overview, algo_detail]):
        ax.imshow(im)
        ax.axis("off")

    # 缩放连接线（红色实线）
    def add_zoom_connectors(fig, ax_src, ax_dst, rect, detail_img):
        if rect is None:
            return
        _, y1, x2, y2 = rect
        dh = detail_img.height
        style = {"color": "#ff0000", "linewidth": 1.0, "linestyle": "-", "alpha": 0.8, "zorder": 10}
        for src, dst in [((x2, y1), (0, 0)), ((x2, y2), (0, dh - 1))]:
            fig.add_artist(ConnectionPatch(
                xyA=src, coordsA=ax_src.transData,
                xyB=dst, coordsB=ax_dst.transData,
                axesA=ax_src, axesB=ax_dst, **style,
            ))

    add_zoom_connectors(fig, axes[0], axes[1], gt_rect, gt_crop)
    add_zoom_connectors(fig, axes[2], axes[3], algo_rect, algo_detail)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.04)
    return fig


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="为 resultlist 8-18 逐例生成可视化图")
    parser.add_argument("--result-dir", default="resultlist", help="结果根目录")
    parser.add_argument("--cases", type=int, nargs="*",
                        default=list(range(8, 19)), help="处理的 case 编号")
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--panel-height", type=int, default=660)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    required = ["4x_dapi.png", "4x.png", "dapi.png", "layers_color_mask.png"]
    ok = fail = 0

    for case_id in args.cases:
        case_dir = result_dir / str(case_id)
        if not case_dir.is_dir():
            print(f"[skip] {case_dir}: 不存在")
            fail += 1
            continue
        missing = [n for n in required if not (case_dir / n).exists()]
        if missing:
            print(f"[skip] {case_dir}: 缺少 {missing}")
            fail += 1
            continue
        if find_gt_mask(case_dir) is None:
            print(f"[skip] {case_dir}: 无 GT mask")
            fail += 1
            continue

        out_path = case_dir / f"result_fig{case_id}.png"
        print(f"Generating {out_path} ...")
        try:
            fig = make_case_figure(
                case_dir, panel_width=args.panel_width,
                panel_height=args.panel_height, dpi=args.dpi,
            )
            fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
            plt.close(fig)
            print(f"  [OK] Saved")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
