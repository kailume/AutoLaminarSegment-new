#!/usr/bin/env python
"""
Interactive boundary drawing tool.

Display an image and let the user draw one or more polygonal boundaries
by clicking.  All boundary vertex coordinates are saved to a CSV file.

Controls
--------
Left-click          Add vertex to current boundary
Right-click / Enter Finalize current boundary (≥ 2 points required)
U / Backspace       Remove last vertex
C                   Cancel (discard) current boundary
D                   Delete last completed boundary
S                   Save boundaries to CSV (without quitting)
Q / Esc             Save and quit
Scroll / toolbar    Zoom & pan as normal (clicks ignored while toolbar is active)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ── colour cycle for completed boundaries ─────────────────────────────────────
BOUNDARY_COLORS = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFCC00",
    "#FF44CC", "#00FFEE", "#FF8800", "#AAAAFF",
]
COLOR_NAMES = {
    "#FF4444": "red", "#44AAFF": "blue", "#44FF88": "green", "#FFCC00": "yellow",
    "#FF44CC": "pink", "#00FFEE": "cyan", "#FF8800": "orange", "#AAAAFF": "purple",
}


# ── LUT definitions (stops, RGB_colors) — press L in UI to cycle ─────────────
LUT_DEFS = {
    "Fire":    ([0, 85, 170, 255], [[0,0,0], [255,0,0], [255,255,0], [255,255,255]]),
    "Gray":    ([0, 255],           [[0,0,0], [255,255,255]]),
    "Ice":     ([0, 100, 200, 255], [[0,0,0], [0,140,255], [180,220,255], [255,255,255]]),
    "Green":   ([0, 100, 200, 255], [[0,0,0], [0,120,0], [100,255,100], [200,255,200]]),
    "Hot":     ([0, 100, 200, 255], [[0,0,0], [200,50,0], [255,200,0], [255,255,255]]),
    "Red":     ([0, 100, 200, 255], [[0,0,0], [120,0,0], [255,100,100], [255,200,200]]),
}

# ── image loading ──────────────────────────────────────────────────────────────

def load_display_image(image_path: Path) -> np.ndarray:
    """Load → uint8 grayscale + 50% enhancement. Returns (H, W)."""
    try:
        import tifffile
        img = tifffile.imread(str(image_path))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    # Normalise to uint8
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    # Convert to grayscale
    if img.ndim == 3:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 50% brightness & contrast enhancement
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=25)

    return img


# ── drawing state machine ──────────────────────────────────────────────────────

class BoundaryDrawer:
    """Manages interactive polygon drawing on a matplotlib axes."""

    def __init__(self, fig: plt.Figure, ax: plt.Axes, save_path: Path,
                 save_mask: bool = False, image_shape: tuple | None = None,
                 gray_img: np.ndarray | None = None):
        self.fig = fig
        self.ax  = ax
        self.save_path = save_path
        self._save_mask_flag = save_mask
        self._image_shape = image_shape

        self.boundaries: list[np.ndarray] = []   # completed, shape (N, 2) each
        self._boundary_colors: list[str] = []     # hex color per boundary
        self.current_pts: list[tuple] = []        # in-progress vertices

        # matplotlib artists
        self._cur_line    = None   # dashed line for current polygon
        self._cur_dots    = None   # vertex markers for current polygon
        self._preview_seg = None   # rubber-band segment to cursor
        self._completed_artists: list[tuple] = []  # (line, dots) per boundary

        self._color_idx = 0
        self._mask_mode = False
        self._border_artists: list = []

        # LUT state
        self._gray_img = gray_img
        self._img_handle = None
        self._lut_idx = 0
        self._lut_names = list(LUT_DEFS.keys())
        if gray_img is not None:
            self._apply_lut()

        self._bg = None
        fig.canvas.mpl_connect("draw_event", self._on_draw)

        cids = [
            fig.canvas.mpl_connect("button_press_event",  self._on_click),
            fig.canvas.mpl_connect("key_press_event",     self._on_key),
            fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
        ]
        self._cids = cids
        self._refresh_status()

    # ── LUT switching ──────────────────────────────────────────────────────────

    def _apply_lut(self):
        """Re-compute RGB from current LUT and update the image."""
        name = self._lut_names[self._lut_idx]
        stops, colors = LUT_DEFS[name]
        lut = np.zeros((256, 3), dtype=np.uint8)
        colors = np.array(colors, dtype=np.uint8)
        for c in range(3):
            lut[:, c] = np.interp(np.arange(256), stops, colors[:, c]).astype(np.uint8)

        rgb = lut[self._gray_img]   # (H, W) → (H, W, 3)

        if self._img_handle is None:
            self._img_handle = self.ax.imshow(rgb, aspect="equal", interpolation="nearest")
        else:
            self._img_handle.set_data(rgb)

        # Invalidate cached background so _blit forces a full redraw instead of
        # restoring the old LUT.  The draw_idle from _blit → _refresh_status will
        # show the new LUT and re-cache _bg via _on_draw.
        self._bg = None
        self.fig.canvas.draw_idle()

    # ── event handlers ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        if self._toolbar_active():
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return

        if event.button == 1:                     # left – add vertex
            self.current_pts.append((event.xdata, event.ydata))
            self._redraw_current()
            self._refresh_status()

        elif event.button == 3:                   # right – close
            self._close_boundary()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k in ("enter", " "):
            self._close_boundary()
        elif k in ("u", "backspace"):
            if self.current_pts:
                self.current_pts.pop()
                self._redraw_current()
                self._refresh_status()
        elif k == "c":
            self._cancel_current()
        elif k == "d":
            self._delete_last()
        elif k == "l":
            self._lut_idx = (self._lut_idx + 1) % len(self._lut_names)
            self._apply_lut()
            self._set_message(f"LUT: {self._lut_names[self._lut_idx]}")
        elif k == "m":
            self._toggle_mask_mode()
        elif k == "s":
            self._save()
        elif k in ("q", "escape"):
            self._save()
            plt.close(self.fig)

    def _on_motion(self, event):
        pass

    # ── boundary operations ────────────────────────────────────────────────────

    def _close_boundary(self):
        if len(self.current_pts) < 2:
            self._set_message("Need at least 2 points to finalize a boundary.", warn=True)
            return

        pts   = np.array(self.current_pts, dtype=np.float64)
        color = BOUNDARY_COLORS[self._color_idx % len(BOUNDARY_COLORS)]
        self._color_idx += 1
        self.boundaries.append(pts)
        self._boundary_colors.append(color)

        # Draw as open polyline (no auto-close back to first point)
        (line,) = self.ax.plot(pts[:, 0], pts[:, 1], "-", color=color,
                               linewidth=2, zorder=9)
        dots     = self.ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color,
                                   zorder=10, linewidths=0)
        self._completed_artists.append((line, dots))

        # Clear in-progress state
        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._remove_artist("_preview_seg")

        self._blit()
        self._refresh_status()
        print(f"Boundary {len(self.boundaries)} finalized  ({len(pts)} vertices)")
        if self._mask_mode:
            self._draw_mask_preview()

    def _cancel_current(self):
        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._remove_artist("_preview_seg")
        self._blit()
        self._refresh_status()

    def _delete_last(self):
        if not self.boundaries:
            return
        self.boundaries.pop()
        self._boundary_colors.pop()
        line, dots = self._completed_artists.pop()
        line.remove()
        dots.remove()
        self._blit()
        self._refresh_status()
        print(f"Last boundary deleted.  {len(self.boundaries)} remaining.")
        if self._mask_mode:
            if self.boundaries:
                self._draw_mask_preview()
            else:
                self._remove_borders()

    # ── mask mode ─────────────────────────────────────────────────────────────

    def _toggle_mask_mode(self):
        """Toggle mask mode: show/hide the actual mask contour."""
        self._mask_mode = not self._mask_mode
        if self._mask_mode:
            self._draw_mask_preview()
            self._set_message("Mask mode ON — showing mask contour")
        else:
            self._remove_borders()
            self._set_message("Mask mode OFF")
        self._refresh_status()

    def _draw_mask_preview(self):
        """Compute the mask contour from all boundaries and draw it dashed."""
        self._remove_borders()
        if not self._image_shape or not self.boundaries:
            return
        H, W = self._image_shape

        # Rasterize boundaries
        canvas = np.zeros((H, W), dtype=np.uint8)
        for pts in self.boundaries:
            interp = self._interpolate(pts)
            for i in range(len(interp) - 1):
                cv2.line(canvas, tuple(interp[i]), tuple(interp[i+1]), 255, 2)

        # Morph close to bridge gaps, find outer contour
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k, iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        biggest = max(contours, key=cv2.contourArea)
        pts = biggest.reshape(-1, 2)

        style = dict(linestyle="--", color="white", linewidth=3.0, alpha=0.9, zorder=20)
        (line,) = self.ax.plot(pts[:, 0], pts[:, 1], **style)
        self._border_artists = [line]
        self._blit()
        print(f"  mask preview: contour ({len(pts)} pts)")

    def _remove_borders(self):
        for a in self._border_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._border_artists = []
        self._blit()


    def _save_mask(self):
        """Save mask as binary image at original resolution.

        先栅格化所有边界线，再形态学闭合弥合端点间隙，
        最后寻找最外层轮廓填充，得到多条边界的闭合包络区域。
        """
        if not self._image_shape:
            print("  [mask] No image shape — cannot save mask")
            return
        H, W = self._image_shape

        # 将所有边界绘制为白色线条
        canvas = np.zeros((H, W), dtype=np.uint8)
        for pts in self.boundaries:
            interp = self._interpolate(pts)
            for i in range(len(interp) - 1):
                cv2.line(canvas, tuple(interp[i]), tuple(interp[i+1]), 255, 2)

        # 形态学闭运算弥合端点间隙
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k, iterations=3)

        # 寻找最外层轮廓并填充
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((H, W), dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        base = self.save_path.parent / self.save_path.stem
        mask_path = base.parent / f"{base.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"  mask ({W}x{H}) → {mask_path}")

    # ── blit ────────────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        """Cache background after any full draw (initial show, resize, zoom)."""
        try:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            pass

    def _blit(self):
        """Redraw only overlay artists — image background stays static."""
        if self._bg is None:
            self.fig.canvas.draw_idle()
            return
        fig, ax = self.fig, self.ax
        fig.canvas.restore_region(self._bg)

        for line, dots in self._completed_artists:
            ax.draw_artist(line)
            ax.draw_artist(dots)
        if self._cur_line is not None:
            ax.draw_artist(self._cur_line)
        if self._cur_dots is not None:
            ax.draw_artist(self._cur_dots)
        if self._preview_seg is not None:
            ax.draw_artist(self._preview_seg)
        for a in self._border_artists:
            ax.draw_artist(a)

        ax.draw_artist(ax.title)
        fig.canvas.blit(fig.bbox)

    # ── rendering helpers ──────────────────────────────────────────────────────

    def _redraw_current(self):
        """Re-render the in-progress polygon (caller must trigger redraw)."""
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")

        if not self.current_pts:
            return

        xs = [p[0] for p in self.current_pts]
        ys = [p[1] for p in self.current_pts]

        (self._cur_line,) = self.ax.plot(
            xs, ys, "--", color="yellow", linewidth=1.5, zorder=10
        )
        self._cur_dots = self.ax.scatter(
            xs, ys, s=25, c="yellow", zorder=11, linewidths=0
        )

    def _remove_artist(self, attr: str):
        artist = getattr(self, attr, None)
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
            setattr(self, attr, None)

    def _toolbar_active(self) -> bool:
        """Return True if zoom/pan toolbar mode is active."""
        try:
            return self.fig.canvas.toolbar.mode != ""
        except Exception:
            return False

    # ── status / title ─────────────────────────────────────────────────────────

    def _refresh_status(self, message: str = ""):
        n_done  = len(self.boundaries)
        n_cur   = len(self.current_pts)
        mode_tag = " [MASK]" if self._mask_mode else ""
        lut_tag = f"  LUT:{self._lut_names[self._lut_idx]}"
        title   = (
            f"Completed: {n_done}  |  Current vertices: {n_cur}{mode_tag}{lut_tag}  "
            f"{'|  ' + message if message else ''}\n"
            "LClick=vertex  RClick/Enter=finalize  U=undo  C=cancel  "
            "D=del last  L=swap LUT  M=mask  S=save  Q=quit"
        )
        self.ax.set_title(title, fontsize=8.5, color="white",
                          loc="left", pad=6)
        self._blit()

    def _set_message(self, msg: str, warn: bool = False):
        print(("WARNING: " if warn else "") + msg)
        self._refresh_status(msg)

    # ── save ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpolate(pts: np.ndarray) -> np.ndarray:
        """
        Densely sample a polyline at ~1-pixel intervals.
        Returns an (N, 2) array of integer pixel coordinates.
        """
        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dist = np.hypot(x1 - x0, y1 - y0)
            n = max(int(np.ceil(dist)), 2)
            # endpoint=False avoids duplicating shared vertices between segments
            xs = np.linspace(x0, x1, n, endpoint=False)
            ys = np.linspace(y0, y1, n, endpoint=False)
            segments.append(np.column_stack([xs, ys]))
        # Append the final endpoint
        segments.append(pts[-1:])
        all_pts = np.vstack(segments)
        return np.round(all_pts).astype(int)

    def _save(self):
        if not self.boundaries:
            print("No boundaries to save.")
            return

        # 保存合并文件
        all_pts = np.vstack([self._interpolate(pts) for pts in self.boundaries])
        df = pd.DataFrame(all_pts, columns=["x", "y"])
        df.to_csv(self.save_path, index=False)
        print(f"Saved merged ({len(self.boundaries)} boundaries, "
              f"{len(df)} points) → {self.save_path}")

        # 每个边界单独保存，文件名带颜色名
        base = self.save_path.parent / self.save_path.stem
        for i, pts in enumerate(self.boundaries):
            hex_color = self._boundary_colors[i] if i < len(self._boundary_colors) else "#FFFFFF"
            color_name = COLOR_NAMES.get(hex_color, f"color{i+1}")
            interp = self._interpolate(pts)
            csv_path = base.parent / f"boundary_{i+1}_{color_name}.csv"
            pd.DataFrame(interp, columns=["x", "y"]).to_csv(csv_path, index=False)
            print(f"  boundary {i+1} ({len(interp)} pts, {color_name}) → {csv_path.name}")

        # mask 模式或 --mask 标志 → 保存二值 mask
        if self._mask_mode or self._save_mask_flag:
            self._save_mask()

        self._set_message(f"Saved {len(self.boundaries)} boundaries to {self.save_path.name}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Draw polygonal boundaries on an image and save coordinates."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: {image_stem}_boundaries.csv beside the image)"
    )
    parser.add_argument(
        "--mask", action="store_true",
        help="Also generate a binary mask image from the enclosed regions"
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    save_path  = (
        Path(args.output) if args.output
        else image_path.parent / f"{image_path.stem}_boundaries.csv"
    )

    print(f"Loading: {image_path}")
    image = load_display_image(image_path)
    H, W  = image.shape[:2]
    print(f"  {W}w × {H}h px\n")

    # ── 打印操作说明 ──────────────────────────────────────────
    print("=" * 58)
    print("  操作方式:")
    print("    [左键点击]     添加顶点")
    print("    [右键/Enter]   完成当前边界")
    print("    [U/Backspace]  撤销上一个顶点")
    print("    [C]            取消当前边界")
    print("    [D]            删除上一条已完成的边界")
    print("    [L]            切换 LUT 配色（Fire/Gray/Ice/Green/Hot/Red）")
    print("    [M]            切换 Mask 模式（显示边框参考线）")
    print("    [S]            保存边界 + Mask (mask模式下)")
    print("    [Q/Esc]        保存并退出")
    print("=" * 58)
    print()
    print(f"输出路径: {save_path}")
    if args.mask:
        print(f"  --mask 已启用, S保存时自动生成 mask")
    print(f"  提示: 按 M 键可在界面内切换 mask 模式")
    print()

    # Figure sized proportionally (max 14 inches on longest side)
    max_in  = 14.0
    ratio   = W / H
    if ratio >= 1:
        fig_w, fig_h = max_in, max_in / ratio
    else:
        fig_w, fig_h = max_in * ratio, max_in

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # image y-axis: 0 at top
    ax.axis("off")

    plt.tight_layout(pad=0.3)

    drawer = BoundaryDrawer(fig, ax, save_path,
                            save_mask=args.mask, image_shape=image.shape[:2],
                            gray_img=image)
    plt.show()


if __name__ == "__main__":
    main()
