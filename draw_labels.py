#!/usr/bin/env python
"""
Interactive semantic boundary and label drawing tool.

This is a semantic variant of draw_boundaries.py.  Each completed line keeps
the label selected when it was finalized.  On save the tool exports:

  - label_boundaries.csv   (x, y, boundary) matching the pipeline format
  - label_mask.png         colourful per-pixel layer mask
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEMANTICS = [
    ("Edge", "Edge", "#FF8800"),
    ("Gray", "Gray", "#44FF88"),
    ("L1", "L1", "#FFCC00"),
    ("L2/3", "L2_3", "#44AAFF"),
    ("L4", "L4", "#FF44CC"),
    ("L5/6(White)", "White", "#FFFFFF"),
]
SEMANTIC_BY_NAME = {name.lower(): (name, slug, color) for name, slug, color in SEMANTICS}
DIRECT_KEYS = {
    "0": "Edge",
    "e": "Edge",
    "g": "Gray",
    "1": "L1",
    "2": "L2/3",
    "3": "L2/3",
    "4": "L4",
    "5": "L5/6(White)",
    "6": "L5/6(White)",
    "w": "L5/6(White)",
}

# Mapping from internal boundary slug → label_boundaries.csv semantic label.
# Slugs not listed (e.g. "Edge") are excluded from the saved output.
MAPPED_LABELS: dict[str, str | None] = {
    "Gray": "pia",
    "L1": "L1_2",
    "L2_3": "L3_4",
    "L4": "L4_5",
    "White": "white",
    "Edge": None,  # skip in saved results
}

# Ordered layer slugs (top → bottom) used when building the colorful mask.
_LAYER_MASK_ORDER = ["Gray", "L1", "L2_3", "L4", "White"]

# RGB colours matching run_new_pipeline.py layer colours.
# Passed directly to OpenCV (cv2.imwrite and cv2.fillPoly store channel
# values as-is, so RGB in the numpy array → correct RGB in the PNG).
_LAYER_MASK_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "Gray": (255, 100, 100),   # L1  light red
    "L1":   (100, 255, 100),   # L2  light green
    "L2_3": (100, 100, 255),   # L3  light blue
    "L4":   (255, 255, 100),   # L4  yellow
}

@dataclass
class LabeledBoundary:
    points: np.ndarray
    label: str
    slug: str
    color: str


def load_display_image(image_path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale for interactive display."""
    try:
        import tifffile

        img = tifffile.imread(str(image_path))
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    if img.ndim == 3:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return cv2.convertScaleAbs(img, alpha=0.8, beta=0)


class LabelDrawer:
    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        save_path: Path,
        image_shape: tuple[int, int],
        gray_img: np.ndarray,
        draw_margin: int = 0,
    ):
        self.fig = fig
        self.ax = ax
        self.save_path = save_path
        self._image_shape = image_shape
        self._gray_img = gray_img
        self._draw_margin = max(int(draw_margin), 0)

        self.boundaries: list[LabeledBoundary] = []
        self.current_pts: list[tuple[float, float]] = []
        self._semantic_idx = 0

        self._cur_line = None
        self._cur_dots = None
        self._completed_artists: list[tuple] = []

        self._img_handle = None
        self._bg = None

        # Display grayscale image as RGB (stack 3 channels)
        rgb = np.stack([self._gray_img] * 3, axis=-1)
        self._img_handle = self.ax.imshow(rgb, aspect="equal", interpolation="nearest")
        fig.canvas.mpl_connect("draw_event", self._on_draw)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_status()

    @property
    def current_semantic(self) -> tuple[str, str, str]:
        return SEMANTICS[self._semantic_idx]

    def _on_click(self, event):
        if self._toolbar_active():
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            self.current_pts.append((float(event.xdata), float(event.ydata)))
            self._redraw_current()
            self._blit()
            self._refresh_status()
        elif event.button == 3:
            self._close_boundary()

    def _on_key(self, event):
        k = (event.key or "").lower()

        if k in ("enter", " "):
            self._close_boundary()
        elif k in ("u", "backspace"):
            if self.current_pts:
                self.current_pts.pop()
                self._redraw_current()
                self._blit()
                self._refresh_status()
        elif k == "c":
            self._cancel_current()
        elif k == "d":
            self._delete_last()
        elif k == "t":
            self._cycle_semantic()
        elif k in DIRECT_KEYS:
            self._set_semantic(DIRECT_KEYS[k])
        elif k == "s":
            self._save()
        elif k in ("q", "escape"):
            self._save()
            plt.close(self.fig)

    def _set_semantic(self, label: str):
        for i, (name, _slug, _color) in enumerate(SEMANTICS):
            if name == label:
                self._semantic_idx = i
                self._set_message(f"Current label: {label}")
                return

    def _cycle_semantic(self):
        self._semantic_idx = (self._semantic_idx + 1) % len(SEMANTICS)
        self._set_message(f"Current label: {self.current_semantic[0]}")

    def _close_boundary(self):
        if len(self.current_pts) < 2:
            self._set_message("Need at least 2 points to finalize a line.", warn=True)
            return

        label, slug, color = self.current_semantic
        pts = np.asarray(self.current_pts, dtype=np.float64)
        boundary = LabeledBoundary(points=pts, label=label, slug=slug, color=color)
        self.boundaries.append(boundary)

        (line,) = self.ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2, zorder=9)
        dots = self.ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color, zorder=10, linewidths=0)
        text = self.ax.text(
            pts[0, 0],
            pts[0, 1],
            f" {label}",
            color=color,
            fontsize=9,
            weight="bold",
            zorder=11,
            path_effects=[],
        )
        self._completed_artists.append((line, dots, text))

        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._blit()
        self._refresh_status()
        print(f"{label} line {len(self.boundaries)} finalized ({len(pts)} vertices)")

    def _cancel_current(self):
        self.current_pts = []
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        self._blit()
        self._refresh_status()

    def _delete_last(self):
        if not self.boundaries:
            return

        removed = self.boundaries.pop()
        for artist in self._completed_artists.pop():
            try:
                artist.remove()
            except Exception:
                pass

        # Invalidate cached background so the removed artist's pixels aren't
        # restored from cache on the next blit.
        self._bg = None
        self._blit()
        self._refresh_status()
        print(f"Deleted last line ({removed.label}). {len(self.boundaries)} remaining.")

    def _save(self):
        if not self.boundaries:
            print("No lines to save.")
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self._save_label_boundaries_csv()
        self._save_label_mask()

        self._set_message(f"Saved {len(self.boundaries)} labeled lines.")

    def _save_label_boundaries_csv(self):
        """Save all boundary points with mapped semantics as x,y,boundary CSV (like all_boundaries.csv)."""
        rows: list[dict[str, float | str]] = []
        for boundary in self.boundaries:
            mapped = MAPPED_LABELS.get(boundary.slug)
            if mapped is None:
                continue  # skip Edge / unlisted
            interp = self._interpolate_in_image(boundary)
            for x, y in interp:
                rows.append({"x": float(x), "y": float(y), "boundary": mapped})
        if not rows:
            print("  [label_boundaries] No non-Edge boundaries to save.")
            return

        csv_path = self.save_path.parent / "label_boundaries.csv"
        pd.DataFrame(rows, columns=["x", "y", "boundary"]).to_csv(csv_path, index=False)
        print(f"  label boundaries ({len(rows)} pts) -> {csv_path.name}")

    def _save_label_mask(self):
        """Build a colorful layer mask from boundary lines and save as label_mask.png.

        Layers are filled between consecutive boundary pairs (polygon fill).
        """
        h, w = self._image_shape
        label_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Collect first boundary (by slug) for each ordered layer
        first_by_slug: dict[str, LabeledBoundary] = {}
        for boundary in self.boundaries:
            if boundary.slug not in first_by_slug and boundary.slug in _LAYER_MASK_ORDER:
                first_by_slug[boundary.slug] = boundary

        # Colour per mask layer matching run_new_pipeline.py palette (RGB)
        layer_color_map = _LAYER_MASK_COLORS_RGB

        # Fill between consecutive boundary pairs
        for i in range(len(_LAYER_MASK_ORDER) - 1):
            slug_top = _LAYER_MASK_ORDER[i]
            slug_bot = _LAYER_MASK_ORDER[i + 1]
            color = layer_color_map[slug_top]

            if slug_top not in first_by_slug or slug_bot not in first_by_slug:
                continue

            top_pts = self._interpolate_in_image(first_by_slug[slug_top])
            bot_pts = self._interpolate_in_image(first_by_slug[slug_bot])

            if len(top_pts) < 2 or len(bot_pts) < 2:
                continue

            # Sort by x coordinate (left → right)
            top_sorted = top_pts[np.argsort(top_pts[:, 0])].astype(np.int32)
            bot_sorted = bot_pts[np.argsort(bot_pts[:, 0])].astype(np.int32)

            # Extend first / last point horizontally to image edges so the
            # polygon fill covers the full width.
            if top_sorted[0, 0] > 0:
                top_sorted = np.vstack([[[0, top_sorted[0, 1]]], top_sorted])
            if top_sorted[-1, 0] < w - 1:
                top_sorted = np.vstack([top_sorted, [[w - 1, top_sorted[-1, 1]]]])

            if bot_sorted[0, 0] > 0:
                bot_sorted = np.vstack([[[0, bot_sorted[0, 1]]], bot_sorted])
            if bot_sorted[-1, 0] < w - 1:
                bot_sorted = np.vstack([bot_sorted, [[w - 1, bot_sorted[-1, 1]]]])

            # Closed polygon: top left→right, then bottom right→left
            polygon = np.vstack([top_sorted, bot_sorted[::-1]])

            cv2.fillPoly(label_mask, [polygon.reshape(-1, 1, 2)], color)

        mask_path = self.save_path.parent / "label_mask.png"
        cv2.imwrite(str(mask_path), label_mask)
        print(f"  label mask -> {mask_path.name}")

    def _redraw_current(self):
        self._remove_artist("_cur_line")
        self._remove_artist("_cur_dots")
        if not self.current_pts:
            return

        label, _slug, color = self.current_semantic
        xs = [p[0] for p in self.current_pts]
        ys = [p[1] for p in self.current_pts]
        (self._cur_line,) = self.ax.plot(xs, ys, "--", color=color, linewidth=1.5, zorder=12)
        self._cur_dots = self.ax.scatter(xs, ys, s=25, c=color, zorder=13, linewidths=0)

    def _on_draw(self, _event):
        try:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg = None

    def _blit(self):
        """Redraw only overlay artists — image background stays static."""
        if self._bg is None:
            self.fig.canvas.draw_idle()
            return
        fig, ax = self.fig, self.ax
        fig.canvas.restore_region(self._bg)

        for line, dots, text in self._completed_artists:
            ax.draw_artist(line)
            ax.draw_artist(dots)
            ax.draw_artist(text)
        if self._cur_line is not None:
            ax.draw_artist(self._cur_line)
        if self._cur_dots is not None:
            ax.draw_artist(self._cur_dots)

        ax.draw_artist(ax.title)
        fig.canvas.blit(fig.bbox)

    def _refresh_status(self, message: str = ""):
        n_done = len(self.boundaries)
        n_cur = len(self.current_pts)
        label, _slug, _color = self.current_semantic
        title = (
            f"Completed: {n_done} | Current vertices: {n_cur} | Label: {label}"
            f"{' | ' + message if message else ''}\n"
            "LClick=point  RClick/Enter=finish  T=cycle label  "
            "0/E Edge  G Gray/pia  1 L1  2/3 L2/3  4 L4  5/6/W White  "
            "U=undo  C=cancel  D=delete  S=save  Q=quit"
        )
        self.ax.set_title(title, fontsize=8.5, color="white", loc="left", pad=6)
        self._blit()

    def _set_message(self, msg: str, warn: bool = False):
        print(("WARNING: " if warn else "") + msg)
        self._refresh_status(msg)

    def _remove_artist(self, attr: str):
        artist = getattr(self, attr, None)
        if artist is None:
            return
        try:
            artist.remove()
        except Exception:
            pass
        setattr(self, attr, None)

    def _toolbar_active(self) -> bool:
        try:
            toolbar = self.fig.canvas.toolbar
            if toolbar is None:
                return False
            mode = getattr(toolbar, "mode", "")
            if mode is None:
                return False
            mode_text = str(mode).strip().lower()
            return "pan" in mode_text or "zoom" in mode_text
        except Exception:
            return False

    @staticmethod
    def _interpolate(pts: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            dist = np.hypot(x1 - x0, y1 - y0)
            n = max(int(np.ceil(dist)), 2)
            xs = np.linspace(x0, x1, n, endpoint=False)
            ys = np.linspace(y0, y1, n, endpoint=False)
            segments.append(np.column_stack([xs, ys]))
        segments.append(pts[-1:])
        return np.round(np.vstack(segments)).astype(np.int32)

    def _interpolate_in_image(self, boundary: LabeledBoundary) -> np.ndarray:
        h, w = self._image_shape
        return self._clip_points(self._interpolate(boundary.points), w, h)

    @staticmethod
    def _clip_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.int32)
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        return pts[valid]


def main():
    parser = argparse.ArgumentParser(
        description="Draw semantic cortical label lines and save label boundaries + mask."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the saved files (default: beside the image)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if args.output:
        save_dir = Path(args.output)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = image_path.parent
    save_path = save_dir / f"{image_path.stem}_labels.csv"

    print(f"Loading: {image_path}")
    image = load_display_image(image_path)
    h, w = image.shape[:2]
    print(f"  {w}w x {h}h px\n")

    print("=" * 72)
    print("Controls:")
    print("  Left click          add vertex")
    print("  Right click/Enter   finalize current semantic line")
    print("  T                   cycle semantic label")
    print("  0/E                 Edge (default)")
    print("  G                   Gray/pia")
    print("  1                   L1")
    print("  2/3                 L2/3")
    print("  4                   L4")
    print("  5/6/W               L5/6(White)")
    print("  S                   save  (label_boundaries.csv + label_mask.png)")
    print("  Q/Esc               save and quit")
    print("=" * 72)
    print(f"Output dir: {save_dir}\n")

    max_in = 14.0
    ratio = w / h
    if ratio >= 1:
        fig_w, fig_h = max_in, max_in / ratio
    else:
        fig_w, fig_h = max_in * ratio, max_in

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    draw_margin = max(50, int(round(max(w, h) * 0.05)))
    ax.set_xlim(-draw_margin, w + draw_margin)
    ax.set_ylim(h + draw_margin, -draw_margin)
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    drawer = LabelDrawer(
        fig,
        ax,
        save_path,
        image_shape=image.shape[:2],
        gray_img=image,
        draw_margin=draw_margin,
    )
    # Keep a strong reference while the Matplotlib window is open.  Callback
    # registries may weak-reference bound methods, so an unnamed instance can
    # be garbage-collected and leave clicks with no handler.
    fig._label_drawer = drawer
    plt.show()


if __name__ == "__main__":
    main()
