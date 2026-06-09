#!/usr/bin/env python3
"""
Generic overlay plot template for budget-allocation curves on 16-frame image rows.

This script contains no task-specific data. Users provide all case metadata through
an external JSON config.

Expected JSON schema:

{
  "figure": {
    "width": 17.0,
    "height": 7.9,
    "curve_strip_height": 0.53,
    "block_gap": -0.1,
    "margins": {
      "left": 0.02,
      "right": 0.995,
      "top": 0.995,
      "bottom": 0.005
    },
    "image_stretch_h": 1.2
  },
  "colors": {
    "V-CAST": "#CF3E40",
    "VidCom2": "#6BB983",
    "FastVID": "#F39A3D",
    "Uniform": "#A9A9A9"
  },
  "curve_order": ["Uniform", "V-CAST", "VidCom2", "FastVID"],
  "cases": [
    {
      "id": "example-case",
      "image_dir": "example-case",
      "curves": {
        "V-CAST": [16 values],
        "VidCom2": [16 values],
        "FastVID": [16 values],
        "Uniform": [0.25, ..., 0.25]
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

try:
    from scipy.interpolate import make_interp_spline  # type: ignore

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


DEFAULT_COLORS = {
    "V-CAST": "#CF3E40",
    "VidCom2": "#6BB983",
    "FastVID": "#F39A3D",
    "Uniform": "#A9A9A9",
}
DEFAULT_CURVE_ORDER = ["Uniform", "V-CAST", "VidCom2", "FastVID"]


def _catmull_rom_dense(y: np.ndarray, points: int) -> np.ndarray:
    n = len(y)
    if n < 4:
        t = np.arange(n, dtype=float)
        td = np.linspace(0.0, float(n - 1), points)
        return np.interp(td, t, y)

    td = np.linspace(0.0, float(n - 1), points)
    out = np.empty_like(td)
    for i, t in enumerate(td):
        j = int(np.floor(t))
        j = max(0, min(j, n - 2))
        u = t - float(j)
        j0 = max(0, j - 1)
        j1 = j
        j2 = j + 1
        j3 = min(n - 1, j + 2)
        p0, p1, p2, p3 = y[j0], y[j1], y[j2], y[j3]
        out[i] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (u * u)
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * (u * u * u)
        )
    return out


def smooth_curve(x: np.ndarray, y: np.ndarray, points: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    x_dense = np.linspace(x.min(), x.max(), points)
    if _HAS_SCIPY and len(x) >= 4:
        spl = make_interp_spline(x, y, k=3)
        y_dense = spl(x_dense)
    else:
        y_dense = _catmull_rom_dense(np.asarray(y, dtype=float), points)
    return x_dense, y_dense


def first_number_key(path: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)", path.name)
    n = int(m.group(1)) if m else 10**9
    return n, path.name


def list_sorted_images(case_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: List[Path] = []
    for pat in exts:
        files.extend(case_dir.glob(pat))
    return sorted(files, key=first_number_key)


def load_rgb_image(path: Path, max_side: int = 640, crop_aspect: float = 1.2) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    r = max(1e-6, float(crop_aspect))

    if (w / h) >= r:
        ch = h
        cw = int(round(ch * r))
    else:
        cw = w
        ch = int(round(cw / r))

    left = (w - cw) // 2
    top = (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))

    longest = max(cw, ch)
    if longest > max_side:
        scale = max_side / float(longest)
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    return np.asarray(img)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a JSON object.")
    return cfg


def build_uniform_curve(length: int, value: float = 0.25) -> List[float]:
    return [float(value)] * int(length)


def normalize_case(curves: Dict[str, List[float]], curve_order: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in curve_order:
        if name == "Uniform" and name not in curves:
            arr = build_uniform_curve(16, value=0.25)
        else:
            if name not in curves:
                raise RuntimeError(f"Missing curve '{name}' in case config.")
            arr = curves[name]
        vals = np.asarray(arr, dtype=float)
        if vals.shape != (16,):
            raise RuntimeError(f"Curve '{name}' must contain exactly 16 values.")
        out[name] = vals
    return out


def plot_case_curve(
    ax: plt.Axes,
    curves: Dict[str, np.ndarray],
    curve_order: List[str],
    colors: Dict[str, str],
) -> None:
    x = np.arange(16, dtype=float)
    x_aug = np.concatenate(([-0.5], x, [15.5]))

    ymax_parts = []
    for zorder, name in enumerate(curve_order, start=1):
        vals = curves[name]
        vals_aug = np.concatenate(([0.25], vals, [0.25]))
        xd, yd = smooth_curve(x_aug, vals_aug)
        yd = np.clip(yd, 0.0, 1.0)
        ymax_parts.append(vals_aug)

        alpha = 0.22 if name == "Uniform" else 0.24
        linestyle = "--" if name == "Uniform" else "-"
        linewidth = 1.6 if name == "Uniform" else 2.5
        ax.fill_between(xd, yd, 0.0, color=colors[name], alpha=alpha, zorder=zorder)
        ax.plot(xd, yd, color=colors[name], linewidth=linewidth, linestyle=linestyle, zorder=zorder + 6)

    ymax = np.nanmax(np.concatenate(ymax_parts))
    pad_up = max(0.02, 0.06 * (ymax + 1e-8))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0.0, min(1.0, ymax + pad_up))
    ax.axis("off")


def plot_case_images(ax: plt.Axes, image_paths: List[Path], frame_zoom: float, image_stretch_h: float) -> None:
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    if len(image_paths) > 16:
        image_paths = list(image_paths[:16])

    for i, p in enumerate(image_paths):
        img = load_rgb_image(p)
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * image_stretch_h)))
        img = np.asarray(Image.fromarray(img).resize((w, new_h), Image.Resampling.BICUBIC))
        oi = OffsetImage(img, zoom=frame_zoom, interpolation="bicubic")
        ab = AnnotationBbox(oi, (i, 1.0), frameon=False, box_alignment=(0.5, 1.0))
        ax.add_artist(ab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="JSON config containing cases and curve data.")
    ap.add_argument("--picture_root", type=Path, default=None, help="Root folder for case image directories.")
    ap.add_argument("--out", type=Path, required=True, help="Output PDF path. SVG and PNG share the same stem.")
    ap.add_argument("--frame_zoom", type=float, default=0.118)
    ap.add_argument("--png_dpi", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    figure_cfg = cfg.get("figure", {})
    colors = {**DEFAULT_COLORS, **cfg.get("colors", {})}
    curve_order = list(cfg.get("curve_order", DEFAULT_CURVE_ORDER))
    cases = cfg.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Config must provide a non-empty 'cases' list.")

    picture_root = args.picture_root if args.picture_root is not None else args.config.parent

    fig_w = float(figure_cfg.get("width", 17.0))
    fig_h = float(figure_cfg.get("height", 7.9))
    curve_strip_h = float(figure_cfg.get("curve_strip_height", 0.53))
    block_gap = float(figure_cfg.get("block_gap", -0.1))
    image_stretch_h = float(figure_cfg.get("image_stretch_h", 1.2))
    margins = figure_cfg.get("margins", {})
    margin_left = float(margins.get("left", 0.02))
    margin_right = float(margins.get("right", 0.995))
    margin_top = float(margins.get("top", 0.995))
    margin_bottom = float(margins.get("bottom", 0.005))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=240)
    gs = fig.add_gridspec(len(cases), 1, hspace=block_gap)

    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RuntimeError("Each case entry must be a JSON object.")
        image_dir = case.get("image_dir")
        curves = case.get("curves")
        if not image_dir or not isinstance(curves, dict):
            raise RuntimeError("Each case must define 'image_dir' and 'curves'.")

        case_dir = Path(image_dir)
        if not case_dir.is_absolute():
            case_dir = picture_root / case_dir
        if not case_dir.exists():
            raise RuntimeError(f"Folder not found: {case_dir}")

        image_paths = list_sorted_images(case_dir)
        normalized_curves = normalize_case(curves, curve_order)

        ax_block = fig.add_subplot(gs[i, 0])
        ax_block.axis("off")
        img_h = 1.0 - curve_strip_h
        ax_curve = ax_block.inset_axes([0.0, img_h, 1.0, curve_strip_h])
        ax_imgs = ax_block.inset_axes([0.0, 0.0, 1.0, img_h])
        plot_case_curve(ax_curve, normalized_curves, curve_order, colors)
        plot_case_images(ax_imgs, image_paths, frame_zoom=args.frame_zoom, image_stretch_h=image_stretch_h)

    fig.subplots_adjust(
        left=margin_left,
        right=margin_right,
        top=margin_top,
        bottom=margin_bottom,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out
    out_svg = out_pdf.with_suffix(".svg")
    out_png = out_pdf.with_suffix(".png")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=int(args.png_dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
