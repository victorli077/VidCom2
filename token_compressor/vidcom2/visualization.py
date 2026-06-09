from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import torch
from PIL import Image


def _describe_video_input_shape(pixel_values_videos: Any) -> Any:
    """Best-effort shape info for debugging frame extraction issues."""
    if pixel_values_videos is None:
        return None
    if torch.is_tensor(pixel_values_videos):
        return list(pixel_values_videos.shape)
    if isinstance(pixel_values_videos, (list, tuple)):
        out: List[Any] = []
        for v in pixel_values_videos:
            if torch.is_tensor(v):
                out.append(list(v.shape))
            else:
                out.append(type(v).__name__)
        return out
    return type(pixel_values_videos).__name__


def _to_2d_patch_tokens(pixel_values_videos: Any) -> Optional[torch.Tensor]:
    """
    Best-effort normalize patchified video input to [N, D].
    """
    if pixel_values_videos is None:
        return None

    x = pixel_values_videos
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    if not torch.is_tensor(x):
        return None

    x = x.detach()
    if x.dim() == 2:
        return x
    if x.dim() > 2:
        # Some pipelines may wrap leading singleton dims.
        while x.dim() > 2 and x.shape[0] == 1:
            x = x[0]
        if x.dim() == 2:
            return x
    return None


def _reconstruct_frames_from_patchified(
    pixel_values_videos: Any,
    video_grid_thw: Optional[Any],
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Reconstruct approximate RGB frames from patchified video tensor [N, D].
    Typical D for Qwen2.5-Omni is 1176 = temporal_patch(2) * 3 * 14 * 14.
    Returns [T, 3, H_img, W_img] or empty tensor when reconstruction is not possible.
    """
    tokens = _to_2d_patch_tokens(pixel_values_videos)
    if tokens is None or video_grid_thw is None:
        return torch.empty(0)

    if torch.is_tensor(video_grid_thw):
        g = video_grid_thw.reshape(-1).tolist()
    elif isinstance(video_grid_thw, (list, tuple)):
        g = list(video_grid_thw)
    else:
        return torch.empty(0)

    if len(g) < 3:
        return torch.empty(0)
    t, h, w = int(g[0]), int(g[1]), int(g[2])
    if t <= 0 or h <= 0 or w <= 0:
        return torch.empty(0)

    n = t * h * w
    if tokens.shape[0] < n:
        return torch.empty(0)

    d = int(tokens.shape[1])
    unit = 3 * patch_size * patch_size
    if d % unit != 0:
        return torch.empty(0)
    temporal_patch = d // unit
    if temporal_patch <= 0:
        return torch.empty(0)

    x = tokens[:n].float().view(t, h, w, temporal_patch, 3, patch_size, patch_size)
    # Use the first temporal slice in each temporal patch group as representative.
    x = x[:, :, :, 0]  # [T, H, W, 3, ps, ps]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [T, 3, H, ps, W, ps]
    x = x.view(t, 3, h * patch_size, w * patch_size)
    return x


def _to_frame_tensor_list(pixel_values_videos: Any) -> torch.Tensor:
    """
    Normalize video tensor layout to [N, C, H, W] for thumbnail sampling.
    Supports common layouts:
      - [N, C, H, W]
      - [B, N, C, H, W] (uses first batch item)
    """
    if pixel_values_videos is None:
        return torch.empty(0)

    if isinstance(pixel_values_videos, (list, tuple)):
        if len(pixel_values_videos) == 0:
            return torch.empty(0)
        pixel_values_videos = pixel_values_videos[0]

    if not torch.is_tensor(pixel_values_videos):
        return torch.empty(0)

    x = pixel_values_videos.detach()

    # Peel leading dims (batch / multi-video wrappers) until 4D.
    while x.dim() > 4:
        if x.shape[0] == 0:
            return torch.empty(0, device=x.device, dtype=x.dtype)
        x = x[0]

    if x.dim() != 4:
        return torch.empty(0, device=x.device, dtype=x.dtype)

    # Normalize to [N, C, H, W] for downstream frame sampling.
    if x.shape[1] in (1, 3):
        return x
    if x.shape[0] in (1, 3):
        return x.permute(1, 0, 2, 3).contiguous()
    if x.shape[-1] in (1, 3):
        return x.permute(0, 3, 1, 2).contiguous()

    # Unknown layout (e.g. already patchified features); cannot reconstruct frames.
    return torch.empty(0, device=x.device, dtype=x.dtype)


def _sample_frames_from_video_file(
    source_video_path: Optional[str],
    num_chunks: int,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Sample frames directly from the original dataset video file.
    Returns PIL RGB frames and sampled frame indices.
    """
    if not source_video_path or num_chunks <= 0:
        return [], []
    if not os.path.exists(source_video_path):
        return [], []

    try:
        import cv2
    except Exception:
        return [], []

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        return [], []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], []

    idx = torch.linspace(0, max(0, total - 1), steps=num_chunks).round().long().tolist()
    frames: List[Image.Image] = []
    kept_idx: List[int] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        kept_idx.append(int(i))
    cap.release()
    return frames, kept_idx


def sample_chunk_frames(
    pixel_values_videos: Any,
    num_chunks: int,
    video_grid_thw: Optional[Any] = None,
    source_video_path: Optional[str] = None,
) -> Tuple[List[Image.Image], List[int], str]:
    """
    Sample one representative frame per temporal chunk from model input video tensor.
    Note: these frames are sampled from `pixel_values_videos` (already-decoded model input),
    not the raw original full-fps video stream.
    Returns PIL frames and sampled frame indices in the source tensor.
    """
    # Prefer original dataset video for clearer visualization.
    file_frames, file_indices = _sample_frames_from_video_file(source_video_path, num_chunks)
    if len(file_frames) > 0:
        return file_frames, file_indices, "dataset_video"

    frames = _to_frame_tensor_list(pixel_values_videos)
    if frames.numel() == 0:
        frames = _reconstruct_frames_from_patchified(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
    if frames.numel() == 0 or num_chunks <= 0:
        return [], [], "none"

    n = int(frames.shape[0])
    idx = torch.linspace(0, max(0, n - 1), steps=num_chunks).round().long().tolist()
    pil_frames: List[Image.Image] = []
    for i in idx:
        frame = torch.nan_to_num(frames[i].float(), nan=0.0, posinf=0.0, neginf=0.0)
        if frame.dim() == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)
        f_min = frame.min()
        f_max = frame.max()
        frame = (frame - f_min) / (f_max - f_min + 1e-6)
        arr = (frame * 255.0).clamp(0, 255).byte().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = arr.repeat(3, axis=-1)
        pil_frames.append(Image.fromarray(arr))
    if len(pil_frames) > 0:
        return pil_frames, idx, "model_input"
    return [], [], "none"


def _build_strip(
    frames: Sequence[Image.Image],
    cell_h: int = 120,
    cell_w: Optional[int] = None,
) -> Optional[Image.Image]:
    if len(frames) == 0:
        return None

    if cell_w is None:
        # Use one fixed slot width per chunk to guarantee bar<->frame alignment.
        # Width follows the widest frame when normalized to cell_h.
        norm_w = [
            max(1, int(round(img.size[0] * (cell_h / max(1, img.size[1])))))
            for img in frames
        ]
        cell_w = max(norm_w) if len(norm_w) > 0 else cell_h

    canvas = Image.new("RGB", (cell_w * len(frames), cell_h), (245, 245, 245))
    for i, img in enumerate(frames):
        w, h = img.size
        scale = min(cell_w / max(1, w), cell_h / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        x0 = i * cell_w + (cell_w - new_w) // 2
        y0 = (cell_h - new_h) // 2
        canvas.paste(resized, (x0, y0))
    return canvas


def _resample_series(values: Sequence[float], target_len: int) -> List[float]:
    if target_len <= 0 or len(values) == 0:
        return []
    if len(values) == target_len:
        return [float(v) for v in values]
    idx = torch.linspace(0, len(values) - 1, steps=target_len).round().long().tolist()
    return [float(values[i]) for i in idx]


def render_budget_comparison(
    budgets_visual: Sequence[int],
    budgets_audio: Sequence[int],
    frame_strip: Optional[Image.Image],
    audio_token_norm: Optional[Sequence[float]],
    out_path: str,
    title: str = "VidCom2 Budget Comparison",
) -> Optional[str]:
    """
    Save a comparison plot:
      - top: bar chart (visual-only vs audio-guided budgets)
      - bottom: corresponding frame strip, aligned by temporal chunk index
    """
    if len(budgets_visual) == 0 or len(budgets_audio) == 0:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    t = min(len(budgets_visual), len(budgets_audio))
    x = list(range(t))
    bv = list(budgets_visual[:t])
    ba = list(budgets_audio[:t])

    if frame_strip is None:
        x_coords = [float(i) for i in x]
        series_width = 0.42
        xlim = (-0.5, t - 0.5)
        strip_w = None
        strip_h = None
    else:
        strip_w, strip_h = frame_strip.size
        chunk_w = float(strip_w) / float(max(1, t))
        # Use frame-strip pixel coordinates so bars align to chunk image centers
        # without stretching the image.
        x_coords = [(i + 0.5) * chunk_w for i in x]
        series_width = 0.42 * chunk_w
        xlim = (0.0, float(strip_w))

    has_audio_panel = audio_token_norm is not None and len(audio_token_norm) > 0
    if frame_strip is None and not has_audio_panel:
        fig, ax = plt.subplots(figsize=(max(12, t * 0.55), 4.5))
        axes = [ax]
        ax_audio = None
        ax_img = None
    elif frame_strip is None and has_audio_panel:
        fig = plt.figure(figsize=(max(12, t * 0.55), 6.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.8], hspace=0.10)
        ax = fig.add_subplot(gs[0])
        ax_audio = fig.add_subplot(gs[1], sharex=ax)
        ax_img = None
        axes = [ax, ax_audio]
    elif frame_strip is not None and not has_audio_panel:
        fig = plt.figure(figsize=(max(12, t * 0.55), 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 2.0], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_img = fig.add_subplot(gs[1], sharex=ax)
        ax_audio = None
        axes = [ax, ax_img]
    else:
        fig = plt.figure(figsize=(max(12, t * 0.55), 7.8))
        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.8, 2.0], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_audio = fig.add_subplot(gs[1], sharex=ax)
        ax_img = fig.add_subplot(gs[2], sharex=ax)
        axes = [ax, ax_audio, ax_img]

    ax.bar(
        [p - series_width / 2 for p in x_coords],
        bv,
        width=series_width,
        color="#c9d7eb",
        edgecolor="#243B5A",
        label="Visual-only",
    )
    ax.bar(
        [p + series_width / 2 for p in x_coords],
        ba,
        width=series_width,
        color="#2f4f88",
        alpha=0.9,
        edgecolor="#1f2e49",
        label="Audio-guided",
    )
    ax.plot(x_coords, ba, color="#e7cf49", marker="*", linestyle="--", linewidth=2.0, markersize=10)
    ax.set_title(title)
    ax.set_xlabel("Temporal Chunk Index")
    ax.set_ylabel("Kept Tokens (Budget)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_xlim(*xlim)
    ax.margins(x=0)
    ax.set_xticks(x_coords)
    ax.set_xticklabels([str(i) for i in x])

    if has_audio_panel and ax_audio is not None:
        token_vals = _resample_series([float(v) for v in audio_token_norm], t)
        token_vals = [max(0.0, min(1.0, v)) for v in token_vals]
        delta_vals = [0.0]
        for i in range(1, len(token_vals)):
            delta_vals.append(abs(token_vals[i] - token_vals[i - 1]))
        delta_max = max(delta_vals) if len(delta_vals) > 0 else 0.0
        if delta_max > 0:
            delta_vals = [v / delta_max for v in delta_vals]

        ax_audio.plot(
            x_coords,
            token_vals,
            color="#ef4444",
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            alpha=0.9,
            label="Normalized audio token",
        )
        ax_audio.set_ylim(0.0, 1.05)
        ax_audio.set_ylabel("Token", color="#ef4444")
        ax_audio.grid(axis="y", alpha=0.18)

        ax_delta = ax_audio.twinx()
        ax_delta.plot(
            x_coords,
            delta_vals,
            color="#0ea5e9",
            marker="x",
            markersize=3.0,
            linewidth=1.2,
            label="Adjacent change |delta|",
        )
        ax_delta.set_ylim(0.0, 1.05)
        ax_delta.set_ylabel("|delta|", color="#0ea5e9")

        h1, l1 = ax_audio.get_legend_handles_labels()
        h2, l2 = ax_delta.get_legend_handles_labels()
        if len(h1) + len(h2) > 0:
            ax_audio.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        ax_audio.set_title("Audio Guidance Token: Normalized Value + Adjacent Change", fontsize=10)

    if frame_strip is not None:
        # Keep original aspect ratio and map x to strip pixel coordinates.
        ax_img.imshow(
            frame_strip,
            extent=(0.0, float(strip_w), float(strip_h), 0.0),
            aspect="equal",
        )
        ax_img.set_xlim(*xlim)
        ax_img.set_ylim(float(strip_h), 0.0)
        ax_img.axis("off")
        ax_img.set_title("Representative Frame per Chunk", fontsize=10)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_budget_comparison_artifact(
    output_dir: str,
    case_name: str,
    budgets_visual: Sequence[int],
    budgets_audio: Sequence[int],
    pixel_values_videos: Optional[torch.Tensor],
    video_grid_thw: Optional[Any] = None,
    source_video_path: Optional[str] = None,
    audio_token_norm: Optional[Sequence[float]] = None,
    title: str = "VidCom2 Budget Comparison",
    extra_meta: Optional[Dict[str, Any]] = None,
    render_now: bool = True,
) -> Dict[str, Any]:
    """
    Save artifacts for one video case:
      - comparison plot png
      - metadata json (budgets + sampled frame indices)
      - sampled thumbnail frames
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = min(len(budgets_visual), len(budgets_audio))
    frames, frame_indices, frame_source = sample_chunk_frames(
        pixel_values_videos,
        num_chunks=t,
        video_grid_thw=video_grid_thw,
        source_video_path=source_video_path,
    )
    strip = _build_strip(frames, cell_h=120)

    frame_dir = out_dir / f"{case_name}_frames"
    frame_paths: List[str] = []
    if len(frames) > 0:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            p = frame_dir / f"chunk_{i:03d}.jpg"
            img.save(p, quality=90)
            frame_paths.append(str(p))

    plot_saved = None
    if render_now:
        plot_path = out_dir / f"{case_name}_budget_compare.png"
        plot_saved = render_budget_comparison(
            budgets_visual=budgets_visual,
            budgets_audio=budgets_audio,
            frame_strip=strip,
            audio_token_norm=audio_token_norm,
            out_path=str(plot_path),
            title=title,
        )

    meta: Dict[str, Any] = {
        "case_name": case_name,
        "num_chunks": t,
        "budgets_visual": list(budgets_visual[:t]),
        "budgets_audio": list(budgets_audio[:t]),
        "sampled_frame_indices": frame_indices,
        "frame_paths": frame_paths,
        "num_saved_frames": len(frame_paths),
        "pixel_values_videos_shape": _describe_video_input_shape(pixel_values_videos),
        "video_grid_thw": (
            video_grid_thw.reshape(-1).tolist() if torch.is_tensor(video_grid_thw) else list(video_grid_thw) if isinstance(video_grid_thw, (list, tuple)) else None
        ),
        "source_video_path": source_video_path,
        "audio_token_norm": (
            [float(v) for v in audio_token_norm] if audio_token_norm is not None else None
        ),
        "frame_source": frame_source,
        "plot_path": plot_saved,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = out_dir / f"{case_name}_budget_compare.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "plot_path": plot_saved,
        "meta_path": str(meta_path),
        "frame_dir": str(frame_dir) if len(frame_paths) > 0 else None,
    }


def render_budget_from_metadata(meta_path: str, out_path: Optional[str] = None, title: Optional[str] = None) -> Optional[str]:
    """
    Re-render comparison plot from saved metadata json.
    Useful when evaluation is interrupted and plotting is done later.
    """
    p = Path(meta_path)
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    frame_paths = meta.get("frame_paths", [])
    frames: List[Image.Image] = []
    for fp in frame_paths:
        img_path = Path(fp)
        if img_path.exists():
            frames.append(Image.open(img_path).convert("RGB"))
    strip = _build_strip(frames, cell_h=120)

    budgets_visual = meta.get("budgets_visual", [])
    budgets_audio = meta.get("budgets_audio", [])
    audio_token_norm = meta.get("audio_token_norm", None)
    if out_path is None:
        out_path = str(p.with_suffix("").with_name(p.stem.replace("_budget_compare", "") + "_budget_compare.png"))
    if title is None:
        title = "WorldSense Budget: Visual-only vs Audio-guided"
    return render_budget_comparison(
        budgets_visual=budgets_visual,
        budgets_audio=budgets_audio,
        frame_strip=strip,
        audio_token_norm=audio_token_norm,
        out_path=out_path,
        title=title,
    )
