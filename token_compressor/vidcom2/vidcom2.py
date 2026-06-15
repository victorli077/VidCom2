import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any

# Configuration: 'mapper' determines the index mapping strategy
# 'tpf' is default token_per_frame, can be overridden for qwen2_vl
MODEL_SPECS = {
    "llava_ov": {"tpf": 196, "mapper": "linear"},
    "llava_vid": {"tpf": 169, "mapper": "grid_vid", "grid": 13},
    "qwen2_vl": {"tpf": None, "mapper": "linear"},  # tpf provided dynamically
    "qwen2_5_vl": {"tpf": None, "mapper": "linear"},  # tpf provided dynamically
    "qwen3_vl": {"tpf": None, "mapper": "linear"},  # tpf provided dynamically
}


def get_audio_guided_frame_features(
    video_grid_thw: torch.Tensor,
    video_second_per_grid: float,
    audio_embeds: torch.Tensor,
    AUDIO_FPS: float,
    context_tokens: int = 25,
) -> torch.Tensor:
    """Align audio tokens to each video grid frame and average them to per-frame features.

    For each frame, use a context-extended window:
      [frame_start - context_tokens, frame_end + context_tokens)
    so the feature includes local temporal context around the current frame.
    """
    if audio_embeds.ndim != 2:
        raise ValueError(f"audio_embeds must be 2D (audio_seq_len, hidden_dim), got {audio_embeds.shape}")

    if torch.is_tensor(video_second_per_grid):
        video_second_per_grid = float(video_second_per_grid.item())
    else:
        video_second_per_grid = float(video_second_per_grid)

    t = int(video_grid_thw.reshape(-1)[0].item())
    audio_seq_len, hidden_dim = audio_embeds.shape
    frame_audio_features: List[torch.Tensor] = []

    for frame_idx in range(t):
        start_idx = int(frame_idx * video_second_per_grid * AUDIO_FPS)
        end_idx = int((frame_idx + 1) * video_second_per_grid * AUDIO_FPS)

        start_idx = max(0, min(start_idx, audio_seq_len))
        end_idx = max(0, min(end_idx, audio_seq_len))

        if end_idx <= start_idx:
            frame_audio_features.append(
                torch.zeros(hidden_dim, dtype=audio_embeds.dtype, device=audio_embeds.device)
            )
            continue

        # Include neighboring tokens around the current frame window.
        ext_start = max(0, start_idx - int(context_tokens))
        ext_end = min(audio_seq_len, end_idx + int(context_tokens))
        if ext_end <= ext_start:
            frame_audio_features.append(
                torch.zeros(hidden_dim, dtype=audio_embeds.dtype, device=audio_embeds.device)
            )
            continue

        frame_audio_features.append(torch.mean(audio_embeds[ext_start:ext_end], dim=0))

    if len(frame_audio_features) == 0:
        return torch.zeros((0, hidden_dim), dtype=audio_embeds.dtype, device=audio_embeds.device)
    return torch.stack(frame_audio_features, dim=0)


def compute_audio_change_scores(
    frame_audio_features: torch.Tensor,
    window_scales: Optional[List[int]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-frame audio novelty score using multiscale negative cosine similarity
    across multiple temporal windows.

    For each scale k in window_scales, compute bidirectional cosine change between
    frames separated by k, weight by hardcoded weights [3.9,1], then sum.
    Short scales capture transient audio events (door knocks, word boundaries);
    long scales capture slow timbral/energy changes (background music fade).
    A final global min-max normalization keeps output in a consistent range.

    Args:
        frame_audio_features: (num_frames, feat_dim) tensor of per-frame audio embeddings.
        window_scales:       List of temporal offsets (in frames). Defaults to [1, 3].

    Returns:
        (num_frames,) tensor of novelty scores; higher = more audio change at that frame.
    """
    if window_scales is None:
        window_scales = [1,3]
    weights = [3.9,1]
    
    num_frames = frame_audio_features.shape[0]
    if num_frames == 0:
        return torch.zeros(0, dtype=frame_audio_features.dtype, device=frame_audio_features.device)

    normed = F.normalize(frame_audio_features, p=2, dim=-1, eps=eps)
    combined = torch.zeros(num_frames, dtype=normed.dtype, device=normed.device)

    for idx, k in enumerate(window_scales):
        if k <= 0 or k >= num_frames:
            continue

        # scores[i] = -cos(frame[i], frame[i+k]) for frames that have a forward neighbor
        #           = -cos(frame[i], frame[i-k]) for frames that have a backward neighbor
        # Inner frames (with both neighbors) get the average of forward and backward.
        scores = torch.zeros(num_frames, dtype=normed.dtype, device=normed.device)

        # Forward neighbors: frame i vs frame i+k, valid for i in [0, num_frames-k-1]
        cos_fwd = (normed[:num_frames - k] * normed[k:]).sum(dim=-1)   # [num_frames - k]
        scores[:num_frames - k] += -cos_fwd

        # Backward neighbors: frame i+k vs frame i, valid for i in [0, num_frames-k-1]
        # scores[k + j] += -cos(frame[j+k], frame[j]) for j in [0, num_frames-k-1]
        cos_bwd = (normed[k:] * normed[:num_frames - k]).sum(dim=-1)  # [num_frames - k]
        scores[k:] += -cos_bwd

        # Inner frames got both directions; divide by 2 to average.
        inner_start, inner_end = k, num_frames - k
        if inner_end > inner_start:
            scores[inner_start:inner_end] *= 0.5

       weight = torch.tensor(
            weights[idx], dtype=torch.float32, device=normed.device
        )
        combined += weight * scores

    return combined


def vidcom2_compression(flattened_feat: torch.Tensor, model: str = "llava_ov",
                        base_scale: float = 0.25, frame_token_len: Optional[int] = None,
                        img_feat: Optional[torch.Tensor] = None,
                        frame_width: Optional[int] = None) -> torch.Tensor:
    """Compression pipeline supporting llava_ov, llava_vid, qwen2_vl, qwen2_5_vl, and qwen3_vl."""
    if model not in MODEL_SPECS: raise ValueError(f"Unknown model: {model}")

    spec = MODEL_SPECS[model]
    # Use dynamic tpf for qwen2_vl/qwen2_5_vl/qwen3_vl, else use constant from spec
    tpf = frame_token_len if model in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"} else spec["tpf"]
    if tpf is None: raise ValueError(f"frame_token_len required for {model}")
    # Frame grid width (square LLaVA frames default to sqrt(tpf); Qwen passes real width)
    fw = frame_width if frame_width is not None else round(tpf ** 0.5)

    # 1. Feature Analysis (Vectorized Gaussian Scores)
    sel_feat, _ = select_low_var_channels(flattened_feat)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, tpf)

    local_variation = -compute_local_variation(sel_feat, tpf).squeeze(-1)
    local_variation_norm = (local_variation - local_variation.min()) / (local_variation.max() - local_variation.min() + 1e-8)
    global_uniqueness = -vid_score.mean(dim=-1)
    global_uniqueness_norm = (global_uniqueness - global_uniqueness.min()) / (global_uniqueness.max() - global_uniqueness.min() + 1e-8)

    combined_tail = (global_uniqueness_norm[1:] + local_variation_norm) / 2
    frame_budgeting = torch.cat([global_uniqueness_norm[0:1], combined_tail])  # the more unique, the higher the frame_budgeting

    # 2. Score Fusion & Selection: uniform backbone + non-overlapping outliers
    # Strategy: Keep a uniform backbone for coverage, plus tokens different from
    # both the Global Video Mean and the Local Frame Mean (outliers).
    backbone_indices = select_backbone_indices(tpf, fw, base_scale * 0.5, flattened_feat.device)
    scales = compute_scales(frame_budgeting, base_scale * 0.5, temp=0.15)
    outlier_indices = select_outlier_indices(vid_score + frame_score, scales, tpf,
                                            exclude=backbone_indices)
    indices = [torch.cat([backbone_indices, ol]).unique() for ol in outlier_indices]

    # 3. Index Mapping (Routes to linear or grid mapper)
    return map_features(indices, flattened_feat, img_feat, spec)

def select_low_var_channels(x: torch.Tensor, ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selects least informative channels (lowest variance).
    Returns selected sub-tensor and the corresponding channel indices.
    """
    variances = x.var(dim=0, unbiased=False)
    k = int(x.shape[-1] * ratio)
    _, topk_idx = torch.topk(variances, k=k, largest=False)
    return x[:, topk_idx], topk_idx

def compute_gaussian_scores(x: torch.Tensor, tpf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Gaussian similarity to Video and Frame centers."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)
    
    # Broadcastable centers: (1, 1, C) and (B, 1, C)
    vid_center = frames.mean(dim=(0, 1), keepdim=True) 
    frame_center = frames.mean(dim=1, keepdim=True)
    
    alphas = [2**k for k in range(-3, 2)]
    v_score = _multi_scale_gaussian(frames, vid_center, alphas)
    f_score = _multi_scale_gaussian(frames, frame_center, alphas)
    return v_score, f_score

def _multi_scale_gaussian(x: torch.Tensor, center: torch.Tensor, alphas: List[float]) -> torch.Tensor:
    """Helper: vectorized multi-scale Gaussian kernel."""
    dist_sq = ((x - center) ** 2).sum(dim=-1)
    return sum(torch.exp(-dist_sq / (2 * a)) for a in alphas)

def compute_local_variation(x: torch.Tensor, tpf: int) -> torch.Tensor:
    """Computes local variation of the feature."""
    frames = x.view(-1, tpf, x.shape[-1])
    frames = F.normalize(frames, dim=-1)
    frame_center = frames.mean(dim=1, keepdim=True)
    alphas = [2**k for k in range(-3, 2)]
    return _multi_scale_gaussian(frame_center[:-1], frame_center[1:], alphas)


def compute_scales(scores: torch.Tensor, base: float, temp: float = 0.01) -> torch.Tensor:
    """Generates dynamic retention rates based on frame importance."""
    probs = F.softmax((scores - scores.max()) / temp, dim=0)
    scales = base * (1 + probs - probs.mean())
    return scales.clamp(max=1.0)


def select_backbone_indices(tpf: int, frame_width: int, ratio: float,
                            device: torch.device) -> torch.Tensor:
    """Uniformly retains ~ratio*tpf tokens via 2D-grid subsampling.

    The frame is an (H x W) token grid (flat index t -> row t//W, col t%W). We keep
    n_h = round(H*sqrt(ratio)) evenly-spaced rows and n_w = round(W*sqrt(ratio))
    evenly-spaced columns, then their grid intersection (~H*W*ratio tokens). At
    ratio=0.25 this is exactly the top-left token of every 2x2 block.
    """
    h = max(1, tpf // frame_width)
    side = ratio ** 0.5
    n_h = max(1, min(h, round(h * side)))
    n_w = max(1, min(frame_width, round(frame_width * side)))
    rows = (torch.arange(n_h, device=device) * h) // n_h
    cols = (torch.arange(n_w, device=device) * frame_width) // n_w
    grid = rows.unsqueeze(1) * frame_width + cols.unsqueeze(0)
    return grid.flatten().sort().values


def select_outlier_indices(scores: torch.Tensor, scales: torch.Tensor, tpf: int,
                           exclude: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """Selects top-k indices with lowest similarity (largest outliers).

    Tokens in `exclude` (e.g. backbone indices) are never picked, so the result
    does not overlap with the backbone selection.
    """
    ks = (scales * tpf).round().long().clamp(min=1).tolist()
    n_excl = 0 if exclude is None else exclude.numel()
    batch_indices = []
    for i, k in enumerate(ks):
        s = scores[i]
        if exclude is not None:
            s = s.clone()
            s[exclude] = float("inf")  # exclude from largest=False topk
        k = min(k, tpf - n_excl)
        # largest=False -> retain most distinct tokens
        _, idx = torch.topk(s, k=k, largest=False, sorted=False)
        batch_indices.append(idx.sort().values)
    return batch_indices

def map_features(indices: List[torch.Tensor], flat: torch.Tensor, 
                 img: Optional[torch.Tensor], spec: Dict[str, Any]) -> torch.Tensor:
    """Dispatches to the correct mapper based on model spec."""
    if spec["mapper"] == "linear":
        # Shared logic for llava_ov and qwen2_vl
        tpf = indices[0].shape[0] if not spec["tpf"] else spec["tpf"] # Infer or use const
        # Note: We must use the tpf used during scoring/splitting
        # But indices logic is per-frame, so we reconstruct global offsets
        # For variable K per frame, we need the ORIGINAL tpf stride
        stride = flat.shape[0] // len(indices)
        global_idx = _map_linear_offset(indices, stride)
        return flat[global_idx]
    
    elif spec["mapper"] == "grid_vid":
        if img is None: raise ValueError("img_feat required for grid mapping")
        global_idx = _map_grid_vid(indices, spec["grid"])
        return img[global_idx]
    return flat

def _map_linear_offset(indices: List[torch.Tensor], tpf: int) -> torch.Tensor:
    """Standard mapping: adds frame_offset to local indices (Llava-OV / Qwen2-VL)."""
    device = indices[0].device
    offsets = torch.arange(len(indices), device=device) * tpf
    return torch.cat([idx + off for idx, off in zip(indices, offsets)])

def _map_grid_vid(indices: List[torch.Tensor], h: int) -> torch.Tensor:
    """Mapping for Llava-Vid: handles 2D grid structure + newline tokens."""
    w_new = h + 1
    stride = h * w_new
    global_idx = []
    for i, idx in enumerate(indices):
        local_mapped = (idx // h) * w_new + (idx % h)
        start = i * stride
        global_idx.append(start + local_mapped)
        global_idx.append(start + (torch.arange(h, device=idx.device) * w_new + h))
    return torch.cat(global_idx)
