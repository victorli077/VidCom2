from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return str(default)
    return str(value)


def v_cast_config_from_env() -> dict[str, float | int | str | bool]:
    return {
        "retain_ratio": _env_float(
            "V_CAST_R_RATIO",
            _env_float("R_RATIO", _env_float("RETAIN_RATIO", _env_float("retain_ratio", 0.25))),
        ),
        "min_k": _env_int("V_CAST_MIN_K", 1),
        "budget_temp": _env_float("V_CAST_BUDGET_TEMP", 0.7),
        "score_mode": _env_str("V_CAST_SCORE_MODE", "hybrid").strip().lower(),
        "outlier_weight": _env_float("V_CAST_OUTLIER_WEIGHT", 1.0),
        "norm_weight": _env_float("V_CAST_NORM_WEIGHT", 1.0),
        "print": _env_bool("V_CAST_PRINT", True),
    }


def _tokens_per_frame(grid_thw_row: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t = int(grid_thw_row[0].item())
    h = int(grid_thw_row[1].item())
    w = int(grid_thw_row[2].item())
    s = max(1, int(spatial_merge_size))
    resize_h = max(1, h // s)
    resize_w = max(1, w // s)
    return t, resize_h, resize_w, resize_h * resize_w


def _compute_curvature(frame_reps: torch.Tensor) -> torch.Tensor:
    t = int(frame_reps.shape[0])
    if t <= 1:
        return torch.ones((t,), device=frame_reps.device, dtype=torch.float32)

    v_in = frame_reps[1:-1] - frame_reps[:-2]
    v_out = frame_reps[2:] - frame_reps[1:-1]
    curv = 1.0 - F.cosine_similarity(v_in, v_out, dim=-1, eps=1e-6)
    ones = torch.ones((1,), device=frame_reps.device, dtype=curv.dtype)
    return torch.cat([ones, curv, ones], dim=0)


def _allocate_budget_per_frame(
    curvature: torch.Tensor,
    total_budget: int,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    t = int(curvature.shape[0])
    if t <= 0:
        return torch.zeros((0,), device=curvature.device, dtype=torch.long)

    total_budget = int(max(0, min(int(total_budget), int(t * max_k))))
    min_k = int(max(0, min(int(min_k), int(max_k))))
    min_k_eff = 0 if total_budget < t * min_k else min_k

    base = torch.full((t,), int(min_k_eff), device=curvature.device, dtype=torch.long)
    remaining = int(total_budget - int(base.sum().item()))
    if remaining <= 0:
        return base

    weights = curvature.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max(0, max_k - min_k_eff)
    extra = torch.minimum(extra, torch.full_like(extra, int(max_extra)))
    alloc = base + extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if frac[idx].item() < 0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1

    return alloc


def _score_frame_tokens(
    frame_tokens: torch.Tensor,
    frame_rep: torch.Tensor,
    *,
    score_mode: str,
    outlier_weight: float,
    norm_weight: float,
) -> torch.Tensor:
    sim = F.cosine_similarity(frame_tokens, frame_rep.unsqueeze(0), dim=-1, eps=1e-6)
    outlier = (1.0 - sim).float()
    norm = frame_tokens.float().norm(dim=-1)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)

    if score_mode == "outlier":
        return outlier
    if score_mode == "norm":
        return norm
    return outlier_weight * outlier + norm_weight * norm


def v_cast_compression(
    flat_features: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    *,
    retain_ratio: float,
    min_k: int,
    budget_temp: float,
    score_mode: str,
    outlier_weight: float,
    norm_weight: float,
    verbose: bool,
    tag: str = "",
) -> torch.Tensor:
    t, _, _, tokens_per_frame = _tokens_per_frame(grid_thw, spatial_merge_size)
    if t <= 0 or tokens_per_frame <= 0 or flat_features.numel() == 0:
        return torch.arange(flat_features.shape[0], device=flat_features.device)

    expected_tokens = t * tokens_per_frame
    if expected_tokens != int(flat_features.shape[0]):
        if verbose:
            print(
                f"[V-CAST] {tag}token count mismatch: expected {expected_tokens}, got {int(flat_features.shape[0])}. "
                "Skipping compression.",
                flush=True,
            )
        return torch.arange(flat_features.shape[0], device=flat_features.device)

    if t <= 1:
        return torch.arange(flat_features.shape[0], device=flat_features.device)

    retain_ratio = float(retain_ratio if retain_ratio > 0 else 0.25)
    min_k = int(max(0, min(int(min_k), int(tokens_per_frame))))
    frames = flat_features.reshape(t, tokens_per_frame, -1)
    frame_reps = F.normalize(frames.mean(dim=1), dim=-1, eps=1e-6)
    curvature = _compute_curvature(frame_reps)
    budget_temp = max(1e-4, float(budget_temp))
    weights = torch.softmax(curvature.float() / budget_temp, dim=0)

    total_budget = int(round(float(t * tokens_per_frame) * retain_ratio))
    total_budget = max(1, min(int(t * tokens_per_frame), total_budget))
    if min_k > 0:
        total_budget = max(total_budget, int(t * min_k))
    k_t = _allocate_budget_per_frame(weights, total_budget, min_k=min_k, max_k=tokens_per_frame)

    keep_indices = []
    for frame_idx in range(t):
        k = int(k_t[frame_idx].item())
        if k <= 0:
            continue
        frame_tokens = frames[frame_idx]
        score = _score_frame_tokens(
            frame_tokens,
            frame_reps[frame_idx],
            score_mode=score_mode,
            outlier_weight=float(outlier_weight),
            norm_weight=float(norm_weight),
        )
        topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        topk_idx, _ = torch.sort(topk_idx)
        keep_indices.append(topk_idx + frame_idx * tokens_per_frame)

    if not keep_indices:
        return torch.zeros((0,), device=flat_features.device, dtype=torch.long)

    keep_index = torch.cat(keep_indices, dim=0).to(torch.long)

    if verbose:
        total_keep = int(keep_index.numel())
        total_tokens = int(t * tokens_per_frame)
        ratio = float(total_keep) / float(total_tokens) if total_tokens > 0 else 0.0
        curv_min = float(curvature.min().item()) if curvature.numel() > 0 else 0.0
        curv_mean = float(curvature.mean().item()) if curvature.numel() > 0 else 0.0
        curv_max = float(curvature.max().item()) if curvature.numel() > 0 else 0.0
        weight_min = float(weights.min().item()) if weights.numel() > 0 else 0.0
        weight_mean = float(weights.mean().item()) if weights.numel() > 0 else 0.0
        weight_max = float(weights.max().item()) if weights.numel() > 0 else 0.0
        tag_prefix = f"{tag} " if tag else ""
        print(
            f"[V-CAST] {tag_prefix}frames={t} tokens/frame={tokens_per_frame} retain_ratio={retain_ratio} "
            f"keep={total_keep}/{total_tokens} ({ratio:.4f})",
            flush=True,
        )
        print(
            f"[V-CAST] {tag_prefix}budget=curvature/softmax@{budget_temp:.4f} "
            f"score_mode={score_mode} min_k={min_k}",
            flush=True,
        )
        print(
            f"[V-CAST] {tag_prefix}curvature[min,mean,max]=[{curv_min:.4f},{curv_mean:.4f},{curv_max:.4f}]",
            flush=True,
        )
        print(
            f"[V-CAST] {tag_prefix}weight[min,mean,max]=[{weight_min:.4f},{weight_mean:.4f},{weight_max:.4f}]",
            flush=True,
        )
        print(f"[V-CAST] {tag_prefix}frame_keep={k_t.tolist()}", flush=True)

    return keep_index
