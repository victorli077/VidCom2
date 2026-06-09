from __future__ import annotations

import os
import torch

from .fastvid_algo import fastvid_compression


def compress_video_features(
    flat_features: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    base_scale: float,
    attention_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = [int(x) for x in grid_thw.tolist()]
    frame_tokens = (h * w) // (spatial_merge_size ** 2)
    total = int(flat_features.shape[0])
    if t <= 0 or frame_tokens <= 0 or total == 0 or total != t * frame_tokens:
        return flat_features, torch.arange(total, device=flat_features.device, dtype=torch.long)

    if attention_scores is None:
        importance_scores = flat_features.float().norm(dim=-1)
    else:
        importance_scores = attention_scores.to(device=flat_features.device, dtype=torch.float32).flatten()
        if importance_scores.numel() != total:
            importance_scores = flat_features.float().norm(dim=-1)
    cfg = {
        "retention_ratio": float(os.getenv("fastvid_retention_ratio", str(base_scale))),
        "dyseg_c": int(os.getenv("fastvid_DySeg_c", "8")),
        "dyseg_tau": float(os.getenv("fastvid_DySeg_tau", "0.84")),
        "dyseg_ignore": float(os.getenv("fastvid_DySeg_ignore", "0.95")),
        "stprune_d": float(os.getenv("fastvid_STPrune_d", "0.4")),
        "dtm_p": int(os.getenv("fastvid_DTM_p", "4")),
        "dtm_beta": float(os.getenv("fastvid_DTM_beta", "0.6")),
    }
    grid_batch = grid_thw.view(1, 3).to(device=flat_features.device)
    compressed, keep = fastvid_compression(
        video_embeds=flat_features,
        importance_scores=importance_scores,
        grid_thw=grid_batch,
        config_args=cfg,
    )
    keep = keep.to(dtype=torch.long, device=flat_features.device)
    keep = keep[(keep >= 0) & (keep < total)]
    compressed = compressed.to(device=flat_features.device, dtype=flat_features.dtype)
    if compressed.shape[0] != keep.numel():
        n = min(int(compressed.shape[0]), int(keep.numel()))
        compressed = compressed[:n]
        keep = keep[:n]
    if keep.numel() == 0:
        keep = torch.tensor([0], device=flat_features.device, dtype=torch.long)
        compressed = flat_features[:1]
    order = torch.argsort(keep)
    keep = keep[order]
    compressed = compressed[order]
    if keep.numel() > 1:
        unique_mask = torch.ones_like(keep, dtype=torch.bool)
        unique_mask[1:] = keep[1:] != keep[:-1]
        keep = keep[unique_mask]
        compressed = compressed[unique_mask]
    return compressed, keep


def compute_keep_indices(
    flat_features: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    base_scale: float,
) -> torch.Tensor:
    _, keep = compress_video_features(flat_features, grid_thw, spatial_merge_size, base_scale)
    return keep
