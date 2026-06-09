from __future__ import annotations

import os
import torch

from .visionzip import visionzip_compression


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

    dominant_ratio = float(os.getenv("visionzip_dominant_ratio", "0.6"))
    k_neighbors = int(os.getenv("visionzip_k_neighbors", "5"))
    retention_ratio = float(max(0.0, min(1.0, base_scale)))

    if attention_scores is None:
        attention_scores = flat_features.norm(dim=-1)
    else:
        attention_scores = attention_scores.to(device=flat_features.device, dtype=torch.float32).flatten()
        if attention_scores.numel() != total:
            attention_scores = flat_features.norm(dim=-1)
    compressed, local_keep = visionzip_compression(
        features=flat_features.unsqueeze(0),
        attention_scores=attention_scores.unsqueeze(0),
        retention_ratio=retention_ratio,
        dominant_ratio=dominant_ratio,
        cls_token=False,
        k_neighbors=max(1, k_neighbors),
    )

    keep = local_keep[0].to(dtype=torch.long, device=flat_features.device)
    compressed_features = compressed[0].to(device=flat_features.device, dtype=flat_features.dtype)
    keep = keep[(keep >= 0) & (keep < total)]
    if keep.numel() != compressed_features.shape[0]:
        n = min(int(keep.numel()), int(compressed_features.shape[0]))
        keep = keep[:n]
        compressed_features = compressed_features[:n]
    if keep.numel() == 0:
        keep = torch.tensor([0], device=flat_features.device, dtype=torch.long)
        compressed_features = flat_features[:1]
    order = torch.argsort(keep)
    keep = keep[order]
    compressed_features = compressed_features[order]
    return compressed_features, keep


def compute_keep_indices(
    flat_features: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    base_scale: float,
) -> torch.Tensor:
    _, keep = compress_video_features(flat_features, grid_thw, spatial_merge_size, base_scale)
    return keep
