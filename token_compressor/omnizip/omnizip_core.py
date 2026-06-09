from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class OmniZipOutput:
    video_keep_indices: torch.Tensor
    kept_video: torch.Tensor
    audio_keep_indices: torch.Tensor
    kept_audio: torch.Tensor
    stats: Dict[str, float]


@dataclass
class OmniZipSequenceOutput:
    input_embeds: torch.Tensor
    keep_mask: torch.Tensor
    stats: Dict[str, float]


def _to_seq(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.mean(0) if x.dim() == 3 else x


def _prepare_attn_logits(audio_feature: torch.Tensor, attn_logits: Optional[torch.Tensor]) -> torch.Tensor:
    n = int(audio_feature.shape[0])
    if attn_logits is not None:
        logits = attn_logits.reshape(-1).to(audio_feature.device)
        if logits.numel() == n:
            return logits
        if logits.numel() > n:
            return logits[:n]
        if logits.numel() > 0:
            reps = (n + logits.numel() - 1) // logits.numel()
            return logits.repeat(reps)[:n]
    # Fallback is kept only for robustness. Faithful OmniZip runs should pass
    # the audio-attention importance emitted by the patched audio encoder.
    return audio_feature.norm(dim=-1)


def omnizip_audio_attn(
    audio_feature: torch.Tensor,
    video_feature: Optional[torch.Tensor],
    attn_logits: torch.Tensor,
    merging_ratio: float = 0.5,
    contextual_ratio: float = 0.03,
    g: int = 3,
    rho_audio: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    """Official OmniZip audio-token selection and contextual anchor plan.

    This mirrors KD-TAO/OmniZip's `omnizip_audio_attn`. `rho_audio` is accepted
    as a backward-compatible alias for the official `merging_ratio`.
    """
    if rho_audio is not None:
        merging_ratio = rho_audio
    device = attn_logits.device
    a = _to_seq(audio_feature).to(device)
    v = _to_seq(video_feature).to(device) if video_feature is not None else None
    n = a.size(0)
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=device), {}

    keep_mask = torch.zeros(n, dtype=torch.bool, device=device)
    dominant_num = int(max(0, min(n, round((1.0 - merging_ratio) * n))))
    if dominant_num > 0:
        _, topk = torch.topk(attn_logits.reshape(-1).to(device), dominant_num)
        keep_mask[topk] = True

    all_idx = torch.arange(n, device=device)
    remaining = all_idx[~keep_mask]
    contextual_num = int(max(0, round(contextual_ratio * n)))
    merge_plan: Dict[int, List[int]] = {}

    if remaining.numel() > 0 and contextual_num > 0 and g > 0:
        contextual_num = min(contextual_num, remaining.numel())
        step = max(1, remaining.numel() // contextual_num)
        init_pos = torch.arange(0, remaining.numel(), step, device=device)[:contextual_num]
        anchors = remaining[init_pos]
        keep_mask[anchors] = True

        rem_local_mask = torch.ones(remaining.numel(), dtype=torch.bool, device=device)
        rem_local_mask[init_pos] = False
        pool_global = remaining[torch.arange(remaining.numel(), device=device)[rem_local_mask]]

        if pool_global.numel() > 0 and g > 0:
            a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
            sim_aa = a_norm[pool_global] @ a_norm[anchors].T
            assign = sim_aa.argmax(dim=1)

            if v is not None and v.numel() > 0:
                v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
                sim_av = a_norm[pool_global] @ v_norm.T
                scores = sim_av.max(dim=1).values
            else:
                scores = sim_aa.max(dim=1).values

            for c in range(anchors.numel()):
                mask_c = assign == c
                cand = pool_global[mask_c]
                if cand.numel() == 0:
                    merge_plan[int(anchors[c].item())] = []
                    continue
                scores_c = scores[mask_c]
                topg = min(g, cand.numel())
                _, sel = torch.topk(scores_c, topg, largest=True)
                merge_plan[int(anchors[c].item())] = cand[sel].tolist()

    return keep_mask, merge_plan


def omnizip_istm(
    video_feature: torch.Tensor,
    num_tokens_per_frame: int,
    merging_ratio: float = 0.5,
    rho_video: Optional[float] = None,
) -> torch.Tensor:
    """Official OmniZip ISTM/ISTC video selection mask."""
    if rho_video is not None:
        merging_ratio = rho_video
    if num_tokens_per_frame <= 0 or video_feature.numel() == 0:
        return torch.ones(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    if num_frames <= 0:
        return torch.ones(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)

    def dpcknn(tokens: torch.Tensor, keep_rate: float = 0.5, k: int = 5) -> torch.Tensor:
        n = tokens.shape[0]
        num_keep = int(n * keep_rate)
        if num_keep <= 0:
            return torch.empty(0, dtype=torch.long, device=tokens.device)
        if num_keep >= n:
            return torch.arange(n, device=tokens.device)
        k_eff = min(k, n - 1)
        with torch.no_grad():
            normed = torch.nn.functional.normalize(tokens, dim=1)
            dist = 1 - torch.mm(normed, normed.T)
            dist.fill_diagonal_(float("inf"))
            knn_dist, _ = torch.topk(dist, k_eff, dim=1, largest=False)
            rho = torch.exp(-knn_dist.mean(dim=1))
            higher_mask = rho.unsqueeze(0) > rho.unsqueeze(1)
            delta_dist = dist.clone()
            delta_dist[~higher_mask] = float("inf")
            delta = delta_dist.min(dim=1).values
            is_peak = delta.isinf()
            if is_peak.any():
                finite_dist = dist.clone()
                finite_dist.fill_diagonal_(0)
                delta[is_peak] = finite_dist[is_peak].max(dim=1).values
            score = delta * rho
            selected = torch.topk(score, num_keep).indices
        return selected

    keep_ratio = 1.0 - merging_ratio
    for t in range(num_frames):
        start_idx = t * num_tokens_per_frame
        end_idx = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start_idx:end_idx]

        if t % 2 == 0:
            num_keep = int(num_tokens_per_frame * keep_ratio)
            if num_keep < num_tokens_per_frame:
                keep_idx = dpcknn(tokens, keep_rate=keep_ratio, k=5)
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx + keep_idx] = True
        else:
            prev_tokens = video_feature[(t - 1) * num_tokens_per_frame : t * num_tokens_per_frame]
            prev_norm = torch.nn.functional.normalize(prev_tokens, p=2, dim=1)
            curr_norm = torch.nn.functional.normalize(tokens, p=2, dim=1)
            similarity = torch.nn.functional.cosine_similarity(curr_norm, prev_norm, dim=1)
            num_keep = int(num_tokens_per_frame * keep_ratio)
            if num_keep < num_tokens_per_frame:
                keep_idx = similarity.topk(num_keep, largest=False).indices
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx + keep_idx] = True

    tail_start = num_frames * num_tokens_per_frame
    if tail_start < video_feature.shape[0]:
        mask[tail_start:] = True
    return mask


def _detect_token_chunks(indices: torch.Tensor) -> List[Tuple[int, int]]:
    if indices.numel() == 0:
        return []
    diffs = indices[1:] - indices[:-1]
    boundaries = (diffs > 1).nonzero(as_tuple=True)[0] + 1
    chunks: List[Tuple[int, int]] = []
    prev = 0
    for boundary in boundaries.cpu().tolist():
        chunks.append((prev, boundary))
        prev = boundary
    chunks.append((prev, indices.numel()))
    return chunks


def _merge_audio_in_place(
    flat_embeds: torch.Tensor,
    audio_feature: torch.Tensor,
    video_feature: torch.Tensor,
    audio_indices: torch.Tensor,
    merge_plan: Dict[int, List[int]],
    g: int,
) -> None:
    if g <= 0 or not merge_plan:
        return
    a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
    v_norm = video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6) if video_feature.numel() > 0 else None
    for anchor_rel_idx, merge_rel_list in merge_plan.items():
        if not merge_rel_list:
            continue
        merge_rel = torch.tensor(merge_rel_list, device=flat_embeds.device, dtype=torch.long)
        if v_norm is not None:
            scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
        else:
            scores = (a_norm[merge_rel] @ a_norm[anchor_rel_idx].unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        anchor_vec = audio_feature[anchor_rel_idx]
        merged_vec = (audio_feature[merge_rel] * weights.unsqueeze(-1)).sum(dim=0)
        new_anchor = (anchor_vec + merged_vec) / (1.0 + weights.sum())
        flat_embeds[audio_indices[anchor_rel_idx]] = new_anchor


def omnizip_sequence(
    input_embeds: torch.Tensor,
    attn_logits: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    video_grid_thw: torch.Tensor,
    merging_ratio_audio: float = 0.5,
    merging_ratio_v: float = 0.5,
    contextual_ratio: float = 0.05,
    g: int = 3,
    audio_preserve: bool = False,
    rho_min: float = 0.35,
    rho_max: float = 0.75,
) -> OmniZipSequenceOutput:
    """Sequence-level OmniZip matching the official implementation.

    Returns updated embeddings and a global keep mask over the input sequence.
    `audio_preserve=True` is our controlled ablation: it keeps all audio tokens
    and skips contextual audio merging, while keeping the official audio-guided
    video budget path.
    """
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        bsz, seq_len, dim = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, dim).clone()
        flat_ids = input_ids.reshape(-1)
    else:
        seq_len, dim = input_embeds.shape
        flat_embeds = input_embeds.clone()
        flat_ids = input_ids

    video_token_mask = (flat_ids == video_token_id).to(device)
    audio_token_mask = (flat_ids == audio_token_id).to(device)
    video_indices = torch.nonzero(video_token_mask, as_tuple=True)[0]
    audio_indices = torch.nonzero(audio_token_mask, as_tuple=True)[0]
    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]

    if video_feature.numel() == 0:
        keep_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
        output = flat_embeds.reshape(bsz, seq_len, dim) if is_batched else flat_embeds
        return OmniZipSequenceOutput(output, keep_mask, {})

    t = int(video_grid_thw[0, 0].item())
    w = int(video_grid_thw[0, 1].item()) // 2
    h = int(video_grid_thw[0, 2].item()) // 2
    video_token_per_frame = max(1, h * w)
    num_video_tokens = int(t * video_token_per_frame)

    logits = _prepare_attn_logits(audio_feature, attn_logits) if audio_feature.numel() > 0 else None
    if audio_feature.numel() > 0 and logits is not None:
        audio_mask, merge_plan = omnizip_audio_attn(
            audio_feature=audio_feature,
            video_feature=video_feature,
            attn_logits=logits,
            merging_ratio=merging_ratio_audio,
            contextual_ratio=contextual_ratio,
            g=g,
        )
    else:
        audio_mask = torch.ones(audio_feature.shape[0], dtype=torch.bool, device=device)
        merge_plan = {}

    guidance_audio_mask = audio_mask
    if audio_preserve:
        audio_mask = torch.ones_like(audio_mask, dtype=torch.bool)
    else:
        _merge_audio_in_place(flat_embeds, audio_feature, video_feature, audio_indices, merge_plan, g)

    video_chunks = _detect_token_chunks(video_indices)
    audio_chunks = _detect_token_chunks(audio_indices)
    num_guided = min(len(video_chunks), len(audio_chunks))

    audio_group_retention: List[float] = []
    for i in range(num_guided):
        a_start, a_end = audio_chunks[i]
        group_mask = guidance_audio_mask[a_start:a_end]
        ratio = group_mask.float().mean().item() if group_mask.numel() > 0 else 0.0
        audio_group_retention.append(ratio)

    base_vs_guided = [
        max(rho_min, min(rho_max, rho_max + (rho_min - rho_max) * ret))
        for ret in audio_group_retention
    ]

    guided_video_tokens = sum(video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided))
    unguided_video_tokens = sum(video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided, len(video_chunks)))
    target_discard_tokens = merging_ratio_v * num_video_tokens
    unguided_discard = merging_ratio_v * unguided_video_tokens
    guided_target_discard = target_discard_tokens - unguided_discard

    adjusted_vs_guided = base_vs_guided.copy()
    if num_guided > 2 and guided_video_tokens > 0:
        guided_group_sizes = [video_chunks[i][1] - video_chunks[i][0] for i in range(num_guided)]
        base_guided_discard = sum(base_vs_guided[i] * guided_group_sizes[i] for i in range(num_guided))
        if abs(base_guided_discard - guided_target_discard) > 1e-3 and base_guided_discard > 0:
            min_idx = base_vs_guided.index(min(base_vs_guided))
            max_idx = base_vs_guided.index(max(base_vs_guided))
            min_val, max_val = base_vs_guided[min_idx], base_vs_guided[max_idx]
            other_indices = [i for i in range(num_guided) if i != min_idx and i != max_idx]
            other_discard = sum(base_vs_guided[i] * guided_group_sizes[i] for i in other_indices)
            if other_indices and other_discard > 0:
                fixed_discard = min_val * guided_group_sizes[min_idx] + max_val * guided_group_sizes[max_idx]
                remain_target = guided_target_discard - fixed_discard
                scaling = remain_target / other_discard
                for i in other_indices:
                    adjusted_vs_guided[i] = max(rho_min, min(rho_max, base_vs_guided[i] * scaling))
                actual_discard = sum(adjusted_vs_guided[i] * guided_group_sizes[i] for i in range(num_guided))
                diff = guided_target_discard - actual_discard
                if abs(diff) > 1e-3:
                    total_other_size = sum(guided_group_sizes[i] for i in other_indices)
                    if total_other_size > 0:
                        adj_per_token = diff / total_other_size
                        for i in other_indices:
                            adjusted_vs_guided[i] = max(rho_min, min(rho_max, adjusted_vs_guided[i] + adj_per_token))

    video_group_masks: List[torch.Tensor] = []
    for i, (v_start, v_end) in enumerate(video_chunks):
        group_feat = video_feature[v_start:v_end]
        num_frames_in_group = group_feat.shape[0] // video_token_per_frame
        if num_frames_in_group == 0:
            video_group_masks.append(torch.ones(group_feat.shape[0], dtype=torch.bool, device=device))
            continue
        ratio = adjusted_vs_guided[i] if i < num_guided else merging_ratio_v
        group_mask = omnizip_istm(
            group_feat,
            num_tokens_per_frame=video_token_per_frame,
            merging_ratio=ratio,
        )
        video_group_masks.append(group_mask)

    video_mask = torch.cat(video_group_masks, dim=0) if video_group_masks else torch.ones(0, dtype=torch.bool, device=device)
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    if video_mask.shape[0] != video_indices.shape[0]:
        raise RuntimeError(f"OmniZip video mask mismatch: {video_mask.shape[0]} vs {video_indices.shape[0]}")
    if audio_mask.shape[0] != audio_indices.shape[0]:
        raise RuntimeError(f"OmniZip audio mask mismatch: {audio_mask.shape[0]} vs {audio_indices.shape[0]}")
    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask

    stats = {
        "pre_audio_tokens": float(audio_indices.numel()),
        "post_audio_tokens": float(audio_mask.sum().item()),
        "audio_retention_ratio": float(audio_mask.float().mean().item()) if audio_mask.numel() > 0 else 0.0,
        "audio_compression_ratio": 1.0 - float(audio_mask.float().mean().item()) if audio_mask.numel() > 0 else 0.0,
        "pre_video_tokens": float(video_indices.numel()),
        "post_video_tokens": float(video_mask.sum().item()),
        "video_retention_ratio": float(video_mask.float().mean().item()) if video_mask.numel() > 0 else 0.0,
    }
    output_embeds = flat_embeds.reshape(bsz, seq_len, dim) if is_batched else flat_embeds
    return OmniZipSequenceOutput(output_embeds, global_mask, stats)


def omnizip(
    audio_feature: Optional[torch.Tensor],
    video_feature: torch.Tensor,
    frame_tokens: int,
    audio_attn_logits: Optional[torch.Tensor] = None,
    rho_audio: float = 0.3,
    rho_video: float = 0.65,
    contextual_ratio: float = 0.05,
    g: int = 3,
    audio_preserve: bool = False,
    rho_min: float = 0.35,
    rho_max: float = 0.75,
) -> OmniZipOutput:
    """Backward-compatible feature-level wrapper around official primitives."""
    device = video_feature.device
    total_video = int(video_feature.shape[0])
    logits = _prepare_attn_logits(audio_feature, audio_attn_logits) if audio_feature is not None and audio_feature.numel() else None
    if audio_feature is not None and audio_feature.numel() and logits is not None:
        audio_mask, merge_plan = omnizip_audio_attn(
            audio_feature,
            video_feature,
            logits,
            merging_ratio=rho_audio,
            contextual_ratio=contextual_ratio,
            g=g,
        )
        audio_keep_indices = torch.arange(audio_feature.shape[0], device=device) if audio_preserve else torch.nonzero(audio_mask, as_tuple=True)[0]
        kept_audio = audio_feature[audio_keep_indices]
    else:
        audio_keep_indices = torch.empty(0, dtype=torch.long, device=device)
        kept_audio = torch.empty(0, video_feature.shape[-1], dtype=video_feature.dtype, device=device)
        audio_mask = torch.empty(0, dtype=torch.bool, device=device)
    video_mask = omnizip_istm(video_feature, frame_tokens, merging_ratio=rho_video)
    video_keep_indices = torch.nonzero(video_mask, as_tuple=True)[0]
    pre_audio = float(0 if audio_feature is None else audio_feature.shape[0])
    post_audio = float(audio_keep_indices.numel())
    stats = {
        "pre_audio_tokens": pre_audio,
        "post_audio_tokens": post_audio,
        "audio_retention_ratio": post_audio / pre_audio if pre_audio > 0 else 0.0,
        "audio_compression_ratio": 1.0 - (post_audio / pre_audio) if pre_audio > 0 else 0.0,
        "pre_video_tokens": float(total_video),
        "post_video_tokens": float(video_keep_indices.numel()),
        "video_retention_ratio": float(video_keep_indices.numel()) / max(float(total_video), 1.0),
    }
    return OmniZipOutput(video_keep_indices, video_feature[video_keep_indices], audio_keep_indices, kept_audio, stats)
