from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _full_output(video_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = torch.arange(video_embeds.shape[0], device=video_embeds.device, dtype=torch.long)
    return video_embeds, keep


def fastvid_compression(
    video_embeds: torch.Tensor,
    importance_scores: torch.Tensor,
    grid_thw: torch.Tensor,
    config_args: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FastVID compression adapted from the Qwen3_VL compressor reference.

    Returns compressed token features plus representative global token indices.
    The representative indices are used only for placeholder/RoPE alignment;
    context tokens carry DTM-merged features.
    """
    device = video_embeds.device
    if grid_thw is None or grid_thw.numel() == 0:
        return _full_output(video_embeds)
    if grid_thw.ndim != 2 or grid_thw.shape[0] != 1:
        return _full_output(video_embeds)

    t = int(grid_thw[0, 0].item())
    total_tokens, dim = video_embeds.shape
    if t <= 0 or total_tokens == 0 or total_tokens % t != 0:
        return _full_output(video_embeds)

    retention_ratio = float(config_args.get("retention_ratio", 0.27))
    dyseg_c = int(config_args.get("dyseg_c", 8))
    dyseg_tau = float(config_args.get("dyseg_tau", 0.84))
    dyseg_ignore = float(config_args.get("dyseg_ignore", 0.95))
    stprune_d = float(config_args.get("stprune_d", 0.4))
    dtm_p = int(config_args.get("dtm_p", 4))
    dtm_beta = float(config_args.get("dtm_beta", 0.6))

    retention_ratio = max(0.0, min(1.0, retention_ratio))
    dyseg_c = max(1, dyseg_c)
    dtm_p = max(1, dtm_p)
    stprune_d = max(0.0, min(1.0, stprune_d))
    dtm_beta = max(0.0, min(1.0, dtm_beta))

    tokens_per_frame = total_tokens // t
    video_reshaped = video_embeds.view(t, tokens_per_frame, dim)
    score_frames = importance_scores.view(t, tokens_per_frame)

    frame_global = F.normalize(video_reshaped.float().mean(dim=1), dim=-1)
    if t > 1:
        similarity_matrix = (frame_global[:-1] * frame_global[1:]).sum(dim=1)
        k_val = min(dyseg_c - 1, t - 2)
    else:
        similarity_matrix = torch.empty(0, device=device, dtype=torch.float32)
        k_val = 0

    if k_val > 0:
        cut_indices = torch.topk(similarity_matrix, k_val, largest=False).indices
        cos_indices = torch.nonzero(similarity_matrix < dyseg_tau, as_tuple=False).squeeze(1)
        cut_indices = torch.unique(torch.cat([cut_indices, cos_indices])).sort().values
        if cut_indices.numel() > 0:
            segment_sizes = [int(cut_indices[0].item()) + 1]
            for i in range(1, int(cut_indices.numel())):
                segment_sizes.append(int(cut_indices[i].item() - cut_indices[i - 1].item()))
            segment_sizes.append(int(t - cut_indices[-1].item() - 1))
        else:
            segment_sizes = [t]
    else:
        segment_sizes = [t]

    segments_embeds = torch.split(video_reshaped, segment_sizes)
    segments_scores = torch.split(score_frames, segment_sizes)
    segments_global = torch.split(frame_global, segment_sizes)

    final_tokens: List[torch.Tensor] = []
    keep_indices_list: List[torch.Tensor] = []
    current_frame_idx = 0
    frame_retain_num = max(1, int(tokens_per_frame * retention_ratio))

    for seg_i, seg_embeds in enumerate(segments_embeds):
        cur_scores = segments_scores[seg_i]
        cur_global = segments_global[seg_i]
        cur_seg_len = int(seg_embeds.shape[0])

        seg_retain_num = min(cur_seg_len * tokens_per_frame, max(1, frame_retain_num * cur_seg_len))
        seg_context_num = max(int(seg_retain_num * stprune_d), 1)
        seg_context_num = min(seg_context_num, seg_retain_num)
        seg_salient_num = seg_retain_num - seg_context_num

        if cur_seg_len == 1:
            frm_salient_num = [min(seg_salient_num, tokens_per_frame)]
            frm_context_num = [min(seg_context_num, max(1, tokens_per_frame // 2))]
        else:
            cur_sim_matrix = (cur_global[:-1] * cur_global[1:]).sum(dim=1)
            cur_cuts = torch.nonzero(cur_sim_matrix < dyseg_ignore, as_tuple=False).squeeze(1)
            if cur_cuts.numel() > 0:
                cur_sub_sizes = [int(cur_cuts[0].item()) + 1]
                for i in range(1, int(cur_cuts.numel())):
                    cur_sub_sizes.append(int(cur_cuts[i].item() - cur_cuts[i - 1].item()))
                cur_sub_sizes.append(int(cur_seg_len - cur_cuts[-1].item() - 1))
            else:
                cur_sub_sizes = [cur_seg_len]

            valid_seg_len = len(cur_sub_sizes)
            salient_chunk = seg_salient_num // valid_seg_len
            salient_rem = seg_salient_num % valid_seg_len
            frm_salient_alloc = [salient_chunk + (1 if i < salient_rem else 0) for i in range(valid_seg_len)]

            temp_num = (valid_seg_len + dtm_p - 1) // dtm_p
            ctx_chunk = seg_context_num // temp_num
            ctx_rem = seg_context_num % temp_num
            temp_context_alloc = [ctx_chunk + (1 if i < ctx_rem else 0) for i in range(temp_num)]

            frm_context_num = []
            tmp_ctx_idx = 0
            for i in range(valid_seg_len):
                if i % dtm_p == 0:
                    val = min(temp_context_alloc[tmp_ctx_idx], max(1, tokens_per_frame // 2))
                    frm_context_num.append(val)
                    tmp_ctx_idx += 1
                else:
                    frm_context_num.append(0)
            frm_context_num.reverse()

            frm_salient_num = []
            final_frm_context_num = []
            for i, size in enumerate(cur_sub_sizes):
                frm_salient_num.extend([0] * (int(size) - 1))
                final_frm_context_num.extend([0] * (int(size) - 1))
                ctx = min(frm_context_num[i], max(1, tokens_per_frame // 2))
                if ctx > 0:
                    final_frm_context_num.append(ctx)
                    frm_salient_num.append(min(frm_salient_alloc[i], tokens_per_frame - ctx))
                else:
                    final_frm_context_num.append(0)
                    frm_salient_num.append(min(frm_salient_alloc[i], tokens_per_frame))
            frm_context_num = final_frm_context_num

        cur_salient_indices: List[torch.Tensor] = []
        cur_context_indices: List[torch.Tensor] = []
        for f_i in range(cur_seg_len):
            frame_offset = f_i * tokens_per_frame
            top_k_indices = None
            if frm_salient_num[f_i] > 0:
                k_sal = min(int(frm_salient_num[f_i]), tokens_per_frame)
                top_k_indices = torch.topk(cur_scores[f_i], k=k_sal, largest=True, sorted=False).indices
                cur_salient_indices.append(top_k_indices + frame_offset)

            if frm_context_num[f_i] > 0:
                all_indices = torch.arange(tokens_per_frame, device=device)
                if top_k_indices is not None:
                    candidates = all_indices[~torch.isin(all_indices, top_k_indices)]
                else:
                    candidates = all_indices
                if candidates.numel() > 0:
                    tmp_feats = seg_embeds[f_i, candidates].unsqueeze(0)
                    _, n_candidates, hidden_dim = tmp_feats.shape
                    dist_matrix = torch.cdist(tmp_feats.float(), tmp_feats.float()) / (hidden_dim**0.5)
                    k_nn = min(4, n_candidates)
                    dist_nearest, _ = torch.topk(dist_matrix, k=k_nn, dim=-1, largest=False)
                    density = (-(dist_nearest**2).mean(dim=-1)).exp()
                    density_mask = (density[:, None, :] > density[:, :, None]).type(tmp_feats.dtype)
                    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
                    dist, _ = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)
                    score = dist * density
                    k_ctx = min(int(frm_context_num[f_i]), n_candidates)
                    _, sampled_local_indices = torch.topk(score, k=k_ctx, dim=-1)
                    ctx_indices = candidates[sampled_local_indices[0]]
                    cur_context_indices.append(ctx_indices + frame_offset)

        idx_salient = torch.cat(cur_salient_indices) if cur_salient_indices else torch.empty(0, device=device, dtype=torch.long)
        idx_context_raw = cur_context_indices
        flat_seg_hidden_states = seg_embeds.reshape(-1, dim).clone()

        if idx_context_raw:
            flat_seg_tokens = F.normalize(flat_seg_hidden_states.float(), dim=-1)
            cur_all_indices = torch.arange(cur_seg_len * tokens_per_frame, device=device)
            for ctx_idxs in idx_context_raw:
                if ctx_idxs.numel() == 0:
                    continue
                retain_idx = torch.cat([idx_salient, ctx_idxs]) if idx_salient.numel() > 0 else ctx_idxs
                retain_mask = torch.isin(cur_all_indices, retain_idx)
                merge_indices = cur_all_indices[~retain_mask]
                if merge_indices.numel() == 0:
                    continue

                target_tokens = flat_seg_tokens[ctx_idxs]
                to_merge_tokens = flat_seg_tokens[merge_indices]
                similarity = torch.mm(to_merge_tokens, target_tokens.T)
                assign_one_hot = torch.zeros(
                    to_merge_tokens.shape[0],
                    ctx_idxs.shape[0],
                    dtype=flat_seg_hidden_states.dtype,
                    device=device,
                )
                assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)
                assign_sum = assign_one_hot.sum(dim=0).unsqueeze(-1)
                avg_weights = (1 / (assign_sum + 1)).clamp(min=dtm_beta)
                counts = assign_sum.clamp(min=1)
                hidden_to_merge = flat_seg_hidden_states[merge_indices]
                aggregated_hidden = torch.mm(assign_one_hot.T, hidden_to_merge) / counts
                target_hidden = flat_seg_hidden_states[ctx_idxs]
                flat_seg_hidden_states[ctx_idxs] = avg_weights * target_hidden + (1 - avg_weights) * aggregated_hidden

        idx_context = torch.cat(idx_context_raw) if idx_context_raw else torch.empty(0, device=device, dtype=torch.long)
        seg_keep_indices = torch.cat([idx_salient, idx_context])
        if seg_keep_indices.numel() == 0:
            seg_keep_indices = torch.tensor([0], device=device, dtype=torch.long)
        seg_keep_indices = torch.unique(seg_keep_indices.to(torch.long), sorted=True)
        final_tokens.append(flat_seg_hidden_states[seg_keep_indices].to(dtype=video_embeds.dtype))
        keep_indices_list.append(seg_keep_indices + current_frame_idx * tokens_per_frame)
        current_frame_idx += cur_seg_len

    if not keep_indices_list:
        return _full_output(video_embeds)
    keep_index = torch.cat(keep_indices_list).to(torch.long)
    compressed = torch.cat(final_tokens, dim=0).to(dtype=video_embeds.dtype)
    return compressed, keep_index
