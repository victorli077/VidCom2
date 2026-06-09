
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb_vision


def _received_attention_scores(block, hidden_states: Tensor, cu_seqlens: Tensor, rotary_pos_emb: Tensor) -> Tensor:
    attn = block.attn
    seq_length = int(hidden_states.shape[0])
    normed_states = block.norm1(hidden_states)

    query_states = attn.q(normed_states).reshape(seq_length, attn.num_heads, -1)
    key_states = attn.k(normed_states).reshape(seq_length, attn.num_heads, -1)
    query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
    key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    scores = []
    offset = 0
    scale = float(getattr(attn, "scaling", attn.head_dim**-0.5))
    for length in lengths:
        length = int(length)
        if length <= 0:
            continue
        q = query_states[:, :, offset : offset + length, :]
        k = key_states[:, :, offset : offset + length, :]
        weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        # Column sum: how much attention each token receives, averaged over heads.
        token_scores = weights[0].mean(dim=0).sum(dim=0)
        scores.append(token_scores)
        offset += length

    if not scores:
        return hidden_states.new_ones((seq_length,), dtype=torch.float32)
    return torch.cat(scores, dim=0).to(device=hidden_states.device, dtype=torch.float32)


def _normalize_scores(scores: Tensor) -> Tensor:
    scores = scores.float()
    min_val = scores.min()
    max_val = scores.max()
    if bool((max_val - min_val) > 1e-8):
        return (scores - min_val) / (max_val - min_val)
    return torch.ones_like(scores)


def _merge_pre_spatial_scores(scores: Tensor, visual_model, grid_thw: Tensor, window_index: Tensor) -> Tensor:
    merge_unit = int(visual_model.spatial_merge_unit)
    if merge_unit <= 1:
        merged = scores
    else:
        merged = scores.reshape(-1, merge_unit).mean(dim=1)
    reverse_indices = torch.argsort(window_index)
    return merged[reverse_indices]


def get_video_features_with_attention_scores(
    visual_model,
    pixel_values_videos: Tensor,
    video_grid_thw: Tensor,
) -> tuple[Tensor, Optional[Tensor]]:
    """Run Qwen2.5-Omni vision encoder and return merged features plus attention scores.

    The returned attention scores are aligned with merged video tokens, i.e. the
    same order and length as `visual_model(pixel_values_videos, video_grid_thw)`.
    If extraction fails, callers should fall back to feature-norm scores.
    """

    hidden_states = visual_model.patch_embed(pixel_values_videos)
    rotary_pos_emb = visual_model.rot_pos_emb(video_grid_thw)

    window_index, cu_window_seqlens = visual_model.get_window_index(video_grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=video_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // visual_model.spatial_merge_unit, visual_model.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual_model.spatial_merge_unit, visual_model.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    cu_seqlens = torch.repeat_interleave(video_grid_thw[:, 1] * video_grid_thw[:, 2], video_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=video_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    attention_scores = None
    last_layer = len(visual_model.blocks) - 1
    for layer_num, blk in enumerate(visual_model.blocks):
        if layer_num in visual_model.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        if layer_num == last_layer:
            raw_scores = _received_attention_scores(blk, hidden_states, cu_seqlens_now, rotary_pos_emb)
            attention_scores = _normalize_scores(raw_scores)

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            rotary_pos_emb=rotary_pos_emb,
        )

    hidden_states = visual_model.merger(hidden_states)
    attention_scores = _merge_pre_spatial_scores(attention_scores, visual_model, video_grid_thw, window_index)
    return hidden_states, attention_scores.to(device=hidden_states.device, dtype=torch.float32)
