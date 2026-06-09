from typing import Optional, Union, List
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)

from token_compressor.vidcom2 import (
    select_low_var_channels,
    compute_gaussian_scores,
    compute_scales,
    select_outlier_indices,
    get_audio_guided_frame_features,
    compute_audio_change_scores,
    _map_linear_offset,
)
from token_compressor.vidcom2.visualization import save_budget_comparison_artifact

_VIDCOM_TOKEN_STATS_ENV = "VIDCOM_TOKEN_STATS"
_VIDCOM_TOKEN_STATS_CASE_ENV = "VIDCOM_TOKEN_STATS_CASE"
_VIDCOM_AUDIO_FPS_ENV = "VIDCOM2_AUDIO_FPS"
_VIDCOM_AUDIO_WEIGHT_TOKEN_ENV = "VIDCOM2_AUDIO_WEIGHT_TOKEN"
_VIDCOM_AUDIO_GUIDANCE_ENV = "VIDCOM2_ENABLE_AUDIO_GUIDANCE"
_VIDCOM_AUDIO_TOKEN_GUIDANCE_ENV = "VIDCOM2_ENABLE_AUDIO_TOKEN_GUIDANCE"
_VIDCOM_BUDGET_VIZ_ENV = "VIDCOM2_DUMP_BUDGET_VIZ"
_VIDCOM_BUDGET_VIZ_DIR_ENV = "VIDCOM2_BUDGET_VIZ_DIR"
_VIDCOM_BUDGET_VIZ_CASES_ENV = "VIDCOM_BUDGET_VIZ_CASES"
_VIDCOM_BUDGET_VIZ_MAX_CASES_ENV = "VIDCOM2_BUDGET_VIZ_MAX_CASES"
_VIDCOM_BUDGET_VIZ_RENDER_NOW_ENV = "VIDCOM2_BUDGET_VIZ_RENDER_NOW"
_VIDCOM_BUDGET_VIZ_STOP_AFTER_TARGET_ENV = "VIDCOM2_BUDGET_VIZ_STOP_AFTER_TARGET"
_AUDIO_ROBUST_CLIP = 2.0


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _parse_case_selector(raw: Optional[str]) -> Optional[set]:
    """
    Parse case selector from env.
    Returns:
      - None: select all
      - set[int]: selected case indices
    Examples:
      "all", "*" -> all
      "0,3,5-8"  -> {0,3,5,6,7,8}
    """
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in {"", "all", "*"}:
        return None

    out = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(p))
    return out


def _minmax_normalize(x: Tensor, dim: Optional[int] = None, eps: float = 1e-6) -> Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    return ((x - x_min) / (x_max - x_min + eps)).clamp(0.0, 1.0)


def _fuse_video_audio_scores(
    u_video: Tensor,
    u_audio: Tensor,
    audio_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Fuse frame-wise u_video and preprocessed u_audio while keeping u_video unchanged.

    Requirements:
    1) Positive scale mapping from audio to video branch (monotonic in u_audio).
    2) Mapped audio branch has the same mean as u_video.
    3) Larger u_audio should contribute more positively to fused score.

    We use affine mapping:
      audio_mapped = a * audio_norm + b
      a = std(u_video) / (std(audio_norm) + eps)   (a >= 0)
      b = mean(u_video) - a * mean(audio_norm)
    so mean(audio_mapped) == mean(u_video), and a >= 0 keeps monotonicity.
    """
    # Convert to float and sanitize non-finite values for robustness.
    video_raw = torch.nan_to_num(u_video.float(), nan=0.0, posinf=0.0, neginf=0.0)
    audio_norm = torch.nan_to_num(u_audio.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    video_mean = torch.nan_to_num(video_raw.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    audio_mean = torch.nan_to_num(audio_norm.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    video_std = torch.nan_to_num(video_raw.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0)
    audio_std = torch.nan_to_num(audio_norm.std(unbiased=False), nan=0.0, posinf=0.0, neginf=0.0)
    audio_std = audio_std.clamp_min(0.05) 

    a = torch.abs(video_std) / (torch.abs(audio_std) + eps)
    b = video_mean - a * audio_mean
    audio_mapped = a * audio_norm + b

    weight = max(float(audio_weight), 0.0)
    return video_raw + weight * audio_mapped


def _robust_zscore(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Median/MAD based robust z-score, less sensitive to extreme spikes."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    med = x.median()
    mad = (x - med).abs().median()
    denom = 1.4826 * mad + eps
    return (x - med) / denom


def _prepare_audio_novelty_scores(raw_novelty: Tensor, clip_c: float = 3.0, eps: float = 1e-6) -> Tensor:
    """
    u_audio = sigmoid(clip(robust_zscore(raw_novelty), -c, c))
    No temporal smoothing here by design.
    """
    z = _robust_zscore(raw_novelty, eps=eps)
    z = z.clamp(min=-float(clip_c), max=float(clip_c))
    return torch.sigmoid(z)


def _fuse_visual_audio_token_scores(
    visual_token_scores: Tensor,
    token_features: Tensor,
    frame_audio_features: Tensor,
    u_audio: Tensor,
    selected_channels: Tensor,
    audio_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """
    Visual-dominant token scoring with audio residual guidance.
    - Keep visual token scores unchanged as the main branch.
    - Build audio token residual by token-audio cosine alignment.
    - Apply frame-wise audio gate from min-max normalized u_audio.
    - Use the same min-max + mean-scale matching logic before fusion.

    Audio features are sliced to the same dimensional subspace used for
    video token selection (via selected_channels), ensuring audio-video
    alignment in a semantically consistent subspace.
    """
    visual_token_scores = torch.nan_to_num(visual_token_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    token_features = torch.nan_to_num(token_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    frame_audio_features = torch.nan_to_num(frame_audio_features.float(), nan=0.0, posinf=0.0, neginf=0.0)

    # Align audio feature dimension to the subspace selected for video tokens.
    aligned_audio = frame_audio_features[:, selected_channels]

    token_unit = F.normalize(token_features, p=2, dim=-1, eps=eps)
    audio_unit = F.normalize(aligned_audio, p=2, dim=-1, eps=eps)
    audio_raw = F.cosine_similarity(token_unit, audio_unit.unsqueeze(1), dim=-1, eps=eps)  # [T, tpf]

    # Min-max normalize within each frame, then match mean scale to visual branch.
    audio_norm = _minmax_normalize(audio_raw, dim=-1, eps=eps)
    visual_mean = visual_token_scores.mean(dim=-1, keepdim=True)
    audio_mean = audio_norm.mean(dim=-1, keepdim=True)
    safe_scale = torch.where(
        torch.abs(audio_mean) < eps,
        torch.zeros_like(visual_mean),
        visual_mean / (audio_mean + eps),
    )
    audio_scaled = audio_norm * safe_scale

    # Frame-level gate: only frames with stronger audio change get stronger token guidance.
    frame_gate = _minmax_normalize(u_audio, eps=eps).unsqueeze(-1)
    return visual_token_scores + audio_weight * frame_gate * audio_scaled


def _resolve_second_per_grid(video_second_per_grid, idx: int) -> Optional[float]:
    if video_second_per_grid is None:
        return None
    if torch.is_tensor(video_second_per_grid):
        flat = video_second_per_grid.reshape(-1)
        if flat.numel() == 0:
            return None
        if flat.numel() == 1:
            return float(flat[0].item())
        return float(flat[min(idx, flat.numel() - 1)].item())
    if isinstance(video_second_per_grid, (list, tuple)):
        if len(video_second_per_grid) == 0:
            return None
        if len(video_second_per_grid) == 1:
            return float(video_second_per_grid[0])
        return float(video_second_per_grid[min(idx, len(video_second_per_grid) - 1)])
    return float(video_second_per_grid)


def _compute_budget_counts(scores: Tensor, scales: Tensor, tpf: int) -> List[int]:
    """Compute integer kept-token budget k_t per frame from scales and tpf."""
    _ = scores  # kept for API symmetry / future use.
    ks = (scales * tpf).round().long().clamp(min=1)
    return [int(v.item()) for v in ks]


def _compute_keep_indices(
    flat_features: Tensor,
    grid_thw: Tensor,
    spatial_merge_size: int,
    base_scale: float,
    enable_audio_guidance: bool = True,
    enable_audio_token_guidance: bool = True,
    audio_embeds: Optional[Tensor] = None,
    video_second_per_grid: Optional[float] = None,
    audio_fps: float = 50.0,
    audio_weight_frame: float = 1.0,
    audio_weight_token: float = 1.0,
    audio_robust_clip: float = _AUDIO_ROBUST_CLIP,
) -> Tensor:
    """Runs VidCom2 scoring to obtain kept token indices for a single video."""
    t, h, w = grid_thw.tolist()
    frame_tokens = (h * w) // (spatial_merge_size**2)
    if frame_tokens <= 0 or flat_features.numel() == 0:
        return torch.arange(flat_features.shape[0], device=flat_features.device)

    sel_feat, selected_channels = select_low_var_channels(flat_features)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
    u_video = -vid_score.mean(dim=-1)
    fused_frame_scores = u_video
    visual_token_scores = vid_score + frame_score
    fused_token_scores = visual_token_scores

    if (
        enable_audio_guidance
        and audio_embeds is not None
        and audio_embeds.ndim == 2
        and video_second_per_grid is not None
        and t > 0
    ):
        frame_audio_features = get_audio_guided_frame_features(
            video_grid_thw=grid_thw,
            video_second_per_grid=video_second_per_grid,
            audio_embeds=audio_embeds,
            AUDIO_FPS=audio_fps,
        )
        if frame_audio_features.shape[0] == fused_frame_scores.shape[0]:
            raw_audio_novelty = compute_audio_change_scores(frame_audio_features)
            u_audio = _prepare_audio_novelty_scores(
                raw_novelty=raw_audio_novelty,
                clip_c=audio_robust_clip,
            )
            fused_frame_scores = _fuse_video_audio_scores(
                u_video=u_video,
                u_audio=u_audio,
                audio_weight=audio_weight_frame,
            )
            # Optional switch: audio can guide budget only (frame scales), while token ranking stays visual-only.
            if enable_audio_token_guidance:
                token_features = sel_feat.view(-1, frame_tokens, sel_feat.shape[-1])
                fused_token_scores = _fuse_visual_audio_token_scores(
                    visual_token_scores=visual_token_scores,
                    token_features=token_features,
                    frame_audio_features=frame_audio_features,
                    u_audio=u_audio,
                    audio_weight=audio_weight_token,
                    selected_channels=selected_channels,
                )

    scales = compute_scales(fused_frame_scores, base_scale)
    indices = select_outlier_indices(fused_token_scores, scales, frame_tokens)
    return _map_linear_offset(indices, frame_tokens)


def _count_tokens(input_ids: Optional[torch.LongTensor], attention_mask: Optional[torch.Tensor], config) -> dict:
    if input_ids is None:
        return {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0}
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    mask = attention_mask.bool()
    total_tokens = mask.sum(dim=1)
    video_id = getattr(config, "video_token_id", None)
    audio_id = getattr(config, "audio_token_id", None)
    image_id = getattr(config, "image_token_id", None)
    video_tokens = ((input_ids == video_id) & mask).sum(dim=1) if video_id is not None else 0
    audio_tokens = ((input_ids == audio_id) & mask).sum(dim=1) if audio_id is not None else 0
    image_tokens = ((input_ids == image_id) & mask).sum(dim=1) if image_id is not None else 0
    text_tokens = total_tokens - video_tokens - audio_tokens - image_tokens
    return {
        "video": video_tokens,
        "audio": audio_tokens,
        "image": image_tokens,
        "text": text_tokens,
        "total": total_tokens,
    }


def _maybe_init_stats(self) -> None:
    if hasattr(self, "_vidcom2_stats"):
        return
    self._vidcom2_stats = {
        "pre": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "post": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "samples": 0,
    }


def Qwen2_5_OmniThinker_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
    """Patched forward that enables VidCom2 token compression for Qwen2.5-Omni (video tokens only)."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    audio_features = None

    if input_features is not None:
        audio_features = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        if isinstance(video_embeds, (list, tuple)):
            video_embeds = torch.cat(video_embeds, dim=0)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    compression_on = (
        os.getenv("COMPRESSOR") == "vidcom2"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if compression_on:
        source_video_path = kwargs.pop("vidcom2_source_video_path", None)
        if source_video_path is None:
            source_video_path = getattr(self, "_vidcom2_source_video_path", None)
        if isinstance(source_video_path, (list, tuple)):
            source_video_path = source_video_path[0] if len(source_video_path) > 0 else None
        if source_video_path is not None:
            source_video_path = str(source_video_path)
        stats_enabled = os.getenv(_VIDCOM_TOKEN_STATS_ENV, "0") == "1"
        stats_case = os.getenv(_VIDCOM_TOKEN_STATS_CASE_ENV, "0") == "1"
        if stats_enabled or stats_case:
            _maybe_init_stats(self)
            pre_stats = _count_tokens(input_ids, attention_mask, self.config)

        merge_size = self.visual.spatial_merge_size
        base_scale = float(os.getenv("R_RATIO", "0.25"))
        enable_audio_guidance = _env_flag(_VIDCOM_AUDIO_GUIDANCE_ENV, default=True)
        enable_audio_token_guidance = _env_flag(_VIDCOM_AUDIO_TOKEN_GUIDANCE_ENV, default=True)
        audio_fps = float(os.getenv(_VIDCOM_AUDIO_FPS_ENV, "25.0"))
        audio_robust_clip = _AUDIO_ROBUST_CLIP
        audio_weight_frame = 1.0
        audio_weight_token = float(os.getenv(_VIDCOM_AUDIO_WEIGHT_TOKEN_ENV, "1.0"))
        dump_budget_viz = _env_flag(_VIDCOM_BUDGET_VIZ_ENV, default=False)
        budget_viz_dir = os.getenv(_VIDCOM_BUDGET_VIZ_DIR_ENV, "logs/vidcom2_budget_viz")
        budget_viz_cases = _parse_case_selector(os.getenv(_VIDCOM_BUDGET_VIZ_CASES_ENV))
        budget_viz_max_cases = int(os.getenv(_VIDCOM_BUDGET_VIZ_MAX_CASES_ENV, "0"))
        budget_viz_render_now = _env_flag(_VIDCOM_BUDGET_VIZ_RENDER_NOW_ENV, default=True)
        budget_viz_stop_after_target = _env_flag(_VIDCOM_BUDGET_VIZ_STOP_AFTER_TARGET_ENV, default=False)
        if not hasattr(self, "_vidcom2_viz_case_counter"):
            self._vidcom2_viz_case_counter = 0
        if not hasattr(self, "_vidcom2_viz_dumped_counter"):
            self._vidcom2_viz_dumped_counter = 0
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()

        video_splits = torch.split(video_embeds, split_sizes)
        kept_indices: List[Tensor] = []
        kept_video_chunks: List[Tensor] = []
        offset = 0

        for video_idx, (grid, feat) in enumerate(zip(video_grid_thw, video_splits)):
            cur_video_second_per_grid = _resolve_second_per_grid(video_second_per_grid, video_idx)
            keep_local = _compute_keep_indices(
                flat_features=feat,
                grid_thw=grid,
                spatial_merge_size=merge_size,
                base_scale=base_scale,
                enable_audio_guidance=enable_audio_guidance,
                enable_audio_token_guidance=enable_audio_token_guidance,
                audio_embeds=audio_features if enable_audio_guidance else None,
                video_second_per_grid=cur_video_second_per_grid if enable_audio_guidance else None,
                audio_fps=audio_fps,
                audio_weight_frame=audio_weight_frame,
                audio_weight_token=audio_weight_token,
                audio_robust_clip=audio_robust_clip,
            )
            kept_indices.append(keep_local + offset)
            kept_video_chunks.append(feat[keep_local])
            offset += feat.shape[0]

        can_dump_this_case = False
        case_idx = int(self._vidcom2_viz_case_counter)
        self._vidcom2_viz_case_counter += 1
        if dump_budget_viz and (not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            selected_by_cases = (budget_viz_cases is None) or (case_idx in budget_viz_cases)
            selected_by_limit = (budget_viz_max_cases <= 0) or (int(self._vidcom2_viz_dumped_counter) < budget_viz_max_cases)
            can_dump_this_case = selected_by_cases and selected_by_limit

        if can_dump_this_case and len(video_splits) > 0:
            try:
                # Visualize the first video sample in the current call.
                grid0 = video_grid_thw[0]
                feat0 = video_splits[0]
                second_per_grid0 = _resolve_second_per_grid(video_second_per_grid, 0)
                t0, h0, w0 = grid0.tolist()
                tpf0 = (h0 * w0) // (merge_size**2)
                if tpf0 > 0 and feat0.numel() > 0:
                    sel0, sel_channels0 = select_low_var_channels(feat0)
                    v0, f0 = compute_gaussian_scores(sel0, tpf0)
                    visual_token_scores0 = v0 + f0
                    u_video0 = -v0.mean(dim=-1)
                    scales_visual0 = compute_scales(u_video0, base_scale)
                    budget_visual0 = _compute_budget_counts(visual_token_scores0, scales_visual0, tpf0)

                    budget_audio0 = budget_visual0
                    audio_token_norm0 = None
                    if (
                        enable_audio_guidance
                        and audio_features is not None
                        and audio_features.ndim == 2
                        and second_per_grid0 is not None
                    ):
                        frame_audio0 = get_audio_guided_frame_features(
                            video_grid_thw=grid0,
                            video_second_per_grid=second_per_grid0,
                            audio_embeds=audio_features,
                            AUDIO_FPS=audio_fps,
                        )
                        if frame_audio0.shape[0] == u_video0.shape[0]:
                            raw_audio_novelty0 = compute_audio_change_scores(frame_audio0)
                            u_audio0 = _prepare_audio_novelty_scores(
                                raw_novelty=raw_audio_novelty0,
                                clip_c=audio_robust_clip,
                            )
                            audio_token_norm0 = [
                                float(v) for v in u_audio0.detach().float().cpu().tolist()
                            ]
                            fused_frame0 = _fuse_video_audio_scores(
                                u_video=u_video0,
                                u_audio=u_audio0,
                                audio_weight=audio_weight_frame,
                            )
                            scales_audio0 = compute_scales(fused_frame0, base_scale)
                            if enable_audio_token_guidance:
                                token_feat0 = sel0.view(-1, tpf0, sel0.shape[-1])
                                fused_token0 = _fuse_visual_audio_token_scores(
                                    visual_token_scores=visual_token_scores0,
                                    token_features=token_feat0,
                                    frame_audio_features=frame_audio0,
                                    u_audio=u_audio0,
                                    audio_weight=audio_weight_token,
                                    selected_channels=sel_channels0,
                                )
                                budget_audio0 = _compute_budget_counts(fused_token0, scales_audio0, tpf0)
                            else:
                                budget_audio0 = _compute_budget_counts(visual_token_scores0, scales_audio0, tpf0)

                    case_name = f"case_{case_idx:06d}"
                    save_budget_comparison_artifact(
                        output_dir=budget_viz_dir,
                        case_name=case_name,
                        budgets_visual=budget_visual0,
                        budgets_audio=budget_audio0,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=grid0,
                        source_video_path=source_video_path,
                        audio_token_norm=audio_token_norm0,
                        title="WorldSense Budget: Visual-only vs Audio-guided",
                        render_now=budget_viz_render_now,
                        extra_meta={
                            "enable_audio_guidance": bool(enable_audio_guidance),
                            "enable_audio_token_guidance": bool(enable_audio_token_guidance),
                            "audio_weight_frame": float(audio_weight_frame),
                            "audio_weight_token": float(audio_weight_token),
                            "audio_robust_clip": float(audio_robust_clip),
                            "base_scale": float(base_scale),
                            "audio_fps": float(audio_fps),
                            "t": int(t0),
                            "frame_tokens": int(tpf0),
                            "case_index": int(case_idx),
                        },
                    )
                    self._vidcom2_viz_dumped_counter += 1

                    reached_limit = budget_viz_max_cases > 0 and int(self._vidcom2_viz_dumped_counter) >= budget_viz_max_cases
                    reached_selected = (
                        budget_viz_cases is not None
                        and len(budget_viz_cases) > 0
                        and all(c < int(self._vidcom2_viz_case_counter) for c in budget_viz_cases)
                    )
                    if budget_viz_stop_after_target and (reached_limit or reached_selected):
                        raise KeyboardInterrupt(
                            "Stopped early after collecting selected budget-viz cases. "
                            "You can now run offline visualization rendering."
                        )
            except KeyboardInterrupt:
                raise
            except Exception:
                # Visualization is optional; compression path should not fail because of it.
                pass

        kept_indices = torch.sort(torch.cat(kept_indices)).values
        video_embeds = torch.cat(kept_video_chunks, dim=0)

        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_video_positions = video_token_positions[kept_indices]
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_positions = all_positions[~video_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

        def _prune_attention(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn is None:
                return None
            if attn.dim() == 2:
                return attn[:, keep_token_indices]
            if attn.dim() == 4:
                return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = _prune_attention(attention_mask)
        if position_ids is not None:
            position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

        if stats_enabled or stats_case:
            post_stats = _count_tokens(input_ids, attention_mask, self.config)
            for key in self._vidcom2_stats["pre"]:
                self._vidcom2_stats["pre"][key] += int(pre_stats[key].sum().item())
                self._vidcom2_stats["post"][key] += int(post_stats[key].sum().item())
            self._vidcom2_stats["samples"] += int(pre_stats["total"].shape[0])
            if stats_case:
                self._vidcom2_last_stats = [
                    {
                        "pre": {k: int(pre_stats[k][i].item()) for k in pre_stats},
                        "post": {k: int(post_stats[k][i].item()) for k in post_stats},
                    }
                    for i in range(pre_stats["total"].shape[0])
                ]

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
        )

    if not return_dict:
        output = (logits,) + outputs
        return (loss,) + output if loss is not None else output

    return Qwen2_5OmniThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
