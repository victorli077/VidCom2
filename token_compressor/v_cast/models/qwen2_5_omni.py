from typing import List, Optional, Union
import os

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)

from token_compressor.v_cast import (
    v_cast_compression,
    v_cast_config_from_env,
)


def _unpack_audio_outputs(audio_outputs):
    if isinstance(audio_outputs, tuple):
        audio_features = audio_outputs[0]
        attn_logits = audio_outputs[1] if len(audio_outputs) > 1 and torch.is_tensor(audio_outputs[1]) else None
        return audio_features, attn_logits
    return audio_outputs, None


def _compute_keep_indices(
    flat_features: Tensor,
    grid_thw: Tensor,
    spatial_merge_size: int,
    config: dict,
) -> Tensor:
    return v_cast_compression(
        flat_features=flat_features,
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        retain_ratio=float(config["retain_ratio"]),
        min_k=int(config["min_k"]),
        budget_temp=float(config["budget_temp"]),
        score_mode=str(config["score_mode"]),
        outlier_weight=float(config["outlier_weight"]),
        norm_weight=float(config["norm_weight"]),
        verbose=bool(config["print"]),
    )


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

    raw_audio_features: Optional[Tensor] = None
    audio_attn_logits: Optional[Tensor] = None

    if input_features is not None:
        audio_outputs = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        raw_audio_features, audio_attn_logits = _unpack_audio_outputs(audio_outputs)
        raw_audio_features = raw_audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, raw_audio_features)

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
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

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
        os.getenv("COMPRESSOR") == "v_cast"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on and inputs_embeds.shape[0] != 1:
        compression_on = False

    if compression_on:
        config = v_cast_config_from_env()

        merge_size = self.visual.spatial_merge_size
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()

        video_splits = torch.split(video_embeds, split_sizes)
        kept_indices: List[Tensor] = []
        kept_video_chunks: List[Tensor] = []
        offset = 0

        for vid_idx, (grid, feat) in enumerate(zip(video_grid_thw, video_splits)):
            keep_local = _compute_keep_indices(
                flat_features=feat,
                grid_thw=grid,
                spatial_merge_size=merge_size,
                config=config,
            )
            if keep_local.numel() == 0:
                keep_local = torch.arange(feat.shape[0], device=feat.device)
            kept_indices.append(keep_local + offset)
            kept_video_chunks.append(feat[keep_local])
            if bool(config["print"]):
                tag = f"[video={vid_idx}] "
                print(
                    f"[V-CAST] {tag}keep={int(keep_local.numel())}/{int(feat.shape[0])}",
                    flush=True,
                )
            offset += feat.shape[0]

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
