from typing import Optional, Union
import os
import types

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)

from token_compressor.omnizip import omnizip_sequence

_OMNIZIP_TOKEN_STATS_ENV = "VIDCOM_TOKEN_STATS"
_OMNIZIP_TOKEN_STATS_CASE_ENV = "VIDCOM_TOKEN_STATS_CASE"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


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
    if hasattr(self, "_omnizip_stats"):
        return
    self._omnizip_stats = {
        "pre": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "post": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "samples": 0,
        "audio_retention_ratio_sum": 0.0,
        "audio_compression_ratio_sum": 0.0,
    }


def _unpack_audio_outputs(audio_outputs):
    if isinstance(audio_outputs, tuple):
        audio_features = audio_outputs[0]
        attn_logits = audio_outputs[1] if len(audio_outputs) > 1 and torch.is_tensor(audio_outputs[1]) else None
        return audio_features, attn_logits
    return audio_outputs, None


def _attention_importance_from_module(
    attn_module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_length, _ = hidden_states.size()
    query_states = attn_module.q_proj(hidden_states).reshape(seq_length, attn_module.num_heads, -1)
    key_states = attn_module.k_proj(hidden_states).reshape(seq_length, attn_module.num_heads, -1)
    value_states = attn_module.v_proj(hidden_states).reshape(seq_length, attn_module.num_heads, -1)

    q = query_states.transpose(0, 1)
    k = key_states.transpose(0, 1)
    scale = q.shape[-1] ** -0.5
    token_importance = torch.zeros(seq_length, device=hidden_states.device)
    head_chunk = 4
    seq_chunk = 512
    with torch.no_grad():
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q_seg = q[:, start:end, :]
            k_seg = k[:, start:end, :]
            seg_len = int(end - start)
            if seg_len <= 0:
                continue
            seg_importance = torch.zeros(seg_len, device=hidden_states.device)
            for h in range(0, q_seg.shape[0], head_chunk):
                q_h = q_seg[h : h + head_chunk]
                k_h = k_seg[h : h + head_chunk]
                for i in range(0, seg_len, seq_chunk):
                    attn_chunk = torch.matmul(q_h[:, i : i + seq_chunk, :], k_h.transpose(-1, -2)) * scale
                    attn_chunk = torch.nn.functional.softmax(attn_chunk, dim=-1)
                    seg_importance += attn_chunk.sum(dim=(0, 1))
                    del attn_chunk
            token_importance[start:end] = seg_importance / (q_seg.shape[0] * seg_len)
    return query_states, key_states, value_states, token_importance


def install_omnizip_audio_attention_patch(thinker) -> None:
    """Patch Qwen2.5-Omni audio tower to expose official OmniZip attention scores."""
    if getattr(thinker, "_omnizip_audio_patch_installed", False):
        return
    audio_tower = getattr(thinker, "audio_tower", None)
    if audio_tower is None:
        return
    for layer in audio_tower.layers:
        layer._omnizip_original_forward = layer.forward

        def layer_forward(self, hidden_states, cu_seqlens, attention_mask=None, return_logits=False, **kwargs):
            residual = hidden_states
            normed = self.self_attn_layer_norm(hidden_states)
            if return_logits:
                _, _, _, logits = _attention_importance_from_module(self.self_attn, normed, cu_seqlens)
            else:
                logits = None
            hidden_states = self.self_attn(
                hidden_states=normed,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.fc2(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            return (hidden_states,), logits

        layer.forward = types.MethodType(layer_forward, layer)

    audio_tower._omnizip_original_forward = audio_tower.forward

    def audio_tower_forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = torch.nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = torch.nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)
        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        logits = None
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs, layer_logits = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                return_logits=idx == len(self.layers) - 1,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if layer_logits is not None:
                logits = layer_logits

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)

        if logits is not None:
            attn_mean = logits
            if attn_mean.shape[0] % 2 == 1:
                attn_mean = attn_mean[:-1]
            if attn_mean.shape[0] >= 2:
                attn_mean = attn_mean.view(-1, 2).mean(dim=-1)
        else:
            attn_mean = None
        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=token_audio), attn_mean

    audio_tower.forward = types.MethodType(audio_tower_forward, audio_tower)
    thinker._omnizip_audio_patch_installed = True


def _get_audio_features_with_logits(
    self,
    input_features: torch.FloatTensor,
    feature_attention_mask: Optional[torch.LongTensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
):
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    else:
        audio_feature_lengths = None

    feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(feature_lens)
    audio_outputs = self.audio_tower(
        input_features,
        feature_lens=feature_lens,
        aftercnn_lens=audio_feat_lengths,
    )
    if isinstance(audio_outputs, tuple):
        audio_base = audio_outputs[0]
        attn_logits = audio_outputs[1] if len(audio_outputs) > 1 else None
    else:
        audio_base = audio_outputs
        attn_logits = None
    audio_features = audio_base.last_hidden_state
    if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
        raise ValueError("length of audio_features should match audio_output_lengths")
    return audio_features, attn_logits


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
    """Patched forward that enables OmniZip token compression for Qwen2.5-Omni."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    install_omnizip_audio_attention_patch(self)

    image_mask = None
    video_mask = None
    audio_mask = None
    audio_features = None
    audio_attn_logits = None

    if input_features is not None:
        audio_outputs = _get_audio_features_with_logits(
            self,
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features, audio_attn_logits = _unpack_audio_outputs(audio_outputs)
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
        os.getenv("COMPRESSOR") == "omnizip"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if compression_on:
        stats_enabled = (
            os.getenv(_OMNIZIP_TOKEN_STATS_ENV, "0") == "1"
            or os.getenv("TOKEN_STATS", "0") == "1"
        )
        stats_case = (
            os.getenv(_OMNIZIP_TOKEN_STATS_CASE_ENV, "0") == "1"
            or os.getenv("TOKEN_STATS_CASE", "0") == "1"
        )
        if stats_enabled or stats_case:
            _maybe_init_stats(self)
            pre_stats = _count_tokens(input_ids, attention_mask, self.config)

        rho_audio = float(os.getenv("OMNIZIP_RHO_AUDIO", "0.3"))
        if "OMNIZIP_RHO_VIDEO" in os.environ:
            rho_video = float(os.getenv("OMNIZIP_RHO_VIDEO", "0.65"))
        else:
            retain_ratio = float(os.getenv("R_RATIO", os.getenv("RETAIN_RATIO", "0.35")))
            rho_video = 1.0 - retain_ratio
        contextual_ratio = float(os.getenv("OMNIZIP_CONTEXTUAL_RATIO", "0.05"))
        g = int(os.getenv("OMNIZIP_G", "3"))
        rho_min = float(os.getenv("OMNIZIP_RHO_MIN", "0.35"))
        rho_max = float(os.getenv("OMNIZIP_RHO_MAX", "0.75"))
        audio_preserve = _env_bool("OMNIZIP_AUDIO_PRESERVE", False)

        if audio_preserve:
            os.environ.setdefault("OMNIZIP_ASSERT_AUDIO_PRESERVED", "1")
        has_audio_tokens = audio_features is not None and audio_features.numel() > 0
        if has_audio_tokens and audio_attn_logits is None and not _env_bool("OMNIZIP_ALLOW_ATTN_FALLBACK", False):
            raise RuntimeError(
                "Faithful OmniZip requires audio attention logits. "
                "Set OMNIZIP_ALLOW_ATTN_FALLBACK=1 only for debugging."
            )

        sequence_output = omnizip_sequence(
            input_embeds=inputs_embeds,
            attn_logits=audio_attn_logits,
            input_ids=input_ids,
            audio_token_id=self.config.audio_token_id,
            video_token_id=self.config.video_token_id,
            video_grid_thw=video_grid_thw,
            merging_ratio_audio=rho_audio,
            merging_ratio_v=rho_video,
            contextual_ratio=contextual_ratio,
            g=g,
            audio_preserve=audio_preserve,
            rho_min=rho_min,
            rho_max=rho_max,
        )
        inputs_embeds = sequence_output.input_embeds
        keep_token_indices = torch.nonzero(sequence_output.keep_mask, as_tuple=True)[0]
        audio_comp_stats = sequence_output.stats

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
        if audio_mask is not None:
            audio_mask = audio_mask[:, keep_token_indices, :]

        if stats_enabled or stats_case:
            post_stats = _count_tokens(input_ids, attention_mask, self.config)
            for key in self._omnizip_stats["pre"]:
                self._omnizip_stats["pre"][key] += int(pre_stats[key].sum().item())
                self._omnizip_stats["post"][key] += int(post_stats[key].sum().item())
            self._omnizip_stats["samples"] += int(pre_stats["total"].shape[0])
            if audio_comp_stats is not None:
                self._omnizip_stats["audio_retention_ratio_sum"] += audio_comp_stats["audio_retention_ratio"]
                self._omnizip_stats["audio_compression_ratio_sum"] += audio_comp_stats["audio_compression_ratio"]
            if stats_case:
                self._omnizip_last_stats = [
                    {
                        "pre": {k: int(pre_stats[k][i].item()) for k in pre_stats},
                        "post": {k: int(post_stats[k][i].item()) for k in post_stats},
                        "audio_retention_ratio": None
                        if audio_comp_stats is None
                        else float(audio_comp_stats["audio_retention_ratio"]),
                        "audio_compression_ratio": None
                        if audio_comp_stats is None
                        else float(audio_comp_stats["audio_compression_ratio"]),
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
