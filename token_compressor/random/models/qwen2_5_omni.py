from typing import Optional, Union
import os

import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


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


def random_config_from_env() -> dict:
    ratio = _env_float("RANDOM_R_RATIO", _env_float("R_RATIO", 0.35))
    return {
        "retain_ratio": max(0.0, min(1.0, ratio)),
        "seed": _env_int("RANDOM_SEED", 1234),
        "print": _env_bool("RANDOM_PRINT", _env_bool("V_CAST_PRINT", True)),
    }


def _bounded_keep_count(num_tokens: int, retain_ratio: float) -> int:
    if num_tokens <= 0:
        return 0
    keep = int(round(float(num_tokens) * float(retain_ratio)))
    if retain_ratio > 0:
        keep = max(1, keep)
    return min(num_tokens, keep)


def _random_keep_indices(num_tokens: int, retain_ratio: float, *, device: torch.device, seed: int) -> Tensor:
    keep = _bounded_keep_count(num_tokens, retain_ratio)
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)
    if keep <= 0:
        return torch.zeros((0,), device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(num_tokens) * 1009 + int(keep) * 9173)
    indices = torch.randperm(num_tokens, device=device, generator=generator)[:keep]
    return torch.sort(indices).values


def _unpack_audio_outputs(audio_outputs):
    if isinstance(audio_outputs, tuple):
        audio_features = audio_outputs[0]
        attn_logits = audio_outputs[1] if len(audio_outputs) > 1 and torch.is_tensor(audio_outputs[1]) else None
        return audio_features, attn_logits
    return audio_outputs, None


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
    zeros = torch.zeros_like(total_tokens)
    video_tokens = ((input_ids == video_id) & mask).sum(dim=1) if video_id is not None else zeros
    audio_tokens = ((input_ids == audio_id) & mask).sum(dim=1) if audio_id is not None else zeros
    image_tokens = ((input_ids == image_id) & mask).sum(dim=1) if image_id is not None else zeros
    text_tokens = total_tokens - video_tokens - audio_tokens - image_tokens
    return {
        "video": video_tokens,
        "audio": audio_tokens,
        "image": image_tokens,
        "text": text_tokens,
        "total": total_tokens,
    }


def _maybe_init_stats(self) -> None:
    if hasattr(self, "_random_stats"):
        return
    self._random_stats = {
        "pre": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "post": {"video": 0, "audio": 0, "image": 0, "text": 0, "total": 0},
        "samples": 0,
    }


def _assert_audio_preserved(pre_stats: dict, post_stats: dict) -> None:
    enabled = _env_bool("RANDOM_ASSERT_AUDIO_PRESERVED", True)
    if not enabled:
        return
    pre_audio = pre_stats["audio"]
    post_audio = post_stats["audio"]
    violations = (pre_audio > 0) & (post_audio != pre_audio)
    if bool(violations.any().item()):
        bad = torch.nonzero(violations, as_tuple=False).flatten().tolist()
        details = [
            f"idx={int(i)} pre_audio={int(pre_audio[i].item())} post_audio={int(post_audio[i].item())}"
            for i in bad[:5]
        ]
        raise RuntimeError("Audio preservation violation in random compressor: " + "; ".join(details))


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

    if input_features is not None:
        audio_outputs = self.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        raw_audio_features, _ = _unpack_audio_outputs(audio_outputs)
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
        os.getenv("COMPRESSOR") == "random"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on and inputs_embeds.shape[0] != 1:
        compression_on = False

    if compression_on:
        config = random_config_from_env()
        stats_enabled = (
            os.getenv("TOKEN_STATS", "0") == "1"
            or os.getenv("VIDCOM_TOKEN_STATS", "0") == "1"
            or _env_bool("RANDOM_ASSERT_AUDIO_PRESERVED", True)
        )
        stats_case = (
            os.getenv("TOKEN_STATS_CASE", "0") == "1"
            or os.getenv("VIDCOM_TOKEN_STATS_CASE", "0") == "1"
        )
        if stats_enabled or stats_case:
            _maybe_init_stats(self)
            pre_stats = _count_tokens(input_ids, attention_mask, self.config)

        merge_size = self.visual.spatial_merge_size
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()
        video_splits = torch.split(video_embeds, split_sizes)

        kept_indices = []
        kept_video_chunks = []
        offset = 0
        for vid_idx, feat in enumerate(video_splits):
            keep_local = _random_keep_indices(
                int(feat.shape[0]),
                float(config["retain_ratio"]),
                device=feat.device,
                seed=int(config["seed"]) + vid_idx * 7919,
            )
            kept_indices.append(keep_local + offset)
            kept_video_chunks.append(feat[keep_local])
            if bool(config["print"]):
                print(
                    f"[Random] video={vid_idx} keep={int(keep_local.numel())}/{int(feat.shape[0])} "
                    f"ratio={float(config['retain_ratio']):.4f}",
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

        if stats_enabled or stats_case:
            post_stats = _count_tokens(input_ids, attention_mask, self.config)
            _assert_audio_preserved(pre_stats, post_stats)
            for key in self._random_stats["pre"]:
                self._random_stats["pre"][key] += int(pre_stats[key].sum().item())
                self._random_stats["post"][key] += int(post_stats[key].sum().item())
            self._random_stats["samples"] += int(pre_stats["total"].shape[0])
            if stats_case:
                self._random_last_stats = [
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

