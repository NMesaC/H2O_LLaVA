"""
H2O KV Cache Attention (Simplification of Implementation)
Author: Nicholas Mesa-Cucalon
Source: https://github.com/meta-llama/llama-recipes/blob/main/recipes/experimental/long_context/H2O/utils/llama.py
18-580 - Topics in Vision-Language Models
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import pdb
import types
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)
from .cache import Cache, HHCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Helper Function for Applying RoPE
def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

"""
H2O Attention Implementation
"""
class H2OLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.positional_rolling = config.enable_position_rolling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    """
    =========
    Forward
    =========
    This rewrite from the original implementation is to simplify the internal logic for readability,
    but differences can later be reimplemented.

    Source: https://github.com/meta-llama/llama-recipes/blob/main/recipes/experimental/long_context/H2O/utils/llama.py

    Note 1: Tensor Rank Parallelism wasn't used during training, so we ommit it for clarity of reading.

    Note 2: In a future Transformers Update, positional embeddings will be mandatory, and we won't
    compute Cos and Sin in the Attention Layers. We might want to add this to save time later.

    Note 3: For simplicity and clarity of the algorithm, we remove positional rolling.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Get batch size and query length from the hidden_state input
        b, q_len, _ = hidden_states.size()

        # Note 1: Computing Q,K,V normally here
        # Shape of Q, K, V: [b, q_len, d_model]
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Q shape -> [b, q_len, n_heads, d_h], K/V Shape -> [b, q_len, n_kv_heads, d_h]
        # Then swap shape to be [b, n_heads, q_len, d_h] for Q,K,V for nn.Linear
        Q = Q.view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get KV Cache
        past_key_value = getattr(self, "past_key_value", past_key_value)

        # Note 3: Computing RoPE Normally
        cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, None)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_value.update(K, V, self.layer_idx, cache_kwargs)

        # K,V go from [b, n_kv_heads, q_len, d_h] -> [b, num_attention_heads, seqlen, d_h] (For GQA)
        K = repeat_kv(K, self.num_key_value_groups)
        V = repeat_kv(V, self.num_key_value_groups)

        # Compute attention probs
        attn_probs = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : K.shape[-2]]
            attn_probs = attn_probs + causal_mask

        # Compute attn softmax and upcast to fp32
        attn_probs = nn.functional.softmax(attn_probs, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Update KV Cache based on Heavy-Hitter Oracle
        """
        TODO: READ THIS CODE
        """
        if past_key_value is not None:
            past_key_value.update_slimming(attn_probs, self.num_key_value_groups, self.layer_idx)

        #Apply dropout to attn_scores
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, V)

        # attn_output: [b, num_attn_heads, q_len, d_h] -> [b, q_len, num_attn_heads, d_h]
        # Now a contigous tensor in memory for optimized usage
        attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output: [b, q_len, num_attn_heads, d_h] -> [b, q_len, num_attn_heads * d_h]
        attn_output = attn_output.reshape(b, q_len, self.hidden_size)

        # Compute output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_probs = None

        return attn_output, attn_probs, past_key_value

"""
=========
Enable H2OCache Forward
=========
This function is unchanged except a single line at Note 3.
Source: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/llama/modeling_llama.py
"""
def enable_h2ocache_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    # Determine if we want to output certain results or not
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Typically done for CausalLM's
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    # Cache + Gradient Checkpoint Check
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # Turn tokenized inputs into embeds if necessary
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Initialize variables to supress warnings
    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            """
            Note 3: This changes DynamicCache -> HHCache
            """
            past_key_values = HHCache.from_legacy_cache(self.num_window_length, self.num_heavy_hitter_tokens, past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    # Create cache position variable if not already provided
    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Create position ids if not already provided
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Create our causal mask
    causal_mask = self._update_causal_mask(attention_mask = attention_mask,
                                           input_tensor = inputs_embeds,
                                           cache_position = cache_position)

    # Embed positions
    hidden_states = inputs_embeds

    # Decoder layers
    all_hidden_states  = () if output_hidden_states else None
    all_self_attns     = () if output_attentions else None
    next_decoder_cache = None

    # Iterate throughout the decoder layers, keeping track of information if needed
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # Add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # Manage Cache state if needed
    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )

    # Manage Return Dict if needed
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
