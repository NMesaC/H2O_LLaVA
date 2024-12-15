"""
H2O LLaMA Implementation
Author: Nicholas Mesa-Cucalon
18-580: Topics in Vision-Language Models
Adapted From:
    https://arxiv.org/abs/2306.14048
    https://github.com/FMInference/H2O/tree/main
    https://github.com/meta-llama/llama-recipes/tree/main/recipes/experimental/long_context/H2O
"""

"""
Imports
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

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)
from .cache import Cache, HHCache, StaticCache

__all__ = ['H2OMaskedLlamaForCausalLM']

class H2OMaskedLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig,
        layer_idx: int = 0, num_h2 : int = 0,
        w_length : int = 0, local : bool = True):
        # Initialize the Super Class
        super().__init__(config)
        # Initialize H2O Variables
        self.layer_idx = layer_idx
        # NOTE: The number of HH Tokens = The number of Recent Tokens
        self.num_heavy_hitter_tokens = num_h2
        self.window_length = w_length
        # Choose masking function
        if local:
            self.mask_fn = self.local_mask
        else:
            self.mask_fn = self.global_mask

    """
    Using local statistics / The Described H2O Algorithm, compute the H2 Mask
    Note 1: We do not compute / normalize the scores
    """
    def local_mask(self, attn_probs):
        # Set needed variables
        num_heavy  = self.num_heavy_hitter_tokens
        num_recent = self.num_heavy_hitter_tokens
        seq_len = attn_probs.shape[-1]

        # Compute row-wise softmax across attn probs
        attn_scores = nn.functional.softmax(attn_probs, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Compute accumulated attn score
        # a. We compute this score for the first "num_heavy" tokens we've seen
        accumulated_attn_score = torch.sum(attn_scores[:,:,0:num_heavy,:], dim=-2)
        # b. Zero out the tokens we are not considering
        accumulated_attn_score[:,:,num_heavy:] = 0

        # Initialize our mask
        mask_bottom = torch.zeros_like(attn_scores, dtype=torch.bool)
        # We set the first "num_heavy" tokens to be true
        mask_bottom[:,:,0:num_heavy,0:num_heavy] = True

        # Iterate through all the token indeces starting from heavy budget, and decide which to keep
        for token_idx in range(num_heavy, seq_len):
            # Find the attention score for the given token_idx for each batch and attn head
            tmp_attn_idx = attn_scores[:,:,token_idx,:]

            # Find the top "num_heavy - 1" indeces to keep across each batch and attn head
            # Shape -> [b,n_attn_heads, seq_len]
            _, tmp_topk_idx = accumulated_attn_score.topk(k=num_heavy-1, dim=-1)

            # Create a zeros idx matrix in the same shape as the top heavy budget indeces
            zeros_idx = torch.zeros_like(tmp_attn_idx, dtype=torch.bool)

            # Across the seq_len dimension, set all indeces tmp_topk_idx specifies in its last dim to be true
            mask_bottom_idx = zeros_idx.scatter(-1, tmp_topk_idx, True)

            # Additionally set the token_idx to be true
            mask_bottom_idx[:,:, token_idx] = True

            # Set the query indeces to be according to the mask_bottom_idx
            mask_bottom[:,:,token_idx,:] = mask_bottom_idx

            # Add the attention idx value to our accumulated attention scores
            accumulated_attn_score += tmp_attn_idx
            accumulated_attn_score = accumulated_attn_score * mask_bottom_idx

        # Compute our recent budget elements to keep
        ones = torch.ones_like(attn_scores, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-num_recent)
        mask_bottom = torch.logical_or(mask_bottom, ones)
        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # Zero out the softmax elements we are going to ignore
        attn_scores[~mask_bottom] = 0
        return attn_scores

    """
    Using global statistics compute the H2 Mask
    Note 1:
        This is accomplished by greedily summing all the attn scores across the columns and rows,
        and computing the tokens we'll keep based on that.
    Note 2:
        Trades speed for accuracy.
    """
    def global_mask(self, attn_probs):
        # Load in how many heavy and recent tokens we have
        num_heavy  = self.num_heavy_hitter_tokens
        num_recent = self.num_heavy_hitter_tokens
        # Compute row-wise softmax across attn_probs
        attn_scores = F.softmax(attn_probs, dim=-1, dtype=torch.float32).to(Q.dtype)
        # Sum down the columns to compute the summed attn score for each column
        tmp_sum = torch.sum(attn_scores, dim=-2)
        # Get idxs of the topk summed attention scores row-wise (We've summed across columns and now we pick from the remaining row)
        _, tmp_topk = tmp_sum.topk(k=num_heavy, dim=-1)
        """
        Build Zeroes of Mask
        """
        # Create a zero matrix in the same shape of tmp_sum; zeros -> [b, n_heads, seq_len]
        zeros       = torch.zeros_like(tmp_sum, dtype=torch.bool)
        # Create a tensor of indeces where we place True at all locations tmp_topk specifies
        mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
        # Turn shape from [b,n_heads,1,seq_len] -> [b,n_heads,seq_len,seq_len]
        mask_bottom = mask_bottom.expand(mask_bottom.shape[0], mask_bottom.shape[1], attn_probs.shape[-2], mask_bottom.shape[-1])
        """
        Build Ones of Mask
        """
        # Create a tensor of 1's in the same shape as attention_probs
        ones = torch.ones_like(attn_probs, dtype=torch.bool)
        # Get the lower triangular matrix of 1's and fill the top with zeroes
        ones = torch.tril(ones, diagonal=num_recent)
        # Get the upper triangular matrix of 0's and keep num_recent extra 1's down the diagonal elems
        ones = torch.triu(ones, diagonal=-num_recent)
        # Combine the zero mask and ones mask
        mask_bottom = torch.logical_or(mask_bottom, ones)
        # Zero out the softmax elements we are going to ignore
        attn_scores[~mask_bottom] = 0
        return attn_scores

    """
    =========
    Forward
    =========
    This forward function will have the same signature as Vanilla LlaMA, but will internally
    compute the attention masks differently. There are slight differences within our forward pass,
    and we will mark them as needed.

    Note 0: The attention masks are computed to be the same as the H2O Cache, but we cannot
    disable caching / using a DynamicCache for this forward function. This is because in
    the generation function, we cannot have a decoder only model and also pass in input embeddings.
    We need to pass in input embeddings due to the LLaVA architecture.
    Justification: decoder-only models with inputs_embeds forwarding must use caching
    (otherwise we can't detect whether we are generating the first new token or not,
    and we only want to use the embeddings for the first new token)

    Note 1: We don't implement Tensor Rank Parallelism here. Future work could do this.

    Note 2: In a future Transformers Update, positional embeddings will be mandatory, and we won't
    compute Cos and Sin in the Attention Layers. We might want to add this to save time later.
    """
    def forward(
        self,
        hidden_states:     torch.Tensor,
        attention_mask:    Optional[torch.Tensor] = None,
        position_ids:      Optional[torch.LongTensor] = None,
        past_key_value:    Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache:         bool = False,
        cache_position:    Optional[torch.LongTensor] = None,
        **kwargs,     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Get batch size and query length from the hidden_state input
        b, seq_len, _ = hidden_states.size()

        # Note 1: Computing Q,K,V normally here
        # Shape of Q, K, V: [b, seq_len, d_model]
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Q shape -> [b, seq_len, n_heads, d_h], K/V Shape -> [b, seq_len, n_kv_heads, d_h]
        # Then swap shape to be [b, n_heads, seq_len, d_h] for Q,K,V for nn.Linear
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get kv cache
        past_key_value = getattr(self, "past_key_value", past_key_value)

        # Note 2: Default using rotary embeddings to compute cos and sin
        cos, sin = self.rotary_emb(V, position_ids)
        Q, K     = apply_rotary_pos_emb(Q, K, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_value.update(K, V, self.layer_idx, cache_kwargs)

        # K,V go from [b, n_kv_heads, seq_len, d_h] -> [b, num_attention_heads, seqlen, d_h] (For GQA)
        K = repeat_kv(K, self.num_key_value_groups)
        V = repeat_kv(V, self.num_key_value_groups)

        # Compute attention probs
        attn_probs = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : K.shape[-2]]
            attn_probs  = attn_probs + causal_mask

        """
        NOTE:
            We call our mask function here and get our attention scores, independent of which
            way we compute the HH Mask.
        """
        attn_scores = self.mask_fn(attn_probs)

        # Apply dropout to attn_scores
        attn_scores = nn.functional.dropout(attn_scores, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_scores, V)

        # attn_output: [b, num_attn_heads, seq_len, d_h] -> [b, seq_len, num_attn_heads, d_h]
        # Now a contigous tensor in memory for optimized usage
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output: [b, seq_len, num_attn_heads, d_h] -> [b, seq_len, num_attn_heads * d_h]
        attn_output = attn_output.reshape(b, seq_len, -1)

        # Compute output projection
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_scores = None

        return attn_output, attn_scores, past_key_value

class H2OMaskedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        num_h2 = config.num_heavy_hitter_tokens
        w_length = config.num_window_length
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OMaskedLlamaAttention(config, layer_idx, num_h2, w_length)
