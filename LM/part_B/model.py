# model.py

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class LoRALayer(nn.Module):
    """Low-rank adapter:  delta(x) = (alpha / rank) * (x A^T) B^T.

    A is initialised randomly and B is initialised to zero, so at the start
    B @ A = 0 and the adapter is a no-op -> the model behaves exactly like the
    pretrained one and adapts gradually as B moves away from zero.

    IMPORTANT: `from_pretrained` runs this __init__ inside transformers'
    `no_init_weights()` context, which replaces every `torch.nn.init.*` function
    (kaiming_uniform_, normal_, ...) with a no-op to speed up loading. Because A
    is NOT present in the pretrained checkpoint, an `nn.init` call here would be
    silently skipped, leaving A as uninitialised `torch.empty` memory (NaN/huge)
    and poisoning training before the first step. We therefore initialise A with
    `torch.randn`, a tensor factory that is never patched. (std = 1/sqrt(in_dim)
    reproduces the scale of kaiming_uniform_(a=sqrt(5)) for this matrix shape.)
    B stays exactly zero, so the adapter is still a no-op at initialisation.
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(in_dim))  # d -> r
        self.B = nn.Parameter(torch.zeros(out_dim, rank))                     # r -> d (init 0)
        self.scaling = alpha / rank

    def forward(self, x):
        # x: (..., in_dim) -> (..., out_dim)
        return self.scaling * ((x @ self.A.t()) @ self.B.t())


class CustomGPT2Attention(GPT2Attention):
    """GPT2 self-attention with a LoRA adapter on each of Q, K, V.

    GPT2 computes Q, K, V together through the fused c_attn projection
    (d -> 3d) and then splits. We add an independent LoRA update to each of
    the three slices right after the split.
    """

    def __init__(self, config, rank, alpha):
        super().__init__(config)
        d = self.embed_dim  # = config.hidden_size (768 for gpt2 small)
        self.lora_q = LoRALayer(d, d, rank, alpha)
        self.lora_k = LoRALayer(d, d, rank, alpha)
        self.lora_v = LoRALayer(d, d, rank, alpha)

    # forward copied from transformers 4.38.0 GPT2Attention, with LoRA injected
    # https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/gpt2/modeling_gpt2.py
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            # --- LoRA adapters on the query, key, and value projections ---
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(hidden_states)
            value = value + self.lora_v(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2_LoRA(GPT2LMHeadModel):
    """GPT2LMHeadModel whose attention blocks carry LoRA adapters on Q/K/V."""

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        for block in self.transformer.h:
            old_attn = block.attn
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # keep the pretrained attention weights; strict=False because the
            # new module has extra LoRA keys not present in old_attn's state_dict
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def mark_only_lora_as_trainable(model):
    """Freeze everything, then unfreeze only the LoRA adapter parameters."""
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALayer):
            for param in module.parameters():
                param.requires_grad = True
    return model