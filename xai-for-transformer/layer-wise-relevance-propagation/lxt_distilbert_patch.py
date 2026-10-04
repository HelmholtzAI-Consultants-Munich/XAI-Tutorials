"""Experimental LXT patch for Hugging Face DistilBERT 4.52.4."""

import math
from functools import partial

import torch
from torch import nn
from transformers.models.distilbert import modeling_distilbert

from lxt.efficient import monkey_patch
from lxt.efficient.patches import dropout_forward, layer_norm_forward, patch_method
from lxt.efficient.rules import divide_gradient, identity_rule_implicit


def _attention_forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
    batch_size, query_length, _ = query.size()
    key_length = key.size(1)
    head_dim = self.dim // self.n_heads

    def split_heads(tensor):
        return tensor.view(batch_size, -1, self.n_heads, head_dim).transpose(1, 2)

    def merge_heads(tensor):
        return tensor.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * head_dim
        )

    query_states = split_heads(self.q_lin(query)) / math.sqrt(head_dim)
    key_states = split_heads(self.k_lin(key))
    value_states = split_heads(self.v_lin(value))

    scores = torch.matmul(query_states, key_states.transpose(2, 3))
    scores = divide_gradient(scores, 2)
    if mask is not None:
        padding = (mask == 0).view(batch_size, 1, 1, key_length).expand_as(scores)
        scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)

    weights = nn.functional.softmax(scores, dim=-1)
    weights = self.dropout(weights)
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, value_states)
    context = divide_gradient(context, 2)
    context = self.out_lin(merge_heads(context))
    return (context, weights) if output_attentions else (context,)


def _ff_chunk(self, inputs):
    hidden = self.lin1(inputs)
    hidden = identity_rule_implicit(self.activation, hidden)
    return self.dropout(self.lin2(hidden))


def patch_distilbert_for_attnlrp(verbose=False):
    """Patch DistilBERT's attention and MLP operations for LXT AttnLRP."""
    patch_map = {
        nn.LayerNorm: partial(patch_method, layer_norm_forward),
        nn.Dropout: partial(patch_method, dropout_forward),
        modeling_distilbert.MultiHeadSelfAttention: partial(
            patch_method, _attention_forward
        ),
        modeling_distilbert.FFN: partial(
            patch_method, _ff_chunk, method_name="ff_chunk"
        ),
    }
    monkey_patch(modeling_distilbert, patch_map=patch_map, verbose=verbose)
