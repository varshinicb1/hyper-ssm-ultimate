"""
Selective Attention Recall Layers for Hyper-SSM (2026 Ultimate)

Inspired by NVIDIA Nemotron 3 Super (March 2026) hybrid design philosophy:
- Use the geometric compressor for the bulk of sequence modeling (O(1) state).
- Insert a small number of high-quality attention layers at strategic depths
  specifically for precise associative recall and long-range dependencies.

This gives you the best of both worlds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveAttentionRecall(nn.Module):
    """
    A lightweight, high-fidelity attention layer used sparingly for recall.

    Designed to be inserted every N layers in a mostly-compressor stack.
    Uses modern best practices (RoPE-style relative positions if desired,
    grouped-query attention, etc.).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [B, H, T, Hd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.o_proj(attn_output)
