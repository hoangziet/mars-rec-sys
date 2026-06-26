"""
bert4rec.py
===========
BERT4Rec — masked item prediction for sequential recommendation.

Architecture:
    - Bidirectional Transformer encoder
    - Input LayerNorm applied after embedding sum (before dropout), matching BERT paper
    - Masked item modeling (15% random mask during training)
    - MLM prediction head: Linear → GELU → LayerNorm (BERT-style)
    - Weight-tied output projection with per-item bias
    - Special mask_token = n_items + 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_len=50, hidden_dim=64, num_heads=2,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.pad_token = 0
        self.mask_token = n_items + 1
        self.vocab_size = n_items + 2

        self.item_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        # Input LayerNorm applied after embedding sum, before dropout.
        # Matches BERT paper and RecBole implementation.
        self.input_ln = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM prediction head: Linear → GELU → LayerNorm.
        # Transforms Transformer output before projecting to vocab logits.
        # Matches BERT paper "masked language model head" and RecBole BERT4Rec.
        self.pred_ffn = nn.Linear(hidden_dim, hidden_dim)
        self.pred_ln  = nn.LayerNorm(hidden_dim, eps=1e-12)

        # Output projection shares weights with item_embedding (weight tying).
        self.out_bias = nn.Parameter(torch.zeros(n_items))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.pred_ffn.weight, std=0.02)
        nn.init.zeros_(self.pred_ffn.bias)
        nn.init.ones_(self.input_ln.weight)
        nn.init.zeros_(self.input_ln.bias)
        nn.init.ones_(self.pred_ln.weight)
        nn.init.zeros_(self.pred_ln.bias)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, input_seq):
        B, L = input_seq.shape
        pos_ids = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)
        x = self.item_embedding(input_seq) + self.pos_embedding(pos_ids)
        x = self.input_ln(x)   # LayerNorm before dropout (BERT convention)
        x = self.dropout(x)

        padding_mask = (input_seq == self.pad_token)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # MLM prediction head: transform hidden states before weight-tied projection
        x = F.gelu(self.pred_ffn(x))
        x = self.pred_ln(x)

        # Output logits cover only padding (col 0) + real items (cols 1..n_items).
        # The MASK token (index n_items+1) is an input-only token, not an output class.
        real_item_weight = self.item_embedding.weight[1 : self.n_items + 1]
        real_item_logits = F.linear(x, real_item_weight, self.out_bias)

        padding_logits = torch.zeros(
            *real_item_logits.shape[:-1], 1,
            dtype=real_item_logits.dtype, device=real_item_logits.device,
        )
        return torch.nan_to_num(
            torch.cat([padding_logits, real_item_logits], dim=-1),
            nan=float("-inf"),
        )

    def loss(self, input_seq, labels):
        logits = self.forward(input_seq)

        valid_mask = labels.ne(self.pad_token)
        if not valid_mask.any():
            return torch.zeros((), device=input_seq.device, requires_grad=True)

        valid_labels = labels[valid_mask]
        if torch.any((valid_labels < 1) | (valid_labels > self.n_items)):
            raise ValueError(
                f"BERT4Rec labels must be real item IDs in [1, {self.n_items}]"
            )

        real_item_logits = logits[..., 1:][valid_mask]
        zero_based_labels = valid_labels - 1
        return F.cross_entropy(real_item_logits, zero_based_labels)


def get_model(n_items, **kwargs):
    return BERT4Rec(n_items=n_items, **kwargs)
