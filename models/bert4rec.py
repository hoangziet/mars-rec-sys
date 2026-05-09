"""
bert4rec.py
===========
BERT4Rec — masked item prediction for sequential recommendation.

Architecture:
    - Bidirectional Transformer encoder
    - Masked item modeling (15% random mask during training)
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, self.vocab_size)

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.pos_embedding.weight, std=0.01)

    def forward(self, input_seq):
        B, L = input_seq.shape
        pos_ids = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)
        x = self.item_embedding(input_seq) + self.pos_embedding(pos_ids)
        x = self.dropout(x)

        padding_mask = (input_seq == self.pad_token)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.out(x)

    def loss(self, input_seq, labels):
        logits = self.forward(input_seq)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
        )


def get_model(n_items, **kwargs):
    return BERT4Rec(n_items=n_items, **kwargs)
