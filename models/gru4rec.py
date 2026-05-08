"""
gru4rec.py
==========
GRU4Rec — sequential recommender using GRU.

Architecture:
    - Item embeddings
    - GRU encoder
    - Final hidden state → linear output → logits
"""

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, n_items + 1)

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, input_seq, mask=None):
        x = self.item_embedding(input_seq)
        _, h = self.gru(x)
        last = self.dropout(h[-1])
        return self.output(last)


def get_model(n_items, **kwargs):
    return GRU4Rec(n_items=n_items, **kwargs)
