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


def _compact_left_padded_batch(input_seq, mask):
    """Move contiguous valid suffix to front for left-padded 0*1* masks."""
    mask = mask.bool()

    if mask.ndim != 2 or input_seq.ndim != 2:
        raise ValueError("GRU4Rec expects 2D input_seq and mask tensors")
    if input_seq.shape != mask.shape:
        raise ValueError("GRU4Rec mask shape must match input_seq shape")

    lengths = mask.long().sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("GRU4Rec mask cannot contain all-padding row; each row needs at least one valid token")

    seq_len = input_seq.size(1)
    base_idx = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
    pad_counts = seq_len - lengths
    expected_mask = base_idx >= pad_counts.unsqueeze(1)
    if not torch.equal(mask, expected_mask):
        raise ValueError("GRU4Rec supports only contiguous left padding masks (0*1*)")

    compact_idx = (base_idx + pad_counts.unsqueeze(1)).clamp(max=seq_len - 1)
    valid_steps = base_idx < lengths.unsqueeze(1)
    compact_idx = torch.where(valid_steps, compact_idx, torch.zeros_like(compact_idx))
    compact_seq = torch.gather(input_seq, 1, compact_idx)
    return compact_seq, lengths


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
        if mask is not None:
            input_seq, lengths = _compact_left_padded_batch(input_seq, mask)

        x = self.item_embedding(input_seq)

        if mask is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h = self.gru(packed)
        else:
            _, h = self.gru(x)

        last = self.dropout(h[-1])
        return self.output(last)


def get_model(n_items, **kwargs):
    return GRU4Rec(n_items=n_items, **kwargs)
