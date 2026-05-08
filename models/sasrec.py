"""
sasrec.py
=========
SASRec — Self-Attentive Sequential Recommendation.

Architecture:
    - Item + positional embeddings
    - Stacked causal self-attention blocks
    - FFN sublayer with Conv1d
    - Output logits over full item vocabulary
"""

import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + x


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        z = self.ln1(x)
        attn_out, _ = self.attn(
            z, z, z,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        z = self.ln2(x)
        x = self.ffn(z)
        return x


class SASRec(nn.Module):
    def __init__(
        self, n_items, max_len=50, hidden_dim=64,
        num_heads=2, num_layers=2, dropout=0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pad_token = 0

        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, n_items + 1)

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.pos_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, input_seq, mask=None):
        B, L = input_seq.shape
        pos_ids = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)
        x = self.item_embedding(input_seq) + self.pos_embedding(pos_ids)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones((L, L), device=input_seq.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = input_seq == self.pad_token

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.final_ln(x)

        if mask is None:
            last_hidden = x[:, -1, :]
        else:
            seq_lens = mask.sum(dim=1) - 1
            seq_lens = torch.clamp(seq_lens, min=0)
            last_hidden = x[torch.arange(B, device=x.device), seq_lens]

        return self.output(last_hidden)


def get_model(n_items, **kwargs):
    return SASRec(n_items=n_items, **kwargs)
