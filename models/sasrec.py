"""
sasrec.py
=========
SASRec — Self-Attentive Sequential Recommendation.
Reference: Kang & McAuley, ICDM 2018.

Architecture:
    - Item embedding (scaled by sqrt(d)) + 1-indexed learnable positional embedding
    - Stacked causal (unidirectional) self-attention blocks
    - Point-wise feed-forward sublayer with Conv1d
    - Binary cross-entropy loss with in-batch negative sampling

Training interface
------------------
    loss = model.loss(input_seq, pos_items, neg_items)

    - input_seq : (B, L)  left-padded item indices
    - pos_items : (B,)    ground-truth next item per sample (scalar)
    - neg_items : (B,)    one randomly sampled negative per sample (scalar)

Inference interface
-------------------
    scores = model.predict(input_seq)   # (B, n_items+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    """Position-wise FFN using two Conv1d layers (equivalent to two linear
    projections applied independently at each sequence position)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> transpose to (B, D, L) for Conv1d
        out = self.relu(self.conv1(x.transpose(1, 2)))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out.transpose(1, 2)  # back to (B, L, D)


class SASRecBlock(nn.Module):
    """One transformer block: causal multi-head self-attention + FFN.

    Supports both Pre-LN (norm_first=True) and Post-LN (norm_first=False).
    Pre-LN applies LayerNorm before each sublayer; Post-LN applies it after.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2,
                 norm_first: bool = True) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            # Pre-LN
            residual = x
            z = self.ln1(x)
            attn_out, _ = self.attn(z, z, z, attn_mask=attn_mask, key_padding_mask=padding_mask)
            x = residual + self.dropout1(attn_out)
            residual = x
            x = residual + self.dropout2(self.ffn(self.ln2(x)))
        else:
            # Post-LN
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=padding_mask)
            x = self.ln1(x + self.dropout1(attn_out))
            x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x


class SASRec(nn.Module):
    """SASRec model.

    Parameters
    ----------
    n_items:
        Total number of items (padding token = 0 is reserved).
    max_len:
        Maximum sequence length for positional embeddings.
    hidden_dim:
        Embedding / hidden dimension d.
    emb_dim:
        Alias for hidden_dim for backward compatibility.
    num_heads:
        Number of attention heads (must divide hidden_dim evenly).
    num_layers:
        Number of stacked SASRec blocks.
    dropout:
        Dropout probability applied throughout.
    norm_first:
        If True, use Pre-LN (LayerNorm before sublayers); if False, Post-LN.
    """

    def __init__(
        self,
        n_items: int,
        max_len: int = 50,
        hidden_dim: int = 64,
        emb_dim: int | None = None,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim

        self.n_items = n_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pad_token = 0

        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        # 1-indexed positional embeddings: index 0 is reserved (padding_idx=0 → zero vector).
        # Padding positions in the input get pos index 0, so their positional embedding
        # is always the zero vector — preventing any non-zero signal at padding positions.
        self.pos_emb = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim, num_heads, dropout, norm_first)
             for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Encode item sequence -> contextualised hidden states (B, L, D)."""
        B, L = input_seq.shape

        # 1-indexed positions; padding positions (input == 0) get pos_id = 0 → zero vector.
        pos_ids = torch.arange(1, L + 1, device=input_seq.device).unsqueeze(0).expand(B, L)
        pos_ids = pos_ids * (input_seq != self.pad_token).long()

        # Scale item embeddings by sqrt(d) to balance magnitude with positional embeddings.
        x = self.item_emb(input_seq) * (self.hidden_dim ** 0.5) + self.pos_emb(pos_ids)

        pad_hidden_mask = (input_seq == self.pad_token).unsqueeze(-1)
        x = x.masked_fill(pad_hidden_mask, 0.0)

        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(L, L, device=input_seq.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = input_seq == self.pad_token  # (B, L) True for padding

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, padding_mask=padding_mask)
            # Re-zero at padding positions: when a padded query has all keys
            # masked, attention softmax becomes all -inf → NaN, which then
            # propagates through the residual to valid positions in later blocks.
            x = x.masked_fill(pad_hidden_mask, 0.0)

        x = self.final_ln(x)
        x = x.masked_fill(pad_hidden_mask, 0.0)
        return x

    def _last_hidden(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Extract hidden state at the last non-padding position. Returns (B, D)."""
        hidden = self._encode(input_seq)
        mask = input_seq != self.pad_token
        has_valid = mask.any(dim=1)
        last_idx = mask.long().flip(dims=[1]).argmax(dim=1)
        last_idx = (input_seq.size(1) - 1) - last_idx
        last_idx = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss at the last valid sequence position.

        Parameters
        ----------
        input_seq : (B, L) — left-padded input.
        pos_items : (B,)   — ground-truth next item (scalar per sample).
        neg_items : (B,)   — one sampled negative (scalar per sample).
        """
        h = self._last_hidden(input_seq)  # (B, D)

        pos_emb = self.item_emb(pos_items)  # (B, D)
        neg_emb = self.item_emb(neg_items)  # (B, D)

        pos_logits = (h * pos_emb).sum(dim=-1)  # (B,)
        neg_logits = (h * neg_emb).sum(dim=-1)  # (B,)

        return F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]),
            torch.cat(
                [
                    torch.ones(pos_logits.size(0), device=input_seq.device),
                    torch.zeros(neg_logits.size(0), device=input_seq.device),
                ]
            ),
        )

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return scores (B, n_items+1) via dot-product with item_emb."""
        return self._last_hidden(input_seq) @ self.item_emb.weight.T

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for predict — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> SASRec:
    return SASRec(n_items=n_items, **kwargs)
