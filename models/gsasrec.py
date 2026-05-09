"""
gsasrec.py
==========
gSASRec — Generalised SASRec with gBCE loss and multiple negative sampling.
Reference: Petrov & Macdonald, ACM RecSys 2023.

Key differences from vanilla SASRec:
    1. Loss: Generalised BCE (gBCE) with temperature beta.
       gBCE = -beta * log(sigmoid(pos)) - (1-beta) * mean_k[log(1-sigmoid(neg_k))]
    2. Confidence weighting: each sample scaled by watch_percentage / 100.
    3. Architecture: identical to SASRec (no structural change).

Training interface
------------------
    loss = model.loss(input_seq, pos_items, neg_items, confidence=None)

    - input_seq  : (B, L)         left-padded item indices
    - pos_items  : (B,)           ground-truth next item
    - neg_items  : (B, num_neg)   pre-sampled negatives
    - confidence : (B,) optional  watch_percentage / 100

Inference interface
-------------------
    scores = model.predict(input_seq)   # (B, n_items+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1   = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2   = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x.transpose(1, 2)))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class GSASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.ln1      = nn.LayerNorm(hidden_dim)
        self.attn     = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.ln2      = nn.LayerNorm(hidden_dim)
        self.ffn      = PointWiseFeedForward(hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        residual = x
        z = self.ln1(x)
        attn_out, _ = self.attn(z, z, z,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = residual + self.dropout1(attn_out)
        residual = x
        x = residual + self.dropout2(self.ffn(self.ln2(x)))
        return x


class GSASRec(nn.Module):
    """gSASRec model.

    Parameters
    ----------
    beta:
        gBCE temperature. beta=1 → standard BCE (= SASRec).
        Petrov & Macdonald recommend beta=0.2.
    num_neg:
        Number of negatives per positive expected during training.
        Used only for documentation; actual K is inferred from neg_items shape.
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
        beta: float = 0.2,
        num_neg: int = 1,
    ) -> None:
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim

        self.n_items    = n_items
        self.max_len    = max_len
        self.hidden_dim = hidden_dim
        self.pad_token  = 0
        self.beta       = beta
        self.num_neg    = num_neg

        self.item_emb    = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList(
            [GSASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)

    # ------------------------------------------------------------------

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        pos_ids = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)
        x = self.emb_dropout(self.item_emb(input_seq) + self.pos_emb(pos_ids))
        causal_mask = torch.triu(
            torch.ones(L, L, device=input_seq.device, dtype=torch.bool), diagonal=1
        )
        kpm = input_seq == self.pad_token
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=kpm)
        return self.final_ln(x)

    def _last_hidden(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Extract hidden state at the last non-padding position."""
        hidden = self._encode(input_seq)
        mask = input_seq != self.pad_token
        has_valid = mask.any(dim=1)
        last_idx = mask.long().flip(dims=[1]).argmax(dim=1)
        last_idx = (input_seq.size(1) - 1) - last_idx
        last_idx = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    # ------------------------------------------------------------------
    # Training: gBCE with multiple negatives + confidence weighting
    # ------------------------------------------------------------------

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generalised BCE loss.

        L = beta * BCE(pos=1) + (1-beta) * mean_k[BCE(neg_k=0)]
        """
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)   # (B,) → (B, 1)

        h = self._last_hidden(input_seq)          # (B, D)

        pos_emb = self.item_emb(pos_items)        # (B, D)
        neg_emb = self.item_emb(neg_items)        # (B, K, D)

        pos_score = (h * pos_emb).sum(dim=-1)                             # (B,)
        neg_score = torch.bmm(neg_emb, h.unsqueeze(-1)).squeeze(-1)       # (B, K)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score), reduction="none"
        )  # (B,)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score), reduction="none"
        ).mean(dim=1)  # (B,)

        gbce = self.beta * pos_loss + (1.0 - self.beta) * neg_loss        # (B,)

        if confidence is not None:
            gbce = gbce * torch.clamp(confidence.float(), min=0.0)

        return gbce.mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return scores (B, n_items+1) over full item vocabulary."""
        return self._last_hidden(input_seq) @ self.item_emb.weight.T

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> GSASRec:
    return GSASRec(n_items=n_items, **kwargs)