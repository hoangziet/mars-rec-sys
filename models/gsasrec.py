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
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x.transpose(1, 2)))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class GSASRecBlock(nn.Module):
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

    def forward(self, x, attn_mask=None, padding_mask=None):
        # Same as SASRec: causal mask only, no explicit padding zeroing.
        # padding_mask is accepted for API compatibility but is not applied.
        if self.norm_first:
            # Pre-LN
            residual = x
            z = self.ln1(x)
            attn_out, _ = self.attn(z, z, z, attn_mask=attn_mask)
            x = residual + self.dropout1(attn_out)
            residual = x
            x = residual + self.dropout2(self.ffn(self.ln2(x)))
        else:
            # Post-LN
            attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
            x = self.ln1(x + self.dropout1(attn_out))
            x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x


class GSASRec(nn.Module):
    """gSASRec model.

    Parameters
    ----------
    t:
        gBCE temperature in [0, 1].  t=0 → standard BCE (= SASRec).
        t=1 → fully sampling-corrected logits.
        Reference default (Petrov & Macdonald, RecSys 2023): t=0.5.
    num_neg:
        Number of negatives per positive sampled during training.
        Reference default: 32.  The effective beta is computed as:
          alpha = num_neg / (n_items - 1)
          beta  = alpha * ((1 - 1/alpha) * t + 1/alpha)
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
        t: float = 0.5,
        num_neg: int = 32,
        pos_smoothing: float = 0.0,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim

        self.n_items    = n_items
        self.max_len    = max_len
        self.hidden_dim = hidden_dim
        self.pad_token  = 0
        self.t          = t
        self.num_neg    = num_neg
        self.pos_smoothing = pos_smoothing

        self.item_emb    = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        # 1-indexed positions; index 0 (padding_idx=0) → zero vector for padding tokens.
        self.pos_emb     = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList(
            [GSASRecBlock(hidden_dim, num_heads, dropout, norm_first)
             for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  std=0.01)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, L = input_seq.shape
        # 1-indexed positions; padding tokens get pos_id = 0 → zero pos embedding.
        pos_ids = torch.arange(1, L + 1, device=input_seq.device).unsqueeze(0).expand(B, L)
        pos_ids = pos_ids * (input_seq != self.pad_token).long()

        if self.training and self.pos_smoothing > 0:
            noise = torch.randn(B, L, device=input_seq.device) * self.pos_smoothing
            pos_ids = (pos_ids.float() + noise).clamp(min=0.0).long()

        x = self.item_emb(input_seq) * (self.hidden_dim ** 0.5) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(L, L, device=input_seq.device, dtype=torch.bool), diagonal=1
        )
        padding_mask = input_seq == self.pad_token
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, padding_mask=padding_mask)
        return self.final_ln(x)

    def _last_hidden(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Extract hidden state at the last non-padding position. Returns (B, D)."""
        hidden    = self._encode(input_seq)
        mask      = input_seq != self.pad_token
        has_valid = mask.any(dim=1)
        last_idx  = mask.long().flip(dims=[1]).argmax(dim=1)
        last_idx  = (input_seq.size(1) - 1) - last_idx
        last_idx  = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    # ------------------------------------------------------------------
    # Training: gBCE with multiple negatives + confidence weighting
    # ------------------------------------------------------------------

    def _gbce_transform_pos(self, pos_score: torch.Tensor, K: int) -> torch.Tensor:
        """Apply gBCE logit transformation to positive scores.

        Reference: Petrov & Macdonald, RecSys 2023.
          alpha = K / (n_items - 1)
          beta  = alpha * ((1 - 1/alpha) * t + 1/alpha)
          pos_score_t = log(1 / (sigmoid(pos_score)^{-beta} - 1))

        At t=0  → beta=1 → pos_score_t == pos_score (standard BCE).
        At t=0.5 and K=32, n_items=2300 → beta ≈ 0.51.
        """
        alpha = K / max(self.n_items - 1, 1)
        beta  = alpha * ((1.0 - 1.0 / alpha) * self.t + 1.0 / alpha)
        eps   = 1e-10
        # Use float64 for numerical stability (same as reference implementation)
        p     = torch.sigmoid(pos_score.double())
        p_adj = p.pow(-beta).clamp(min=1.0 + eps)
        return torch.log((1.0 / (p_adj - 1.0)).clamp(min=eps)).float()

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generalised BCE loss at the last valid sequence position.

        Parameters
        ----------
        input_seq  : (B, L)
        pos_items  : (B,) or (B, L) — if 2-D, last valid position is extracted.
        neg_items  : (B,), (B, K), or (B, L) — reshaped accordingly.
        confidence : (B,) optional watch_percentage / 100.
        """
        # Normalise pos/neg to (B,) and (B, K)
        if pos_items.dim() == 2 and pos_items.size(1) > 1:
            mask      = pos_items != self.pad_token
            has_valid = mask.any(dim=1)
            last_idx  = mask.long().flip(dims=[1]).argmax(dim=1)
            last_idx  = (pos_items.size(1) - 1) - last_idx
            last_idx  = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))

            B      = input_seq.size(0)
            arange = torch.arange(B, device=input_seq.device)
            pos_items = pos_items[arange, last_idx]              # (B,)
            neg_items = neg_items[arange, last_idx].unsqueeze(1) # (B, 1)
        elif neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)                   # (B,) -> (B, 1)

        h = self._last_hidden(input_seq)                                      # (B, D)
        pos_emb   = self.item_emb(pos_items)                                  # (B, D)
        neg_emb   = self.item_emb(neg_items)                                  # (B, K, D)
        pos_score = (h * pos_emb).sum(dim=-1)                                 # (B,)
        neg_score = torch.bmm(neg_emb, h.unsqueeze(-1)).squeeze(-1)           # (B, K)

        K = neg_items.size(1)
        pos_score_t = self._gbce_transform_pos(pos_score, K)                  # (B,)

        # Concatenate transformed positive + negatives → (B, 1+K)
        all_scores = torch.cat([pos_score_t.unsqueeze(1), neg_score], dim=1)
        all_labels = torch.zeros_like(all_scores)
        all_labels[:, 0] = 1.0

        # Loss per sample (B,) — mean over 1+K logits per sample
        loss_per_sample = F.binary_cross_entropy_with_logits(
            all_scores, all_labels, reduction="none"
        ).mean(dim=1)                                                          # (B,)

        if confidence is not None:
            loss_per_sample = loss_per_sample * torch.clamp(confidence.float(), min=0.0)

        return loss_per_sample.mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return scores (B, n_items+1) via dot-product with item_emb."""
        return self._last_hidden(input_seq) @ self.item_emb.weight.T

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for predict — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> GSASRec:
    return GSASRec(n_items=n_items, **kwargs)
