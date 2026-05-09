"""
sasrec.py
=========
SASRec — Self-Attentive Sequential Recommendation.
Reference: Kang & McAuley, ICDM 2018.

Architecture:
    - Item embedding + learnable positional embedding
    - Stacked causal (unidirectional) self-attention blocks
    - Point-wise feed-forward sublayer with Conv1d
    - Binary cross-entropy loss with in-batch negative sampling

Key differences from naive CE-over-vocab:
    - Scoring via dot product between sequence hidden state and item embedding
    - Loss computed at every valid position in the sequence (not just last)
    - Negative items sampled per training step; not a classification head
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
        # x: (B, L, D) — transpose to (B, D, L) for Conv1d
        out = self.relu(self.conv1(x.transpose(1, 2)))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out.transpose(1, 2)  # back to (B, L, D)


class SASRecBlock(nn.Module):
    """One transformer block: causal multi-head self-attention + FFN.

    Pre-LN formulation (LayerNorm applied before each sublayer) following
    the original SASRec implementation.  Residual connections wrap both
    sublayers.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # --- Self-attention sublayer ---
        residual = x
        z = self.ln1(x)
        attn_out, _ = self.attn(
            z, z, z,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(attn_out)

        # --- Feed-forward sublayer ---
        residual = x
        z = self.ln2(x)
        x = residual + self.dropout2(self.ffn(z))

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
        Embedding / hidden dimension ``d``.
    num_heads:
        Number of attention heads (must divide ``hidden_dim`` evenly).
    num_layers:
        Number of stacked SASRec blocks.
    dropout:
        Dropout probability applied throughout.
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
    ) -> None:
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim

        self.n_items = n_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pad_token = 0

        # Embeddings — item index 0 is reserved as the padding token
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

    # ------------------------------------------------------------------
    # Core encoder
    # ------------------------------------------------------------------

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Encode an item sequence into contextualised hidden states.

        Parameters
        ----------
        input_seq:
            Long tensor of shape ``(B, L)`` with item indices.
            Padding positions must contain ``self.pad_token`` (0).

        Returns
        -------
        torch.Tensor
            Hidden states of shape ``(B, L, D)``.
        """
        B, L = input_seq.shape
        pos_ids = torch.arange(L, device=input_seq.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        # Causal mask: upper-triangular True → those positions are masked out
        causal_mask = torch.triu(
            torch.ones(L, L, device=input_seq.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = input_seq == self.pad_token  # (B, L)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        return self.final_ln(x)  # (B, L, D)

    # ------------------------------------------------------------------
    # Training: BCE loss with negative sampling
    # ------------------------------------------------------------------

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss over the full sequence.

        At each position ``t``, the model observes ``input_seq[:, :t+1]``
        and is asked to rank ``pos_items[:, t]`` above ``neg_items[:, t]``.
        Padding positions (pos_items == 0) are excluded from the loss.

        Parameters
        ----------
        input_seq:
            ``(B, L)`` — input item indices (shifted left by one from targets).
        pos_items:
            ``(B, L)`` — ground-truth next items at each position.
        neg_items:
            ``(B, L)`` — one randomly sampled negative item per position.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        hidden = self._encode(input_seq)  # (B, L, D)

        pos_emb = self.item_emb(pos_items)   # (B, L, D)
        neg_emb = self.item_emb(neg_items)   # (B, L, D)

        pos_logits = (hidden * pos_emb).sum(dim=-1)  # (B, L)
        neg_logits = (hidden * neg_emb).sum(dim=-1)  # (B, L)

        # Mask out padding positions
        mask = pos_items != self.pad_token   # (B, L)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits[mask], neg_logits[mask]]),
            torch.cat([
                torch.ones(mask.sum(), device=input_seq.device),
                torch.zeros(mask.sum(), device=input_seq.device),
            ]),
        )
        return loss

    # ------------------------------------------------------------------
    # Inference: score all items from the last valid position
    # ------------------------------------------------------------------

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return logit scores for all items given an input sequence.

        Scores are computed as dot products between the hidden state at
        the last *valid* (non-padding) position and every item embedding.

        Parameters
        ----------
        input_seq:
            ``(B, L)`` — item-index sequence.  Padding on the left.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``(B, n_items + 1)``.
        """
        hidden = self._encode(input_seq)  # (B, L, D)

        # Locate the last non-padding position for each sequence in the batch
        mask = input_seq != self.pad_token          # (B, L)
        has_valid = mask.any(dim=1)                 # (B,)
        last_idx = (
            mask.long().cumsum(dim=1).argmax(dim=1) # rightmost non-zero
            if True
            else torch.zeros(input_seq.size(0), dtype=torch.long)
        )
        # More reliable: find last True by flipping
        last_idx = mask.long().flip(dims=[1]).argmax(dim=1)
        last_idx = (input_seq.size(1) - 1) - last_idx
        last_idx = torch.where(has_valid, last_idx, torch.zeros_like(last_idx))

        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        # (B, D) @ (D, n_items+1) → (B, n_items+1)
        return last_hidden @ self.item_emb.weight.T

    # ------------------------------------------------------------------
    # Convenience alias kept for eval harnesses that call .forward()
    # ------------------------------------------------------------------

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for ``predict`` — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> SASRec:
    return SASRec(n_items=n_items, **kwargs)