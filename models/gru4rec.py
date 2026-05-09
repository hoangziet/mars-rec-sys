"""
gru4rec.py
==========
GRU4Rec — sequential recommender using GRU.
Reference: Hidasi et al., ICLR 2016.

Architecture:
    - Item embeddings
    - GRU encoder (PackedSequence for variable-length left-padded inputs)
    - Binary cross-entropy loss with negative sampling
    - Scoring via dot product with item embedding table (no Linear head)

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


class GRU4Rec(nn.Module):
    """GRU4Rec model.

    Parameters
    ----------
    n_items:
        Total number of items (padding token = 0 is reserved).
    emb_dim:
        Item embedding dimension.
    hidden_dim:
        GRU hidden state dimension. When hidden_dim != emb_dim a linear
        projection layer maps hidden states into emb_dim for dot-product scoring.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout applied to GRU output (and between layers when num_layers > 1).
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.pad_token = 0

        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        # Project GRU hidden dim -> emb_dim for dot-product scoring
        self.proj: nn.Linear | None = None
        if hidden_dim != emb_dim:
            self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        nn.init.normal_(self.item_emb.weight, std=0.01)

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Run GRU over item sequence, return last valid hidden state (B, emb_dim).

        Uses PackedSequence so padding tokens are ignored by the GRU.
        """
        lengths = (input_seq != self.pad_token).long().sum(dim=1).clamp(min=1)
        x = self.item_emb(input_seq)  # (B, L, emb_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (num_layers, B, hidden_dim)
        last = self.output_dropout(h[-1])  # (B, hidden_dim)

        if self.proj is not None:
            last = self.proj(last)  # (B, emb_dim)

        return last

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss.

        Parameters
        ----------
        input_seq : (B, L) — left-padded input.
        pos_items : (B,)   — ground-truth next item (scalar per sample).
        neg_items : (B,)   — one sampled negative (scalar per sample).
        """
        h = self._encode(input_seq)  # (B, emb_dim)

        pos_emb = self.item_emb(pos_items)  # (B, emb_dim)
        neg_emb = self.item_emb(neg_items)  # (B, emb_dim)

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
        return self._encode(input_seq) @ self.item_emb.weight.T

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for predict — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> GRU4Rec:
    return GRU4Rec(n_items=n_items, **kwargs)
