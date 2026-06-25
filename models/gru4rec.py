"""
gru4rec.py
==========
GRU4Rec — sequential recommender using GRU.
Reference: Hidasi et al., ICLR 2016.

Architecture:
    - Item embeddings with embedding dropout
    - GRU encoder (PackedSequence for variable-length left-padded inputs)
    - Pairwise ranking losses (BPR-max / TOP1) or optional full-catalog CE
    - Scoring via dot product with item embedding table

Training interface
------------------
    loss = model.loss(input_seq, pos_items)

    - input_seq : (B, L)  left-padded item indices
    - pos_items : (B,)    ground-truth next item per sample

Inference interface
-------------------
    scores = model.predict(input_seq)   # (B, n_items+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    """GRU4Rec model with pairwise ranking losses and optional CE fallback.

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
        # Embedding dropout applied before GRU (matches RecBole GRU4Rec)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=False,   # matches RecBole GRU4Rec
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        # Project GRU hidden dim -> emb_dim for dot-product scoring
        self.proj: nn.Linear | None = None
        if hidden_dim != emb_dim:
            self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()   # keep padding row zeroed
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        if self.proj is not None:
            nn.init.xavier_uniform_(self.proj.weight)

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Run GRU over left-padded item sequence, return last-position hidden state.

        With ``bias=False`` and ``item_emb(0) = 0`` (padding token), the GRU
        hidden state stays exactly 0 through all leading padding steps
        (h=0, x=0 → r=z=0.5, n=0 → h_new=0).  Real items appear at the END
        of the left-padded sequence, so ``output[:, -1, :]`` is always the
        hidden state after processing all real items in chronological order.

        PackedSequence must NOT be used here: it packs the first ``lengths``
        timesteps, which for left-padded sequences are all padding — causing
        the GRU to see only zeros and output near-zero hidden states.
        """
        x = self.emb_dropout(self.item_emb(input_seq))  # (B, L, emb_dim)
        output, _ = self.gru(x)                          # (B, L, hidden_dim)
        last = self.output_dropout(output[:, -1, :])     # (B, hidden_dim)

        if self.proj is not None:
            last = self.proj(last)  # (B, emb_dim)

        return last

    def loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-Entropy loss over the full item catalog.

        Pairwise benchmark paths call ``top1_loss`` or ``bpr_max_loss`` directly.
        """
        h = self._encode(input_seq)
        logits = h @ self.item_emb.weight.T
        return F.cross_entropy(logits, pos_items)

    def top1_loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """TOP1 pairwise loss from Hidasi et al. ICLR 2016.

        L = (1/K) * Σ [σ(s_j - s_i) + σ(s_j²)]
        where i=positive, j=negatives.
        """
        h = self._encode(input_seq)
        pos_score = (h * self.item_emb(pos_items)).sum(dim=-1, keepdim=True)  # (B, 1)
        neg_score = (h.unsqueeze(1) * self.item_emb(neg_items)).sum(dim=-1)   # (B, K)
        diff = neg_score - pos_score                                           # (B, K)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(neg_score.pow(2)).mean()
        return loss

    def bpr_max_loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BPR-max pairwise loss from Hidasi & Karatzoglou, DLRS 2016.

        L = -log( Σ softmax_neg(s_j) * σ(s_i - s_j) ) + λ * Σ softmax_neg(s_j) * s_j²
        """
        h = self._encode(input_seq)
        pos_score = (h * self.item_emb(pos_items)).sum(dim=-1, keepdim=True)  # (B, 1)
        neg_score = (h.unsqueeze(1) * self.item_emb(neg_items)).sum(dim=-1)   # (B, K)
        neg_softmax = torch.softmax(neg_score, dim=1)                          # (B, K)
        bpr_term = torch.sigmoid(pos_score - neg_score)                        # (B, K)
        loss_bpr = -torch.log((neg_softmax * bpr_term).sum(dim=1) + 1e-24).mean()
        loss_reg = (neg_softmax * neg_score.pow(2)).sum(dim=1).mean()
        return loss_bpr + 0.5 * loss_reg

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return scores (B, n_items+1) via dot-product with item_emb."""
        return self._encode(input_seq) @ self.item_emb.weight.T

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for predict — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> GRU4Rec:
    return GRU4Rec(n_items=n_items, **kwargs)
