"""
gsasrec.py
==========
gSASRec — Generalised SASRec with gBCE loss and multiple negative sampling.
Reference: Petrov & Macdonald, ACM RecSys 2023.

Key differences from vanilla SASRec:
    1. Loss: Generalised BCE (gBCE) with temperature beta.
       gBCE = -beta * log(sigmoid(pos)) - (1-beta) * mean_k[log(1-sigmoid(neg_k))]
    2. Architecture: identical to SASRec (no structural change).

Training interface
------------------
    loss = model.loss(input_seq, pos_items, neg_items)

    - input_seq  : (B, L)         left-padded item indices
    - pos_items  : (B,)           ground-truth next item
    - neg_items  : (B, num_neg)   pre-sampled negatives

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

    Reference: Petrov & Macdonald, ACM RecSys 2023.

    Implementation note: the gBCE loss uses the score transformation
    pos_score_t = log(1 / (sigmoid(pos)^-beta - 1)) as in the reference.
    The "pos_smoothing" parameter is a project-specific extension.

    Parameters
    ----------
    t:
        gBCE temperature in [0, 1].  t=0 → standard BCE (= SASRec).
        t=1 → fully sampling-corrected logits.
        Project default: t=0.5 (see configs/model/gsasrec.yaml).
    num_neg:
        Number of negatives per positive sampled during training.
        Project default: 32 (see configs/model/gsasrec.yaml).
        The original paper's experiments use K=256; this project
        uses K=32 to fit within memory budget. Do NOT confuse the
        project default with the paper default.
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
        item_encoder=None,
        reuse_item_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim

        self.n_items    = n_items
        self.max_len    = max_len
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.pad_token  = 0
        self.t          = t
        self.num_neg    = num_neg
        self.pos_smoothing = pos_smoothing

        if item_encoder is not None:
            self.item_emb = item_encoder
            self.item_emb_is_embedding = False
        else:
            self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
            nn.init.normal_(self.item_emb.weight, std=0.01)
            with torch.no_grad():
                self.item_emb.weight[0].zero_()
            self.item_emb_is_embedding = True

        self.reuse_item_embeddings = reuse_item_embeddings
        if self.reuse_item_embeddings:
            if item_encoder is not None:
                raise ValueError("reuse_item_embeddings=True conflicts with item_encoder")
            self.output_emb = self.item_emb
        else:
            self.output_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
            nn.init.normal_(self.output_emb.weight, std=0.01)
            with torch.no_grad():
                self.output_emb.weight[0].zero_()

        # 1-indexed positions; index 0 (padding_idx=0) → zero vector for padding tokens.
        self.pos_emb     = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList(
            [GSASRecBlock(hidden_dim, num_heads, dropout, norm_first)
             for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_dim, eps=1e-8)

        nn.init.normal_(self.pos_emb.weight,  std=0.01)
        with torch.no_grad():
            self.pos_emb.weight[0].zero_()

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
            # Only add noise at non-padding positions (pos_ids > 0).
            # Padding has pos_ids=0 and pos_emb(0)=0 — keep it that way.
            noise = noise * (pos_ids > 0).float()
            pos_ids = (pos_ids.float() + noise).clamp(min=0.0, max=float(L)).long()

        x = self.item_emb(input_seq) * (self.hidden_dim ** 0.5) + self.pos_emb(pos_ids)

        # Zero hidden at padding positions before attention (unifies behavior with ItemEncoder)
        pad_hidden_mask = (input_seq == self.pad_token).unsqueeze(-1)
        x = x.masked_fill(pad_hidden_mask, 0.0)

        x = self.emb_dropout(x)

        # RecBole-style additive attention mask: causal + padding-key exclusion.
        # Padding queries are allowed to attend their own position (self-diagonal)
        # so softmax never sees an entirely masked row (which would produce NaN).
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=input_seq.device)).unsqueeze(0)   # [1, L, L]
        valid_keys = input_seq.ne(self.pad_token).unsqueeze(1)                                           # [B, 1, L]
        allowed = causal & valid_keys                                                                    # [B, L, L]
        pad_queries = input_seq.eq(self.pad_token).unsqueeze(-1)                                         # [B, L, 1]
        diag = torch.eye(L, dtype=torch.bool, device=input_seq.device).unsqueeze(0)                      # [1, L, L]
        allowed = allowed | (pad_queries & diag)                                                         # [B, L, L]
        add_mask = torch.zeros(allowed.shape, dtype=x.dtype, device=input_seq.device)
        add_mask = add_mask.masked_fill(~allowed, -10000.0)
        add_mask = add_mask.repeat_interleave(self.num_heads, dim=0)                                     # [B*H, L, L]

        for block in self.blocks:
            x = block(x, attn_mask=add_mask)
            x = x.masked_fill(pad_hidden_mask, 0.0)
        x = self.final_ln(x)
        x = x.masked_fill(pad_hidden_mask, 0.0)
        return torch.nan_to_num(x, nan=0.0)

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
    # Training: gBCE with multiple negatives
    # ------------------------------------------------------------------

    def _gbce_transform_pos(self, pos_score: torch.Tensor, num_negatives: int) -> torch.Tensor:
        """Apply gBCE logit transformation to positive scores.

        Reference: Petrov & Macdonald, RecSys 2023.
          alpha = K / (n_items - 1)
          beta  = alpha * ((1 - 1/alpha) * t + 1/alpha)
          pos_score_t = log(1 / (sigmoid(pos_score)^{-beta} - 1))

        At t=0  → beta=1 → pos_score_t == pos_score (standard BCE).
        At t=0.5 and K=32, n_items=2300 → beta ≈ 0.51.
        """
        alpha = num_negatives / max(self.n_items - 1, 1)
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
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Generalised BCE loss at the last valid sequence position.

        Parameters
        ----------
        input_seq  : (B, L)
        pos_items  : (B,) or (B, L) — if 2-D, last valid position is extracted.
        neg_items  : (B,), (B, K), or (B, L) — reshaped accordingly.
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
        pos_emb   = self.output_emb(pos_items)                                 # (B, D)
        neg_emb   = self.output_emb(neg_items)                                 # (B, K, D)
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

        if reduction == "none":
            return loss_per_sample
        return loss_per_sample.mean()

    def _negative_sampling_rate(self, num_negatives: int) -> float:
        if self.n_items <= 1:
            raise ValueError("gBCE requires at least two catalogue items")
        if not (1 <= num_negatives <= self.n_items - 1):
            raise ValueError(f"num_negatives must be in [1, {self.n_items - 1}]")
        return num_negatives / (self.n_items - 1)

    def sequence_loss(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        loss_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if neg_items.dim() != 3:
            raise ValueError("neg_items must have shape [B, L, K]")
        if reduction not in ("mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")

        hidden = self._encode(input_seq)

        pos_emb = self.output_emb(pos_items)
        neg_emb = self.output_emb(neg_items)

        pos_score = (hidden * pos_emb).sum(dim=-1)
        neg_score = torch.einsum("bld,blkd->blk", hidden, neg_emb)

        K = neg_items.size(-1)
        pos_score_t = self._gbce_transform_pos(pos_score, K)

        logits = torch.cat([pos_score_t.unsqueeze(-1), neg_score], dim=-1)
        labels = torch.zeros_like(logits)
        labels[..., 0] = 1.0

        per_position = (
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            .mean(dim=-1)
            .masked_fill(~loss_mask, 0.0)
        )

        if reduction == "none":
            return per_position
        return per_position.sum() / loss_mask.sum().clamp_min(1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Return scores (B, n_items+1) via dot-product with output embedding."""
        h = self._last_hidden(input_seq)  # (B, D)
        if hasattr(self.output_emb, "weight"):
            scores = h @ self.output_emb.weight.T
        else:
            all_ids = torch.arange(self.n_items + 1, device=input_seq.device)
            all_embs = self.output_emb(all_ids)
            scores = h @ all_embs.T
        return torch.nan_to_num(scores, nan=float("-inf"))

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Alias for predict — returns scores over full item vocab."""
        return self.predict(input_seq)


def get_model(n_items: int, **kwargs) -> GSASRec:
    return GSASRec(n_items=n_items, **kwargs)
