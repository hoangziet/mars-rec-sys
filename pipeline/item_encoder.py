from __future__ import annotations

import torch
import torch.nn as nn


class ItemEncoder(nn.Module):
    """Fuses item ID embedding with structured metadata and text embeddings.

    Architecture:
        ID embedding (H) + structured (7 × H//4) + text (H//2)
        → Concatenation → Linear(total → H) → LayerNorm(H)
    """

    def __init__(
        self,
        n_items: int,
        hidden_dim: int,
        metadata_tensors: dict[str, torch.Tensor] | None = None,
        text_embeddings: torch.Tensor | None = None,
        use_structured: bool = True,
        use_text: bool = True,
    ):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.use_structured = use_structured and metadata_tensors is not None
        self.use_text = use_text and text_embeddings is not None

        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)

        struct_dim = 0
        if self.use_structured:
            self._struct_fields = list(metadata_tensors.keys())
            self._sub_dim = hidden_dim // 4

            for field, tensor in metadata_tensors.items():
                self.register_buffer(f"meta_{field}", tensor)

            for field in self._struct_fields:
                tensor = metadata_tensors[field]
                if tensor.dim() == 1:
                    if field == "duration":
                        setattr(self, f"proj_{field}", nn.Linear(1, self._sub_dim))
                    else:
                        vocab_size = int(tensor.max().item()) + 1
                        setattr(self, f"emb_{field}", nn.Embedding(vocab_size, self._sub_dim, padding_idx=0))
                else:
                    vocab_size = int(tensor.max().item()) + 1
                    setattr(self, f"emb_{field}", nn.Embedding(vocab_size, self._sub_dim, padding_idx=0))

            struct_dim = len(self._struct_fields) * self._sub_dim

        text_dim = 0
        if self.use_text:
            self._text_dim = hidden_dim // 2
            text_dim = self._text_dim
            self.register_buffer("text_emb", text_embeddings)
            self.text_proj = nn.Linear(text_embeddings.size(1), self._text_dim)

        total_dim = hidden_dim + struct_dim + text_dim
        if total_dim > hidden_dim:
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.fusion = None

        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        h = self.item_emb(item_ids)

        parts = [h]

        if self.use_structured:
            for field in self._struct_fields:
                tensor = getattr(self, f"meta_{field}")
                field_vals = tensor[item_ids]

                if field == "duration":
                    proj = getattr(self, f"proj_{field}")
                    encoded = proj(field_vals.unsqueeze(-1))
                elif tensor.dim() == 1:
                    emb = getattr(self, f"emb_{field}")
                    encoded = emb(field_vals)
                else:
                    emb = getattr(self, f"emb_{field}")
                    embedded = emb(field_vals)
                    mask = (field_vals != 0).unsqueeze(-1).float()
                    encoded = (embedded * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1)

                parts.append(encoded)

        if self.use_text:
            text_vecs = self.text_emb[item_ids]
            parts.append(self.text_proj(text_vecs))

        if len(parts) > 1:
            h = torch.cat(parts, dim=-1)
            if self.fusion is not None:
                h = self.fusion(h)

        pad_mask = (item_ids == 0).unsqueeze(-1)
        h = h.masked_fill(pad_mask, 0.0)

        return h
