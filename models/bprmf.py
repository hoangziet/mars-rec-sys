"""
bprmf.py
========
Bayesian Personalized Ranking Matrix Factorization (BPR-MF).

Architecture:
    - User and item embeddings
    - Score = dot(user_emb, item_emb)
    - BPR pairwise loss with optional L2 regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.n_items = n_items
        self.user_embedding = nn.Embedding(n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.user_emb = self.user_embedding
        self.item_emb = self.item_embedding
        # Xavier uniform matches the reference implementation and is standard for MF.
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # Keep padding row zeroed out after Xavier fills it.
        with torch.no_grad():
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()

    def forward(self, users, pos_items, neg_items):
        u = self.user_embedding(users)
        p = self.item_embedding(pos_items)
        n = self.item_embedding(neg_items)
        return (u * p).sum(dim=-1), (u * n).sum(dim=-1)

    def loss(self, users, pos_items, neg_items, reg_lambda=1e-4):
        u = self.user_embedding(users)
        p = self.item_embedding(pos_items)
        n = self.item_embedding(neg_items)
        pos_scores = (u * p).sum(dim=-1)
        neg_scores = (u * n).sum(dim=-1)
        return bpr_loss(pos_scores, neg_scores, reg_lambda=reg_lambda,
                        u_emb=u, p_emb=p, n_emb=n)

    def score(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u.unsqueeze(1) * i).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores, reg_lambda=1e-4, model=None,
             u_emb=None, p_emb=None, n_emb=None):
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    if reg_lambda > 0 and u_emb is not None:
        # Regularize only the batch-fetched embeddings (not the full table),
        # matching the reference implementation. Full-table L2 is O(N/B)× stronger.
        reg = (u_emb.pow(2).sum() + p_emb.pow(2).sum() + n_emb.pow(2).sum()) / (2 * u_emb.size(0))
        loss = loss + reg_lambda * reg
    return loss


def get_model(n_users, n_items, **kwargs):
    return BPRMF(n_users=n_users, n_items=n_items, **kwargs)
