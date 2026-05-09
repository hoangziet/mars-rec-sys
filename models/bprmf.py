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


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.user_emb = self.user_embedding
        self.item_emb = self.item_embedding
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, pos_items, neg_items):
        u = self.user_embedding(users)
        p = self.item_embedding(pos_items)
        n = self.item_embedding(neg_items)
        return (u * p).sum(dim=-1), (u * n).sum(dim=-1)

    def loss(self, users, pos_items, neg_items, reg_lambda=1e-4):
        pos_scores, neg_scores = self.forward(users, pos_items, neg_items)
        return bpr_loss(pos_scores, neg_scores, reg_lambda=reg_lambda, model=self)

    def score(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u.unsqueeze(1) * i).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores, reg_lambda=1e-4, model=None):
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    if model is not None:
        reg = (
            model.user_embedding.weight.norm(2).pow(2)
            + model.item_embedding.weight.norm(2).pow(2)
        ) / model.user_embedding.weight.numel()
        loss = loss + reg_lambda * reg
    return loss


def get_model(n_users, n_items, **kwargs):
    return BPRMF(n_users=n_users, n_items=n_items, **kwargs)
