# -*- coding: utf-8 -*-
"""

Teste rapido de um modelo

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json


class TextDataset(Dataset):
  def __init__(self, tokens, seq_len):
    self.tokens = tokens
    self.seq_len = seq_len

  def __len__(self):
    return len(self.tokens) - self.seq_len

  def __getitem__(self, idx):
    x = self.tokens[idx         : idx + self.seq_len]
    y = self.tokens[idx + 1     : idx + self.seq_len + 1]
    return x, y


class Encoder_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads,
                 num_layers, max_seq_len, ffn_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=0.1,  # dropout ativo agora
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb(
            torch.arange(seq_len, device=tokens.device)
        ).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tokens.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.linear_head(x)
