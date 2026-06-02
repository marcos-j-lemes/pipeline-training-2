import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import json
    # Prepare tokenizer and dataset binary for training
import os
import numpy as np

import torch.utils.Dataloader as DataLoader
from tokenizador import tokenize_dataset, train_tokenizer

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

def generate(model, prompt_ids, max_new_tokens, seq_len, temperature, valid_token_ids, device):
    model.eval()
    generated = prompt_ids.copy()
    input_ids = torch.tensor(generated[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1] / temperature
            next_token_logits[~torch.isin(torch.arange(lognext_token_logits.size(-1), device=device), valid_token_ids)] = -float('inf')
            next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).item()
            generated.append(next_token_id)

            input_ids = torch.tensor(generated[-seq_len:], dtype=torch.long, device=device).unsqueeze(0)

    return generated


if __name__ == "__main__":

    # Settings

    config ={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vocab_size": 5000,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "max_seq_len": 32,
        "ffn_dim": 128,
    }    





    # Initialize the model

    model = Encoder_model(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        ffn_dim=config["ffn_dim"],
    ).to(config["device"])

    print(model)