from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_vocab(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf8") as file:
        raw_vocab = json.load(file)
    return {int(token_id): token for token_id, token in raw_vocab.items()}


def build_reverse_vocab(vocab: dict[int, str]) -> dict[str, int]:
    return {token: token_id for token_id, token in vocab.items()}


def encode_with_vocab(text: str, reverse_vocab: dict[str, int]) -> list[int]:
    ids: list[int] = []
    tokens_by_size = sorted(reverse_vocab, key=len, reverse=True)
    index = 0
    while index < len(text):
        match = None
        for token in tokens_by_size:
            if token and text.startswith(token, index):
                match = token
                break

        if match is not None:
            ids.append(reverse_vocab[match])
            index += len(match)
            continue

        char = text[index]
        ids.append(ord(char) if ord(char) < 256 else reverse_vocab.get("?", 63))
        index += 1

    return ids


def decode_with_vocab(ids: list[int], vocab: dict[int, str]) -> str:
    return "".join(vocab.get(token_id, "") for token_id in ids)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    valid_token_ids: torch.Tensor,
) -> int:
    valid_logits = logits.index_select(dim=-1, index=valid_token_ids)
    if temperature <= 0:
        selected_index = int(torch.argmax(valid_logits, dim=-1).item())
    else:
        probs = F.softmax(valid_logits / temperature, dim=-1)
        selected_index = int(torch.multinomial(probs, num_samples=1).item())
    return int(valid_token_ids[selected_index].item())


def generate_training_sample(
    model: torch.nn.Module,
    vocab_path: Path,
    prompt: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    vocab = load_vocab(vocab_path)
    reverse_vocab = build_reverse_vocab(vocab)
    valid_token_ids = torch.tensor(sorted(vocab), dtype=torch.long, device=device)
    prompt_ids = encode_with_vocab(prompt, reverse_vocab)
    if not prompt_ids:
        return ""

    was_training = model.training
    ids = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            ids.append(sample_next_token(logits, temperature, valid_token_ids))

    if was_training:
        model.train()

    generated_ids = ids[len(prompt_ids) :]
    return decode_with_vocab(generated_ids, vocab)
