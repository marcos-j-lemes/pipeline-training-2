from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config as load_project_config
from config import model_config as project_model_config
from config import resolve_path
from modelo.encoder import Encoder_model


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        return load_project_config(path)
    with path.open("r", encoding="utf8") as file:
        return json.load(file)


def normalize_config(config: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    if "model" in config:
        model_settings = project_model_config(config)
        model_path = resolve_path(model_settings["model_save_path"])
        vocab_path = resolve_path(config["tokenizer"]["artifacts_dir"]) / "vocab.json"
        return model_settings, model_path, vocab_path

    model_settings = config["modelo"]
    model_path = resolve_path(model_settings["model_save_path"])
    vocab_path = resolve_path(config["tokenizer"]["outdir"]) / "vocab.json"
    return model_settings, model_path, vocab_path


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_weights(model: Encoder_model, model_path: Path, device: torch.device) -> None:
    checkpoint = torch_load(model_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return
    model.load_state_dict(checkpoint)


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
    pieces = []
    for token_id in ids:
        token = vocab.get(token_id)
        if token is None:
            continue
        pieces.append(token)
    return "".join(pieces)


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


def generate(
    model: Encoder_model,
    prompt_ids: list[int],
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    valid_token_ids: torch.Tensor,
    device: torch.device,
) -> list[int]:
    ids = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            ids.append(sample_next_token(logits, temperature, valid_token_ids))
    return ids


def stream_text(text: str, delay: float) -> None:
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def chat_loop(
    model: Encoder_model,
    vocab: dict[int, str],
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    delay: float,
    device: torch.device,
) -> None:
    reverse_vocab = build_reverse_vocab(vocab)
    valid_token_ids = torch.tensor(sorted(vocab), dtype=torch.long, device=device)
    print("Chat iniciado. Digite 'sair' para encerrar.")
    while True:
        user_text = input("Voce: ").strip()
        if user_text.lower() in {"sair", "exit", "quit"}:
            print("Encerrado.")
            return
        if not user_text:
            continue

        prompt_ids = encode_with_vocab(user_text, reverse_vocab)
        if not prompt_ids:
            print("Modelo: nao consegui converter esse texto em tokens.")
            continue

        generated_ids = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            seq_len=seq_len,
            temperature=temperature,
            valid_token_ids=valid_token_ids,
            device=device,
        )
        response_text = decode_with_vocab(generated_ids[len(prompt_ids) :], vocab)
        print("Modelo: ", end="", flush=True)
        stream_text(response_text, delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat simples no terminal com o modelo treinado.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt-ids", default=None)
    parser.add_argument("--once", default=None, help="Executa uma unica resposta para o texto informado.")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--delay", type=float, default=0.01)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    model_config, model_path, vocab_path = normalize_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Encoder_model(
        vocab_size=int(model_config["vocab_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        max_seq_len=int(model_config["max_seq_len"]), # 
        ffn_dim=int(model_config["ffn_dim"]),
    ).to(device)
    load_model_weights(model, model_path, device)

    vocab = load_vocab(vocab_path)
    seq_len = int(model_config["seq_len"])
    valid_token_ids = torch.tensor(sorted(vocab), dtype=torch.long, device=device)

    if args.prompt_ids:
        prompt_ids = [int(item.strip()) for item in args.prompt_ids.split(",") if item.strip()]
        generated_ids = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            seq_len=seq_len,
            temperature=args.temperature,
            valid_token_ids=valid_token_ids,
            device=device,
        )
        print("IDs:", generated_ids)
        print("Texto aproximado:", decode_with_vocab(generated_ids, vocab))
        return

    if args.once is not None:
        reverse_vocab = build_reverse_vocab(vocab)
        prompt_ids = encode_with_vocab(args.once, reverse_vocab)
        generated_ids = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            seq_len=seq_len,
            temperature=args.temperature,
            valid_token_ids=valid_token_ids,
            device=device,
        )
        response_text = decode_with_vocab(generated_ids[len(prompt_ids) :], vocab)
        print("Modelo: ", end="", flush=True)
        stream_text(response_text, args.delay)
        return

    chat_loop(
        model=model,
        vocab=vocab,
        max_new_tokens=args.max_new_tokens,
        seq_len=seq_len,
        temperature=args.temperature,
        delay=args.delay,
        device=device,
    )


if __name__ == "__main__":
    main()
