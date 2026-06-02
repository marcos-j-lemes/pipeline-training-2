from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modelo.encoder import Encoder_model, TextDataset
from tokenizador import tokenize_dataset, train_tokenizer


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def load_config() -> dict:
    config_path = Path(__file__).resolve().with_name("config.json")
    with config_path.open("r", encoding="utf8") as file:
        return json.load(file)


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def maybe_regenerate_dataset(config: dict) -> None:
    tokenizer_config = config["tokenizer"]
    if not tokenizer_config.get("regenerate_dataset", False):
        return

    tokenizer = train_tokenizer(
        str(resolve_project_path(tokenizer_config["dataset_path"])),
        int(tokenizer_config["vocab_size"]),
        str(resolve_project_path(tokenizer_config["outdir"])),
    )
    tokenize_dataset(
        tokenizer,
        str(resolve_project_path(tokenizer_config["dataset_text_path"])),
        str(resolve_project_path(tokenizer_config["outdir_tokenizer"])),
    )


def train() -> None:
    config = load_config()
    maybe_regenerate_dataset(config)

    train_path = resolve_project_path(config["tokenizer"]["train_bin_path"])
    tokens_np = np.fromfile(train_path, dtype=np.uint32)
    if tokens_np.size == 0:
        raise ValueError(f"Nenhum token encontrado em {train_path}")

    model_config = config["modelo"]
    train_config = config["treino"]
    seq_len = int(model_config["seq_len"])
    if tokens_np.size <= seq_len:
        raise ValueError(
            f"O arquivo {train_path} tem {tokens_np.size} tokens, mas seq_len={seq_len}. "
            "Use um seq_len menor ou gere um train.bin maior."
        )

    seed = int(train_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokens = torch.tensor(tokens_np.astype(np.int64), dtype=torch.long)
    dataset = TextDataset(tokens, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=2,  # Use 0 para evitar problemas de multiprocessing em alguns ambientes
        pin_memory=True,  # Carrega tudo na memória para acelerar o acesso
    )

    #device = select_device(str(train_config.get("device", "auto")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = Encoder_model(
        vocab_size=int(model_config["vocab_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        max_seq_len=int(model_config["max_seq_len"]),
        ffn_dim=int(model_config["ffn_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]))

    print(f"info modelo: {sum(p.numel() for p in model.parameters())} parâmetros")
    print(f"tamanho do dataset: {len(dataset)} tokens, {len(loader)} batches")
    print(f"Epochs: {train_config['num_epochs']}, Batch size: {train_config['batch_size']}")

    model.train()
    for epoch in range(1, int(train_config["num_epochs"]) + 1):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch {epoch:03d} | loss {avg_loss:.4f}")

    model_save_path = resolve_project_path(model_config["model_save_path"])
    config_save_path = resolve_project_path(model_config["config_save_path"])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    config_save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_save_path)
    with config_save_path.open("w", encoding="utf8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"Modelo salvo em: {model_save_path}")
    print(f"Config salva em: {config_save_path}")


if __name__ == "__main__":
    train()
