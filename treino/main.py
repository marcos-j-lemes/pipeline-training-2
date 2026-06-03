from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import legacy_checkpoint_config, load_config, model_config, resolve_path, save_json
from modelo.encoder import Encoder_model, TextDataset


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_model(config: dict[str, Any], device: torch.device) -> Encoder_model:
    return Encoder_model(
        vocab_size=int(config["vocab_size"]),
        embedding_dim=int(config["embedding_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        max_seq_len=int(config["max_seq_len"]),
        ffn_dim=int(config["ffn_dim"]),
    ).to(device)


def maybe_regenerate_dataset(config: dict[str, Any]) -> None:
    train_config = config["training"]
    if not train_config.get("regenerate_tokenizer_and_dataset", False):
        return

    from tokenizador import tokenize_dataset, train_tokenizer

    tokenizer_config = config["tokenizer"]
    tokenizer = train_tokenizer(
        str(resolve_path(tokenizer_config["bpe_dataset_path"])),
        int(tokenizer_config["vocab_size"]),
        str(resolve_path(tokenizer_config["artifacts_dir"])),
    )
    tokenize_dataset(
        tokenizer,
        str(resolve_path(tokenizer_config["model_dataset_path"])),
        str(resolve_path(tokenizer_config["dataset_dir"])),
        float(tokenizer_config.get("split_ratio", 0.1)),
    )


def train(config_path: str) -> None:
    config = load_config(config_path)
    maybe_regenerate_dataset(config)

    tokenizer_config = config["tokenizer"]
    model_settings = model_config(config)
    train_settings = config["training"]

    train_path = resolve_path(tokenizer_config["train_bin_path"])
    tokens_np = np.fromfile(train_path, dtype=np.uint32)
    if tokens_np.size == 0:
        raise ValueError(f"Nenhum token encontrado em {train_path}")

    seq_len = int(model_settings["seq_len"])
    if tokens_np.size <= seq_len:
        raise ValueError(
            f"O arquivo {train_path} tem {tokens_np.size} tokens, mas seq_len={seq_len}. "
            "Use um seq_len menor ou gere um train.bin maior."
        )

    seed = int(train_settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokens = torch.tensor(tokens_np.astype(np.int64), dtype=torch.long)
    dataset = TextDataset(tokens, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=int(train_settings["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(train_settings.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    device = select_device(str(train_settings.get("device", "auto")))
    print(f"Usando dispositivo: {device}")

    model = build_model(model_settings, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_settings["learning_rate"]),
    )

    print(f"info modelo: {sum(p.numel() for p in model.parameters())} parametros")
    print(f"dataset: {train_path}")
    print(f"tamanho do dataset: {len(dataset)} exemplos, {len(loader)} batches")
    print(f"Epochs: {train_settings['num_epochs']}, Batch size: {train_settings['batch_size']}")

    model.train()
    last_epoch = 0
    for epoch in range(1, int(train_settings["num_epochs"]) + 1):
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

        last_epoch = epoch
        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch {epoch:03d} | loss {avg_loss:.4f}")

    model_save_path = resolve_path(model_settings["model_save_path"])
    config_save_path = resolve_path(model_settings["config_save_path"])
    full_checkpoint_path = resolve_path(model_settings["full_checkpoint_path"])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    full_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_config = legacy_checkpoint_config(config)
    torch.save(model.state_dict(), model_save_path)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": checkpoint_config,
        },
        full_checkpoint_path,
    )
    save_json(checkpoint_config, config_save_path)

    print(f"Modelo salvo em: {model_save_path}")
    print(f"Checkpoint completo salvo em: {full_checkpoint_path}")
    print(f"Config salva em: {config_save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo a partir do train.bin.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
