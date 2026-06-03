from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
RETRAIN_DIR = Path(__file__).resolve().parent
INPUT_DIR = RETRAIN_DIR / "entrada"
OUTPUT_DIR = RETRAIN_DIR / "checkpoints"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modelo.encoder import Encoder_model, TextDataset
from config import legacy_checkpoint_config, load_config as load_project_config
from config import model_config as project_model_config
from config import resolve_path
from utils import (
    generate_training_sample,
    save_periodic_checkpoint,
    should_save_periodic_checkpoint,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_config: dict[str, Any], device: torch.device) -> Encoder_model:
    return Encoder_model(
        vocab_size=int(model_config["vocab_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        max_seq_len=int(model_config["max_seq_len"]),
        ffn_dim=int(model_config["ffn_dim"]),
    ).to(device)


def load_model_checkpoint(
    model: Encoder_model,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    checkpoint = torch_load(checkpoint_path, device)

    # Existem dois tipos de arquivo que este script entende:
    # 1. encoder.pth antigo: contem apenas model.state_dict().
    #    Nesse caso os pesos continuam de onde pararam, mas o AdamW recomeca
    #    sem historico de medias/momentos. Funciona para continuar treinando,
    #    so nao e uma retomada 100% identica ao mesmo processo original.
    # 2. checkpoint completo novo: contem modelo, otimizador e epoch.
    #    Esse e o melhor formato para pausar e retomar no futuro.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("epoch", 0))

    model.load_state_dict(checkpoint)
    return 0


def prepare_retrain_config(config: dict[str, Any]) -> dict[str, Any]:
    if "model" in config:
        retraining = config["retraining"]
        input_dir = resolve_path(retraining["input_dir"])
        output_dir = resolve_path(retraining["output_dir"])
        prepared = legacy_checkpoint_config(config)
        prepared["tokenizer"]["train_bin_path"] = str(input_dir / retraining["train_bin_name"])
        prepared["tokenizer"]["outdir"] = str(input_dir)
        prepared["tokenizer"]["regenerate_dataset"] = False
        prepared["modelo"] = project_model_config(config)
        prepared["modelo"]["model_save_path"] = str(output_dir / retraining["model_save_name"])
        prepared["modelo"]["config_save_path"] = str(output_dir / "config.json")
        prepared["modelo"]["full_checkpoint_path"] = str(
            output_dir / retraining["full_checkpoint_name"]
        )
        return prepared

    config = json.loads(json.dumps(config))
    config["tokenizer"]["train_bin_path"] = str(INPUT_DIR / "train.bin")
    config["tokenizer"]["outdir"] = str(INPUT_DIR)
    config["tokenizer"]["regenerate_dataset"] = False
    config["modelo"]["model_save_path"] = str(OUTPUT_DIR / "encoder_retreinado.pth")
    config["modelo"]["config_save_path"] = str(OUTPUT_DIR / "config.json")
    return config


def validate_inputs(config: dict[str, Any]) -> tuple[Path | None, Path, Path, Path, Path]:
    if "retraining" in config:
        retraining = config["retraining"]
        input_dir = resolve_path(retraining["input_dir"])
        output_dir = resolve_path(retraining["output_dir"])
        config_path = input_dir / retraining["config_name"]
        checkpoint_path = input_dir / retraining["checkpoint_name"]
        train_path = input_dir / retraining["train_bin_name"]
        vocab_path = input_dir / retraining["vocab_name"]
    else:
        output_dir = OUTPUT_DIR
        config_path = INPUT_DIR / "config.json"
        checkpoint_path = INPUT_DIR / "encoder.pth"
        train_path = INPUT_DIR / "train.bin"
        vocab_path = INPUT_DIR / "vocab.json"

    missing = [
        str(path)
        for path in (checkpoint_path, train_path, vocab_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Arquivos ausentes para re-treino:\n- " + "\n- ".join(missing)
        )

    return (config_path if config_path.exists() else None), checkpoint_path, train_path, vocab_path, output_dir


def maybe_print_generation_sample(
    model: Encoder_model,
    project_config: dict[str, Any],
    epoch: int,
    seq_len: int,
    vocab_path: Path,
    device: torch.device,
) -> None:
    generation_config = project_config.get("generation", {})
    if not generation_config.get("enabled", False):
        return

    interval = int(generation_config.get("interval_epochs", 1))
    if interval <= 0 or epoch % interval != 0:
        return

    prompt = str(generation_config.get("prompt", "Ola"))
    text = generate_training_sample(
        model=model,
        vocab_path=vocab_path,
        prompt=prompt,
        seq_len=seq_len,
        max_new_tokens=int(generation_config.get("max_new_tokens", 40)),
        temperature=float(generation_config.get("temperature", 0.8)),
        device=device,
    )
    print(f"[geracao epoch {epoch:03d}] prompt: {prompt!r}")
    print(f"[geracao epoch {epoch:03d}] modelo: {text}")


def maybe_save_periodic_checkpoint(
    model: Encoder_model,
    optimizer: torch.optim.Optimizer,
    project_config: dict[str, Any],
    checkpoint_config: dict[str, Any],
    epoch: int,
    output_dir: Path,
) -> None:
    if not should_save_periodic_checkpoint(project_config, epoch):
        return

    save_periodic_checkpoint(
        model=model,
        optimizer=optimizer,
        config=checkpoint_config,
        checkpoint_config=project_config.get("checkpointing", {}),
        epoch=epoch,
        output_dir=output_dir,
    )


def train(config_path_arg: str = "config.yaml") -> None:
    project_config = load_project_config(config_path_arg)
    input_config_path, checkpoint_path, train_path, vocab_path, output_dir = validate_inputs(project_config)
    config = prepare_retrain_config(project_config)

    tokens_np = np.fromfile(train_path, dtype=np.uint32)
    if tokens_np.size == 0:
        raise ValueError(f"Nenhum token encontrado em {train_path}")

    model_config = config["modelo"]
    train_config = config["treino"]
    seq_len = int(model_config["seq_len"])
    if tokens_np.size <= seq_len:
        raise ValueError(
            f"O arquivo {train_path} tem {tokens_np.size} tokens, mas seq_len={seq_len}. "
            "Use um seq_len menor ou um train.bin maior."
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
        num_workers=int(train_config.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    device = select_device(str(train_config.get("device", "auto")))
    print(f"Usando dispositivo: {device}")

    model = build_model(model_config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]))
    start_epoch = load_model_checkpoint(model, optimizer, checkpoint_path, device)
    checkpoint_config = dict(config)
    checkpoint_config["checkpointing"] = project_config.get("checkpointing", {})
    periodic_dir = output_dir / str(project_config.get("checkpointing", {}).get("directory_name", "periodicos"))

    print(f"Config base: {input_config_path or resolve_path(config_path_arg)}")
    print(f"Checkpoint inicial: {checkpoint_path}")
    print(f"Vocab mantido junto do re-treino: {vocab_path}")
    print(f"Dataset tokenizado: {train_path}")
    print(f"info modelo: {sum(p.numel() for p in model.parameters())} parametros")
    print(f"tamanho do dataset: {len(dataset)} exemplos, {len(loader)} batches")
    print(f"Epoch inicial registrada: {start_epoch}")
    print(f"Novas epochs nesta execucao: {train_config['num_epochs']}")

    model.train()
    last_epoch = start_epoch
    for offset in range(1, int(train_config["num_epochs"]) + 1):
        epoch = start_epoch + offset
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
        maybe_save_periodic_checkpoint(
            model=model,
            optimizer=optimizer,
            project_config=project_config,
            checkpoint_config=checkpoint_config,
            epoch=epoch,
            output_dir=periodic_dir,
        )
        maybe_print_generation_sample(
            model=model,
            project_config=project_config,
            epoch=epoch,
            seq_len=seq_len,
            vocab_path=vocab_path,
            device=device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = Path(config["modelo"]["model_save_path"])
    config_save_path = Path(config["modelo"]["config_save_path"])
    full_checkpoint_path = Path(
        config["modelo"].get("full_checkpoint_path", output_dir / "checkpoint_completo.pt")
    )

    torch.save(model.state_dict(), model_save_path)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        full_checkpoint_path,
    )
    save_json(config, config_save_path)

    print(f"Modelo re-treinado salvo em: {model_save_path}")
    print(f"Checkpoint completo salvo em: {full_checkpoint_path}")
    print(f"Config para inferencia salva em: {config_save_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Continua o treinamento a partir de pesos existentes.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
