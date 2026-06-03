from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def should_save_periodic_checkpoint(config: dict[str, Any], epoch: int) -> bool:
    checkpoint_config = config.get("checkpointing", {})
    if not checkpoint_config.get("enabled", False):
        return False

    interval = int(checkpoint_config.get("interval_epochs", 1))
    return interval > 0 and epoch % interval == 0


def save_periodic_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    checkpoint_config: dict[str, Any],
    epoch: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_config.get("save_weights_only", True):
        weights_path = output_dir / f"encoder_epoch_{epoch:04d}.pth"
        torch.save(model.state_dict(), weights_path)
        print(f"[checkpoint epoch {epoch:03d}] pesos salvos em: {weights_path}")

    if checkpoint_config.get("save_full_checkpoint", True):
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            checkpoint_path,
        )
        print(f"[checkpoint epoch {epoch:03d}] completo salvo em: {checkpoint_path}")
