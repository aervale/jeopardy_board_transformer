"""Checkpoint helpers.

Checkpoints are just PyTorch dictionaries saved to disk. Keeping the checkpoint
format in one file makes it easier to change the training loop later without
losing track of what gets saved.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def _to_plain_dict(value: Any) -> Any:
    """Convert dataclass configs into plain dictionaries for serialization."""

    if is_dataclass(value):
        return asdict(value)
    return value


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: Any,
    train_config: Any,
    best_val_loss: float | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save model, optimizer, configs, and training progress atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": _to_plain_dict(model_config),
        "train_config": _to_plain_dict(train_config),
        "best_val_loss": best_val_loss,
        "meta": meta or {},
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
