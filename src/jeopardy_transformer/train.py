"""Training loop for the Jeopardy Transformer.

The important hardware rule in this file is:

- CUDA gets automatic mixed precision because it is well supported there.
- MPS and CPU use normal full precision because many PyTorch builds still do
  not support `torch.autocast(device_type="mps")`.

That rule is what fixes the MPS crash:

    RuntimeError: User specified an unsupported autocast device_type 'mps'
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jeopardy_transformer.checkpointing import load_checkpoint, save_checkpoint
from jeopardy_transformer.metrics import (
    compute_scale_metrics,
    plot_training_history,
    write_history_files,
)
from jeopardy_transformer.model import ModelConfig, TransformerLM


@dataclass
class TrainConfig:
    """Hyperparameters that control optimization, logging, and checkpointing."""

    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 40
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 1337
    verbose: bool = False
    metrics_dir: str = "metrics"
    save_numbered_checkpoints: bool = True


@dataclass
class TrainingRunSummary:
    """Small return value describing the completed training run."""

    final_step: int
    best_val_loss: float | None
    latest_checkpoint: str
    best_checkpoint: str
    metrics_dir: str | None


def pick_device() -> torch.device:
    """Choose CUDA, then MPS, then CPU, depending on what PyTorch can use."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    # `torch.backends.mps.is_available()` is true on Macs where PyTorch can use
    # Apple Silicon GPU acceleration.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_json_config(path: Path) -> tuple[dict[str, Any], TrainConfig]:
    """Load a JSON config and split it into model and training settings."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    model_overrides = raw.get("model", {})
    train_config = TrainConfig(**raw.get("training", {}))
    return model_overrides, train_config


def load_data_meta(data_dir: Path) -> dict[str, Any]:
    """Read `meta.json` from a prepared data directory.

    This is also where we give a friendlier error if `--data-dir` points at the
    wrong folder. For example, if you prepared `data/jeopardy_tiny`, training
    must use `--data-dir data/jeopardy_tiny`.
    """

    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    parent = data_dir.parent if data_dir.parent.exists() else Path("data")
    nearby = sorted(path.name for path in parent.glob("*") if (path / "meta.json").exists())
    hint = f" Prepared data directories nearby: {nearby}." if nearby else ""
    raise FileNotFoundError(
        f"Missing {meta_path}. Run scripts/prepare_data.py first, or pass the "
        f"prepared folder with --data-dir.{hint}"
    )


def load_token_file(path: Path) -> np.memmap:
    """Open a compact `.bin` token file without reading it all into RAM."""

    if not path.exists():
        raise FileNotFoundError(f"Missing token file: {path}")
    return np.memmap(path, dtype=np.uint32, mode="r")


def get_batch(
    data: np.memmap,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random next-token-prediction windows from one token array.

    `x` contains tokens `[t0, t1, ..., tN]`.
    `y` contains the same window shifted left by one token:
    `[t1, t2, ..., tN+1]`.
    """

    if len(data) <= block_size + 1:
        raise ValueError(
            f"Token file has {len(data)} tokens, but block_size is {block_size}. "
            "Prepare more data or lower model.block_size in the config."
        )

    # Each start index chooses one training example from the long flat token file.
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    # `np.array(...)` copies out of the memmap. PyTorch warns on non-writable
    # memmap slices, so the explicit copy keeps the conversion quiet and safe.
    x = torch.stack(
        [
            torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64))
            for i in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64))
            for i in starts
        ]
    )

    # `non_blocking` is useful on CUDA and harmless elsewhere.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def configure_optimizer(
    model: torch.nn.Module,
    train_config: TrainConfig,
) -> torch.optim.Optimizer:
    """Create AdamW with weight decay only on matrix-like parameters."""

    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue

        # Biases and LayerNorm weights are one-dimensional. GPT-style training
        # usually avoids weight decay on those small scale/shift parameters.
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )


def learning_rate_for_step(step: int, train_config: TrainConfig) -> float:
    """Return the learning rate for one step using warmup plus cosine decay."""

    if step < train_config.warmup_steps:
        # Warmup avoids a large optimizer jump when gradients are still chaotic.
        return train_config.learning_rate * (step + 1) / train_config.warmup_steps

    progress = (step - train_config.warmup_steps) / max(
        1,
        train_config.max_steps - train_config.warmup_steps,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return train_config.min_learning_rate + cosine * (
        train_config.learning_rate - train_config.min_learning_rate
    )


def build_autocast_context(device: torch.device, use_amp: bool):
    """Return CUDA autocast or a no-op context manager for MPS/CPU.

    This is deliberately stricter than `torch.autocast(..., enabled=False)`.
    Some PyTorch versions validate the device type before checking `enabled`,
    which means `torch.autocast(device_type="mps", enabled=False)` can still
    crash. Returning `nullcontext()` for non-CUDA avoids constructing autocast at
    all.
    """

    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(use_amp: bool):
    """Create a CUDA GradScaler when mixed precision is active.

    On MPS and CPU the scaler is disabled, but keeping the same `.scale`,
    `.step`, and `.update` calls makes the training loop easier to read.
    """

    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer tensors to the active device after loading a checkpoint."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: np.memmap,
    val_data: np.memmap,
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Average train and validation loss over several random batches."""

    model.eval()
    losses: dict[str, list[float]] = {"train": [], "val": []}

    for split, data in (("train", train_data), ("val", val_data)):
        for _ in range(train_config.eval_batches):
            x, y = get_batch(
                data,
                batch_size=train_config.batch_size,
                block_size=model.config.block_size,
                device=device,
            )

            # Evaluation follows the same hardware rule as training: CUDA may
            # use autocast, while MPS and CPU stay in normal full precision.
            with build_autocast_context(device, device.type == "cuda"):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Expected a loss during evaluation")
            losses[split].append(float(loss.item()))

    model.train()
    return {split: sum(values) / len(values) for split, values in losses.items()}


def _metric_row(
    *,
    step: int,
    epoch: float,
    train_loss: float,
    learning_rate: float,
    elapsed_seconds: float,
    scale_metrics: dict[str, float],
    eval_losses: dict[str, float] | None,
    best_val_loss: float | None,
) -> dict[str, float | int | None]:
    """Build one serializable metric row for CSV/JSONL logging."""

    return {
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed_seconds,
        "weight_rms": scale_metrics["weight_rms"],
        "grad_rms": scale_metrics["grad_rms"],
        "estimated_update_rms": scale_metrics["estimated_update_rms"],
        "estimated_update_to_weight_ratio": scale_metrics[
            "estimated_update_to_weight_ratio"
        ],
        "eval_train_loss": None if eval_losses is None else eval_losses["train"],
        "val_loss": None if eval_losses is None else eval_losses["val"],
        "best_val_loss": best_val_loss,
    }


def train(
    *,
    config_path: Path,
    data_dir: Path,
    checkpoints_dir: Path,
    resume: Path | None = None,
    verbose: bool | None = None,
    save_numbered_checkpoints: bool | None = None,
) -> TrainingRunSummary:
    """Train the model, save checkpoints, and optionally write metric plots."""

    model_overrides, train_config = load_json_config(config_path)
    requested_verbose = verbose
    requested_save_numbered = save_numbered_checkpoints

    if requested_verbose is not None:
        train_config.verbose = requested_verbose
    if requested_save_numbered is not None:
        train_config.save_numbered_checkpoints = requested_save_numbered

    data_dir = data_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoints_dir = checkpoints_dir.expanduser().resolve()

    meta = load_data_meta(data_dir)

    torch.manual_seed(train_config.seed)
    device = pick_device()

    train_data = load_token_file(data_dir / "train.bin")
    val_data = load_token_file(data_dir / "val.bin")

    start_step = 0
    best_val_loss: float | None = None
    checkpoint: dict[str, Any] | None = None

    if resume is not None:
        resume = resume.expanduser().resolve()
        checkpoint = load_checkpoint(resume, map_location=device)
        model_config = ModelConfig(**checkpoint["model_config"])
        train_config = TrainConfig(**{**asdict(train_config), **checkpoint["train_config"]})
        if requested_verbose is not None:
            train_config.verbose = requested_verbose
        if requested_save_numbered is not None:
            train_config.save_numbered_checkpoints = requested_save_numbered
    else:
        model_config = ModelConfig(
            vocab_size=int(meta["vocab_size"]),
            **model_overrides,
        )

    model = TransformerLM(model_config).to(device)
    optimizer = configure_optimizer(model, train_config)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = checkpoint.get("best_val_loss")
        print(f"Resumed from {resume} at step {start_step}.")

    use_amp = device.type == "cuda"
    scaler = build_grad_scaler(use_amp)
    metrics_dir = checkpoints_dir / train_config.metrics_dir
    history: list[dict[str, Any]] = []
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * model.config.block_size
    )

    print(f"Training source: {Path(__file__).resolve()}")
    print(f"Config: {config_path}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoints_dir}")
    print(f"Device: {device}")
    print(f"Mixed precision: {'cuda autocast' if use_amp else 'off'}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    if device.type == "mps":
        print("MPS note: training uses full precision because MPS autocast is not used.")

    model.train()
    last_log_time = time.time()
    final_step = start_step - 1

    for step in range(start_step, train_config.max_steps):
        final_step = step
        lr = learning_rate_for_step(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(train_config.gradient_accumulation_steps):
            x, y = get_batch(
                train_data,
                batch_size=train_config.batch_size,
                block_size=model.config.block_size,
                device=device,
            )

            # This context is CUDA autocast on NVIDIA GPUs and a no-op elsewhere.
            # It never constructs MPS autocast, which is the source of your error.
            with build_autocast_context(device, use_amp):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Expected a loss during training")

                # Divide before backward so accumulated gradients average together
                # instead of becoming larger when accumulation_steps increases.
                loss = loss / train_config.gradient_accumulation_steps

            running_loss += float(loss.detach().item())
            scaler.scale(loss).backward()

        if use_amp:
            # Gradients must be unscaled before clipping or measuring them.
            scaler.unscale_(optimizer)

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        should_log = step % train_config.log_interval == 0
        should_eval = (
            step % train_config.eval_interval == 0
            or step == train_config.max_steps - 1
        )
        should_save = (
            step % train_config.save_interval == 0
            or step == train_config.max_steps - 1
        )

        if should_log or train_config.verbose:
            scale_metrics = compute_scale_metrics(model, learning_rate=lr)
        else:
            scale_metrics = {
                "weight_rms": math.nan,
                "grad_rms": math.nan,
                "estimated_update_rms": math.nan,
                "estimated_update_to_weight_ratio": math.nan,
            }

        scaler.step(optimizer)
        scaler.update()

        eval_losses: dict[str, float] | None = None
        elapsed_for_row = time.time() - last_log_time
        if should_eval:
            eval_losses = estimate_loss(model, train_data, val_data, train_config, device)
            print(
                f"eval step {step:05d} | "
                f"train {eval_losses['train']:.4f} | val {eval_losses['val']:.4f}"
            )

            if best_val_loss is None or eval_losses["val"] < best_val_loss:
                best_val_loss = eval_losses["val"]
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    model_config=model.config,
                    train_config=train_config,
                    best_val_loss=best_val_loss,
                    meta=meta,
                )

        if should_log:
            now = time.time()
            elapsed = now - last_log_time
            elapsed_for_row = elapsed
            last_log_time = now
            epoch = ((step + 1) * tokens_per_step) / max(1, len(train_data))
            print(
                f"step {step:05d} | "
                f"epoch {epoch:.3f} | "
                f"loss {running_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"update/weight {scale_metrics['estimated_update_to_weight_ratio']:.2e} | "
                f"{elapsed:.2f}s"
            )

        if train_config.verbose and (should_log or should_eval):
            epoch = ((step + 1) * tokens_per_step) / max(1, len(train_data))
            history.append(
                _metric_row(
                    step=step,
                    epoch=epoch,
                    train_loss=running_loss,
                    learning_rate=lr,
                    elapsed_seconds=elapsed_for_row,
                    scale_metrics=scale_metrics,
                    eval_losses=eval_losses,
                    best_val_loss=best_val_loss,
                )
            )
            write_history_files(history, metrics_dir)

        if should_save:
            save_checkpoint(
                checkpoints_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                model_config=model.config,
                train_config=train_config,
                best_val_loss=best_val_loss,
                meta=meta,
            )

            if train_config.save_numbered_checkpoints:
                save_checkpoint(
                    checkpoints_dir / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    model_config=model.config,
                    train_config=train_config,
                    best_val_loss=best_val_loss,
                    meta=meta,
                )

            print(f"saved checkpoint at step {step}")

    if train_config.verbose:
        write_history_files(history, metrics_dir)
        plot_training_history(history, metrics_dir)
        print(f"Verbose metrics written to {metrics_dir}")

    return TrainingRunSummary(
        final_step=final_step,
        best_val_loss=best_val_loss,
        latest_checkpoint=str(checkpoints_dir / "latest.pt"),
        best_checkpoint=str(checkpoints_dir / "best.pt"),
        metrics_dir=str(metrics_dir) if train_config.verbose else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for `scripts/train.py`."""

    parser = argparse.ArgumentParser(description="Train the Jeopardy Transformer.")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/jeopardy"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write metrics CSV/JSONL and plot loss/scale graphs.",
    )
    parser.add_argument(
        "--no-numbered-checkpoints",
        action="store_true",
        help="Only save latest.pt and best.pt, which uses less disk space.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and start training."""

    args = build_arg_parser().parse_args()
    train(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        resume=args.resume,
        verbose=True if args.verbose else None,
        save_numbered_checkpoints=False if args.no_numbered_checkpoints else None,
    )


if __name__ == "__main__":
    main()
