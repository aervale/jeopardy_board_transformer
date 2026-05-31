"""Metric logging and plotting helpers for training.

These helpers are intentionally separate from the training loop so that the
loop stays readable. The values here are meant for learning and debugging, not
for writing a perfect experiment-tracking system.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def compute_scale_metrics(
    model: torch.nn.Module,
    *,
    learning_rate: float,
) -> dict[str, float]:
    """Return global weight, gradient, and estimated update scale numbers.

    `weight_rms` is the root-mean-square size of all trainable weights.
    `grad_rms` is the root-mean-square size of all gradients after backprop.
    `estimated_update_to_weight_ratio` is roughly `learning_rate * grad / weight`.

    AdamW changes each parameter with adaptive per-parameter scaling, so this is
    an estimate, not the exact optimizer update. It is still useful: if this
    number is huge, training is probably unstable; if it is tiny, learning may
    be very slow.
    """

    weight_sq_sum = 0.0
    grad_sq_sum = 0.0
    weight_count = 0
    grad_count = 0

    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue

            # Convert to float32 for the statistic even if CUDA autocast trained
            # the forward pass in float16. This keeps the metric comparable.
            weights = parameter.detach().float()
            weight_sq_sum += float((weights * weights).sum().item())
            weight_count += weights.numel()

            if parameter.grad is None:
                continue

            gradients = parameter.grad.detach().float()
            grad_sq_sum += float((gradients * gradients).sum().item())
            grad_count += gradients.numel()

    weight_rms = math.sqrt(weight_sq_sum / max(1, weight_count))
    grad_rms = math.sqrt(grad_sq_sum / max(1, grad_count))
    estimated_update_rms = learning_rate * grad_rms

    return {
        "weight_rms": weight_rms,
        "grad_rms": grad_rms,
        "estimated_update_rms": estimated_update_rms,
        "estimated_update_to_weight_ratio": estimated_update_rms
        / max(weight_rms, 1e-12),
    }


def write_history_files(
    history: list[dict[str, Any]],
    metrics_dir: Path,
) -> None:
    """Write the accumulated metric history as JSONL and CSV files."""

    if not history:
        return

    metrics_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = metrics_dir / "history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in history:
            file.write(json.dumps(row) + "\n")

    # Keep column order stable so the CSV is easy to compare between runs.
    fieldnames = list(history[0].keys())
    for row in history[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path = metrics_dir / "history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _series(
    history: list[dict[str, Any]],
    key: str,
) -> tuple[list[float], list[float]]:
    """Extract `(epoch, value)` pairs for rows where `key` is present."""

    x_values: list[float] = []
    y_values: list[float] = []
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        x_values.append(float(row["epoch"]))
        y_values.append(float(value))
    return x_values, y_values


def plot_training_history(
    history: list[dict[str, Any]],
    metrics_dir: Path,
) -> None:
    """Create loss and scale plots from the metric history.

    The function imports Matplotlib lazily so training still works on a minimal
    environment. If Matplotlib is not installed, the CSV/JSONL metrics are still
    written and a short message explains what happened.
    """

    if not history:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Verbose metrics were saved, but matplotlib is not installed for plots.")
        return

    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = _series(history, "train_loss")
    eval_train_x, eval_train_y = _series(history, "eval_train_loss")
    val_x, val_y = _series(history, "val_loss")

    figure, axis = plt.subplots(figsize=(8, 5))
    if train_y:
        axis.plot(train_x, train_y, label="training batch loss", linewidth=1.5)
    if eval_train_y:
        axis.plot(eval_train_x, eval_train_y, label="eval train loss", marker="o")
    if val_y:
        axis.plot(val_x, val_y, label="validation loss", marker="o")
    axis.set_title("Loss vs Token Epoch")
    axis.set_xlabel("token epoch")
    axis.set_ylabel("cross-entropy loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(metrics_dir / "loss_vs_epoch.png", dpi=160)
    plt.close(figure)

    weight_x, weight_y = _series(history, "weight_rms")
    grad_x, grad_y = _series(history, "grad_rms")
    ratio_x, ratio_y = _series(history, "estimated_update_to_weight_ratio")

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    if weight_y:
        axes[0].plot(weight_x, weight_y, label="weight RMS")
    if grad_y:
        axes[0].plot(grad_x, grad_y, label="gradient RMS")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if ratio_y:
        axes[1].plot(ratio_x, ratio_y, color="#244fb0", label="estimated update / weight")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("token epoch")
    axes[1].set_ylabel("ratio")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    figure.suptitle("Weight And Update Scale")
    figure.tight_layout()
    figure.savefig(metrics_dir / "scale_metrics.png", dpi=160)
    plt.close(figure)

