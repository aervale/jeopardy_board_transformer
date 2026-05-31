"""Small validation-set hyperparameter sweeps.

For language modeling, full k-fold cross-validation is usually expensive: every
fold means another training run, and the model already sees millions of random
token windows from the training split. A practical alternative is to keep one
validation split fixed, run several short training jobs, and pick the config
with the best validation loss.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from jeopardy_transformer.train import train


def _set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using a key like `training.learning_rate`."""

    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _grid_items(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a grid dictionary into one dictionary per hyperparameter trial."""

    keys = list(grid.keys())
    value_lists = [grid[key] for key in keys]
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*value_lists)
    ]


def _release_accelerator_memory() -> None:
    """Ask Python, CUDA, and MPS to release cached memory between sweep trials."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def run_sweep(
    *,
    base_config_path: Path,
    sweep_config_path: Path,
    data_dir: Path,
    out_dir: Path,
    verbose: bool,
) -> Path:
    """Run a validation sweep and write a sorted summary CSV.

    The sweep config has three useful sections:

    - `fixed_training`: overrides applied to every trial.
    - `fixed_model`: model overrides applied to every trial.
    - `grid`: dotted config paths mapped to lists of values.
    """

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    sweep_config = json.loads(sweep_config_path.read_text(encoding="utf-8"))
    trials = _grid_items(sweep_config["grid"])

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    (out_dir / "sweep_results.jsonl").write_text("", encoding="utf-8")

    for trial_index, trial_values in enumerate(trials, start=1):
        run_name = f"trial_{trial_index:03d}"
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = deepcopy(base_config)
        for key, value in sweep_config.get("fixed_model", {}).items():
            _set_dotted_value(config, f"model.{key}", value)
        for key, value in sweep_config.get("fixed_training", {}).items():
            _set_dotted_value(config, f"training.{key}", value)
        for dotted_key, value in trial_values.items():
            _set_dotted_value(config, dotted_key, value)

        # Sweep trials keep only latest/best checkpoints to avoid using lots of
        # storage while comparing many small experiments.
        _set_dotted_value(config, "training.save_numbered_checkpoints", False)

        trial_config_path = run_dir / "config.json"
        trial_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(f"\n=== Sweep {trial_index}/{len(trials)}: {trial_values} ===")
        summary = train(
            config_path=trial_config_path,
            data_dir=data_dir,
            checkpoints_dir=run_dir,
            verbose=verbose,
            save_numbered_checkpoints=False,
        )

        result = {
            "trial": run_name,
            "best_val_loss": summary.best_val_loss,
            "final_step": summary.final_step,
            "config_path": str(trial_config_path),
            "best_checkpoint": summary.best_checkpoint,
            **trial_values,
        }
        results.append(result)

        with (out_dir / "sweep_results.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

        _release_accelerator_memory()

    results.sort(
        key=lambda row: float("inf")
        if row["best_val_loss"] is None
        else float(row["best_val_loss"])
    )

    csv_path = out_dir / "sweep_results.csv"
    fieldnames = list(results[0].keys()) if results else []
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if results:
        best = results[0]
        print(
            "\nBest trial: "
            f"{best['trial']} | val loss {best['best_val_loss']} | "
            f"config {best['config_path']}"
        )

    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for validation sweeps."""

    parser = argparse.ArgumentParser(description="Run a small validation sweep.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/sweep_tiny.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/jeopardy_tiny"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/sweeps"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments and run the sweep."""

    args = build_arg_parser().parse_args()
    csv_path = run_sweep(
        base_config_path=args.base_config,
        sweep_config_path=args.sweep_config,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )
    print(f"Wrote sweep summary to {csv_path}")


if __name__ == "__main__":
    main()
