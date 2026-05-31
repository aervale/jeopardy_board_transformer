"""Dataset formatting and token-cache creation.

Language models learn by next-token prediction:

    x = "The capital of France is"
    y = " capital of France is Paris"

The model sees the first sequence and learns to predict the second sequence one
position at a time. For Jeopardy, we turn every row into one structured text
record first, then concatenate all records into long token streams.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from jeopardy_transformer.special_tokens import (
    AIR_DATE,
    ANSWER,
    CATEGORY,
    DOLLAR_VALUE,
    END_OF_EXAMPLE,
    PROMPT,
    ROUND_NAME,
    SHOW_NUMBER,
    build_tokenizer,
)


DATASET_NAME = "soldni/jeopardy"
DATASET_CONFIG = "all_questions"
DATASET_SPLIT = "train"


@dataclass(frozen=True)
class PreparedDataStats:
    """Small summary saved next to the tokenized files."""

    dataset_name: str
    dataset_config: str
    dataset_split: str
    train_examples: int
    val_examples: int
    train_tokens: int
    val_tokens: int
    tokenizer_base: str
    vocab_size: int
    special_tokens: list[str]


def clean_field(value: Any, fallback: str = "Unknown") -> str:
    """Normalize one cell from the dataset into a single safe line of text."""

    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    # Keeping each field on one line makes the formatted JSONL easy to inspect.
    return " ".join(text.split())


def format_jeopardy_row(row: dict[str, Any]) -> str:
    """Turn one Hugging Face row into the structured text the model sees.

    The target for this model is not just "answer the prompt"; it is "continue
    this structured Jeopardy record." That lets generation produce a round,
    category, dollar value, prompt, and answer in one sequence.
    """

    round_name = clean_field(row.get("round"))
    category = clean_field(row.get("og-category") or row.get("category"))
    value = clean_field(row.get("value"), fallback="No value")
    question = clean_field(row.get("question"))
    answer = clean_field(row.get("answer"))
    air_date = clean_field(row.get("air_date"), fallback="Unknown date")
    show_number = clean_field(row.get("show_number"), fallback="Unknown show")

    return (
        f"{ROUND_NAME} {round_name}\n"
        f"{CATEGORY} {category}\n"
        f"{DOLLAR_VALUE} {value}\n"
        f"{PROMPT} {question}\n"
        f"{ANSWER} {answer}\n"
        f"{AIR_DATE} {air_date}\n"
        f"{SHOW_NUMBER} {show_number}\n"
        f"{END_OF_EXAMPLE}\n"
    )


def _write_tokens(binary_file, token_ids: list[int]) -> int:
    """Append token ids as compact uint32 values and return how many were written."""

    array = np.asarray(token_ids, dtype=np.uint32)
    binary_file.write(array.tobytes())
    return int(array.size)


def prepare_dataset(
    out_dir: Path,
    *,
    validation_fraction: float = 0.02,
    max_examples: int | None = None,
    seed: int = 1337,
    tokenizer_base: str = "gpt2",
    write_formatted_jsonl: bool = True,
) -> PreparedDataStats:
    """Download, reformat, tokenize, and save the Jeopardy dataset.

    The token files are simple flat arrays:

    - `train.bin`: token ids for training
    - `val.bin`: token ids for validation

    During training we randomly slice fixed-length windows out of those arrays.
    That is the same basic data layout used in many small GPT implementations.
    """

    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction should be between 0 and 0.5")

    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(tokenizer_base)

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    dataset = dataset.shuffle(seed=seed)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    val_examples_target = max(1, int(len(dataset) * validation_fraction))

    train_bin_path = out_dir / "train.bin"
    val_bin_path = out_dir / "val.bin"
    train_jsonl_path = out_dir / "formatted_train.jsonl"
    val_jsonl_path = out_dir / "formatted_val.jsonl"

    train_examples = val_examples = 0
    train_tokens = val_tokens = 0

    train_jsonl = (
        train_jsonl_path.open("w", encoding="utf-8")
        if write_formatted_jsonl
        else None
    )
    val_jsonl = (
        val_jsonl_path.open("w", encoding="utf-8") if write_formatted_jsonl else None
    )

    try:
        with train_bin_path.open("wb") as train_bin, val_bin_path.open("wb") as val_bin:
            for i, row in enumerate(tqdm(dataset, desc="Formatting/tokenizing")):
                split = "val" if i < val_examples_target else "train"
                text = format_jeopardy_row(row)
                token_ids = tokenizer.encode(text)

                record = {
                    "source_id": clean_field(row.get("id"), fallback=str(i)),
                    "text": text,
                }

                if split == "val":
                    val_examples += 1
                    val_tokens += _write_tokens(val_bin, token_ids)
                    if val_jsonl is not None:
                        val_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    train_examples += 1
                    train_tokens += _write_tokens(train_bin, token_ids)
                    if train_jsonl is not None:
                        train_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if train_jsonl is not None:
            train_jsonl.close()
        if val_jsonl is not None:
            val_jsonl.close()

    stats = PreparedDataStats(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        train_examples=train_examples,
        val_examples=val_examples,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer_base=tokenizer_base,
        vocab_size=tokenizer.vocab_size,
        special_tokens=list(tokenizer.special_tokens),
    )

    (out_dir / "meta.json").write_text(
        json.dumps(asdict(stats), indent=2),
        encoding="utf-8",
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for `scripts/prepare_data.py`."""

    parser = argparse.ArgumentParser(description="Prepare the Jeopardy dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/jeopardy"))
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tokenizer-base", type=str, default="gpt2")
    parser.add_argument(
        "--skip-formatted-jsonl",
        action="store_true",
        help="Only write compact token .bin files, not human-readable JSONL.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, prepare the dataset, and print a short summary."""

    args = build_arg_parser().parse_args()
    stats = prepare_dataset(
        args.out_dir,
        validation_fraction=args.validation_fraction,
        max_examples=args.max_examples,
        seed=args.seed,
        tokenizer_base=args.tokenizer_base,
        write_formatted_jsonl=not args.skip_formatted_jsonl,
    )
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
