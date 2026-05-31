"""Generate Jeopardy-style clues and write an interactive HTML file."""

from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

import torch

from jeopardy_transformer.checkpointing import load_checkpoint
from jeopardy_transformer.html import GeneratedClue, write_generated_clues_html
from jeopardy_transformer.model import ModelConfig, TransformerLM
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
from jeopardy_transformer.train import pick_device


FIELD_TOKENS = (
    ROUND_NAME,
    CATEGORY,
    DOLLAR_VALUE,
    PROMPT,
    ANSWER,
    AIR_DATE,
    SHOW_NUMBER,
)
FIELD_RE = re.compile("|".join(re.escape(token) for token in FIELD_TOKENS))


def _extract_fields(block: str) -> dict[str, str]:
    """Pull known special-token fields out of one generated text block."""

    matches = list(FIELD_RE.finditer(block))
    fields: dict[str, str] = {}

    for i, match in enumerate(matches):
        token = match.group(0)
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        fields[token] = block[value_start:value_end].strip()

    return fields


def parse_generated_text(text: str) -> list[GeneratedClue]:
    """Parse sampled text into clue objects.

    Generation can be messy, especially early in training. The parser is
    intentionally forgiving: it extracts whatever fields it can find and skips
    blocks that have neither a prompt nor an answer.
    """

    clues: list[GeneratedClue] = []
    for block in text.split(END_OF_EXAMPLE):
        block = block.strip()
        if not block:
            continue

        fields = _extract_fields(block)
        prompt = fields.get(PROMPT, "")
        answer = fields.get(ANSWER, "")
        if not prompt and not answer:
            continue

        clues.append(
            GeneratedClue(
                round_name=fields.get(ROUND_NAME, ""),
                category=fields.get(CATEGORY, ""),
                dollar_value=fields.get(DOLLAR_VALUE, ""),
                prompt=prompt,
                answer=answer,
                raw_text=block,
            )
        )

    return clues


def load_model_for_generation(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[TransformerLM, object]:
    """Load model weights and the matching tokenizer."""

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = TransformerLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer_base = checkpoint.get("meta", {}).get("tokenizer_base", "gpt2")
    tokenizer = build_tokenizer(tokenizer_base)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint. "
            f"tokenizer={tokenizer.vocab_size}, model={model.config.vocab_size}"
        )

    return model, tokenizer


@torch.no_grad()
def generate_clues(
    model: TransformerLM,
    tokenizer,
    *,
    num_clues: int,
    max_tokens_per_clue: int = 160,
    temperature: float = 0.9,
    top_k: int | None = 50,
    seed_text: str = ROUND_NAME,
) -> list[GeneratedClue]:
    """Sample until we have the requested number of parsed clues."""

    device = next(model.parameters()).device
    stop_token_id = tokenizer.encode(END_OF_EXAMPLE)[0]
    clues: list[GeneratedClue] = []

    # A few extra attempts help when a young checkpoint emits malformed samples.
    max_attempts = max(num_clues * 4, 8)
    for _ in range(max_attempts):
        input_ids = torch.tensor(
            [tokenizer.encode(seed_text)],
            dtype=torch.long,
            device=device,
        )
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens_per_clue,
            temperature=temperature,
            top_k=top_k,
            stop_token_id=stop_token_id,
        )
        text = tokenizer.decode(output_ids[0].tolist())
        clues.extend(parse_generated_text(text))

        if len(clues) >= num_clues:
            break

    if not clues:
        clues.append(
            GeneratedClue(
                round_name="Unparsed",
                category="Raw Generation",
                dollar_value="No value",
                prompt="The model generated text, but it did not contain a parsed prompt.",
                answer="Try a later checkpoint or a higher max_tokens_per_clue value.",
                raw_text=text if "text" in locals() else "",
            )
        )

    return clues[:num_clues]


def generate_html_from_checkpoint(
    *,
    checkpoint_path: Path,
    out_path: Path,
    num_clues: int,
    max_tokens_per_clue: int,
    temperature: float,
    top_k: int | None,
    seed_text: str,
    open_in_browser: bool,
) -> Path:
    """Load a checkpoint, sample clues, write HTML, and optionally open it."""

    device = pick_device()
    model, tokenizer = load_model_for_generation(checkpoint_path, device=device)
    clues = generate_clues(
        model,
        tokenizer,
        num_clues=num_clues,
        max_tokens_per_clue=max_tokens_per_clue,
        temperature=temperature,
        top_k=top_k,
        seed_text=seed_text,
    )
    write_generated_clues_html(clues, out_path)

    if open_in_browser:
        webbrowser.open(out_path.resolve().as_uri())

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for `scripts/generate_html.py`."""

    parser = argparse.ArgumentParser(description="Generate an interactive HTML page.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/generated_jeopardy.html"))
    parser.add_argument("--num-clues", type=int, default=12)
    parser.add_argument("--max-tokens-per-clue", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed-text", type=str, default=ROUND_NAME)
    parser.add_argument("--open", action="store_true", help="Open the HTML page.")
    return parser


def main() -> None:
    """Parse CLI arguments and write a generated HTML page."""

    args = build_arg_parser().parse_args()
    top_k = args.top_k if args.top_k > 0 else None
    out_path = generate_html_from_checkpoint(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        num_clues=args.num_clues,
        max_tokens_per_clue=args.max_tokens_per_clue,
        temperature=args.temperature,
        top_k=top_k,
        seed_text=args.seed_text,
        open_in_browser=args.open,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
