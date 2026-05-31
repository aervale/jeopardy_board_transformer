"""Tokenizer helpers.

The model is trained from scratch, but it still needs a way to turn text into
integers. We use `tiktoken` for the normal text pieces and add a few special
Jeopardy field markers as indivisible tokens.

Why special tokens?
-------------------
Without special tokens, the string "<PROMPT>" is split into smaller pieces like
"<", "PROM", "PT", ">". Giving each field marker its own id makes the dataset
structure obvious to the model and to us when we inspect generated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken


ROUND_NAME = "<ROUND_NAME>"
CATEGORY = "<CATEGORY>"
DOLLAR_VALUE = "<DOLLAR_VALUE>"
PROMPT = "<PROMPT>"
ANSWER = "<ANSWER>"
AIR_DATE = "<AIR_DATE>"
SHOW_NUMBER = "<SHOW_NUMBER>"
END_OF_EXAMPLE = "<END_OF_EXAMPLE>"

JEOPARDY_SPECIAL_TOKENS: tuple[str, ...] = (
    ROUND_NAME,
    CATEGORY,
    DOLLAR_VALUE,
    PROMPT,
    ANSWER,
    AIR_DATE,
    SHOW_NUMBER,
    END_OF_EXAMPLE,
)


@dataclass(frozen=True)
class JeopardyTokenizer:
    """A tiny wrapper around a `tiktoken.Encoding`.

    The wrapper keeps the rest of the code from needing to remember
    `allowed_special=...` every time it encodes text.
    """

    encoding: tiktoken.Encoding
    special_tokens: tuple[str, ...] = JEOPARDY_SPECIAL_TOKENS

    @property
    def vocab_size(self) -> int:
        """Number of token ids the model's embedding table must support."""

        return self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids, allowing our custom Jeopardy tokens."""

        return self.encoding.encode(text, allowed_special=set(self.special_tokens))

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token ids back to text."""

        return self.encoding.decode(list(token_ids))


def build_tokenizer(base_encoding_name: str = "gpt2") -> JeopardyTokenizer:
    """Create a GPT-2 `tiktoken` encoding extended with Jeopardy tags.

    GPT-2's vocabulary is smaller than newer OpenAI encodings, so the model's
    input and output embedding matrices are smaller too. That helps keep this
    educational model laptop-friendly.
    """

    base = tiktoken.get_encoding(base_encoding_name)

    # `tiktoken` does not expose a public "add_special_tokens" method, so the
    # standard approach is to build a new Encoding from the base internals.
    next_id = base.n_vocab
    jeopardy_specials = {
        token: next_id + i for i, token in enumerate(JEOPARDY_SPECIAL_TOKENS)
    }

    encoding = tiktoken.Encoding(
        name=f"{base.name}_jeopardy",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **jeopardy_specials},
    )
    return JeopardyTokenizer(encoding=encoding)

