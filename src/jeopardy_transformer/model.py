"""A small decoder-only Transformer language model.

This is the same broad family as GPT:

1. Convert token ids to vectors with an embedding table.
2. Add position embeddings so the model knows token order.
3. Repeatedly apply Transformer blocks.
4. Project the final vectors back to vocabulary logits.

The model is "decoder-only" because it predicts the next token from previous
tokens. There is no separate encoder, and attention is causal: position 10 can
look at positions 0..10, but not at position 11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Hyperparameters that determine the model's shape."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    A single attention head asks: "for this token, which previous tokens matter?"
    Multi-head attention repeats that question in parallel with different learned
    projections, giving the model several ways to relate tokens to each other.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Create the projections, dropout layers, and fixed causal mask."""

        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One projection creates queries, keys, and values at once for efficiency.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Shape: (1, 1, block_size, block_size), broadcast over batch and heads.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head attention to a batch of token vectors."""

        batch_size, seq_len, channels = x.shape

        # Project every token vector into query/key/value vectors. These three
        # tensors have the same shape at first: (batch, time, embedding).
        qkv = self.qkv_proj(x)
        query, key, value = qkv.split(channels, dim=-1)

        # Rearrange from (B, T, C) to (B, heads, T, head_dim).
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Dot products compare each query with each key. Dividing by sqrt(head_dim)
        # keeps the softmax from becoming too sharp early in training.
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values, then restore shape back to (B, T, C).
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    """The per-token MLP inside each Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the two-layer MLP used after attention."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same MLP independently to every token position."""

        return self.net(x)


class TransformerBlock(nn.Module):
    """One Transformer block: attention, MLP, and residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the normalization, attention, and MLP submodules."""

        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one Transformer block and return updated token vectors."""

        # Pre-norm layout: normalize before each sublayer, then add residual.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """A GPT-style language model."""

    def __init__(self, config: ModelConfig) -> None:
        """Create embeddings, Transformer blocks, final norm, and LM head."""

        super().__init__()
        self.config = config

        # Token embeddings learn a vector for each token id in the tokenizer.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embeddings let the model distinguish "first token" from
        # "twentieth token" even if the token ids are the same.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the matrix used to read token meanings is also used to
        # write token logits. This saves parameters and is standard in GPT-like LMs.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize Linear and Embedding weights with GPT-style small noise."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits, and optionally cross-entropy loss.

        `input_ids` has shape (batch, time). `logits` has shape
        (batch, time, vocab_size), where logits[b, t] tries to predict the next
        token after input_ids[b, t].
        """

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int | None = 50,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the model."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = top_values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if stop_token_id is not None and bool((next_token == stop_token_id).all()):
                break

        return input_ids

    def parameter_count(self) -> int:
        """Total trainable parameter count."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
