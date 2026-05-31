# Jeopardy Transformer

A small, readable decoder-only Transformer trained from scratch on
[`soldni/jeopardy`](https://huggingface.co/datasets/soldni/jeopardy), using
`tiktoken` plus a few Jeopardy-specific special tokens.

The project is written to be understandable if you know the early deep-learning
building blocks from CS231n and have watched Karpathy-style Transformer
walkthroughs. The code favors clear names and comments over maximum cleverness.

## What It Does

1. Downloads the `all_questions` split from Hugging Face.
2. Reformats each clue into a plain text record with custom tokens:

   ```text
   <ROUND_NAME> Jeopardy!
   <CATEGORY> HISTORY
   <DOLLAR_VALUE> $200
   <PROMPT> For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory
   <ANSWER> Copernicus
   <END_OF_EXAMPLE>
   ```

3. Encodes those records with `tiktoken` and writes compact `.bin` token files.
4. Trains a small GPT-style Transformer language model.
5. Saves checkpoints to `checkpoints/latest.pt` and periodic numbered files.
6. Generates new clues and writes an interactive HTML page where each prompt can
   be clicked to reveal its answer.

## Project Layout

```text
jeopardy-transformer/
  configs/
    tiny.json                  # A laptop-friendly default config
  scripts/
    prepare_data.py            # Download, reformat, tokenize
    train.py                   # Train or resume training
    sweep.py                   # Try a small hyperparameter grid
    generate_html.py           # Sample the model and create HTML
  src/jeopardy_transformer/
    checkpointing.py           # Save/load checkpoint files
    data.py                    # Dataset formatting and token cache building
    generate.py                # Sampling + parsed clue generation
    html.py                    # Interactive HTML rendering
    metrics.py                 # Verbose CSV logs and plots
    model.py                   # Decoder-only Transformer
    special_tokens.py          # tiktoken wrapper with custom tokens
    sweep.py                   # Validation-sweep runner
    train.py                   # Training loop
```

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you have an Apple Silicon Mac, PyTorch will usually use `mps` automatically.
MPS runs in normal full precision in this project because many PyTorch versions
do not support `torch.autocast(device_type="mps")`. If you have an NVIDIA GPU,
CUDA still uses automatic mixed precision. Otherwise the code runs on CPU, just
more slowly.

## Prepare The Data

```bash
python3 scripts/prepare_data.py --out-dir data/jeopardy
```

Useful smaller run while learning the code:

```bash
python3 scripts/prepare_data.py --out-dir data/jeopardy_tiny --max-examples 5000
```

If you prepare the tiny folder, train with the matching data directory:

```bash
python3 scripts/train.py \
  --config configs/tiny.json \
  --data-dir data/jeopardy_tiny \
  --checkpoints-dir checkpoints_tiny
```

By default the prepare step writes both:

- `train.bin` / `val.bin`: compact token files used by training.
- `formatted_train.jsonl` / `formatted_val.jsonl`: human-readable reformatted
  examples so you can inspect exactly what the model sees.

Use `--skip-formatted-jsonl` if you want to save disk space.

## Train

```bash
python3 scripts/train.py \
  --config configs/tiny.json \
  --data-dir data/jeopardy \
  --checkpoints-dir checkpoints
```

At startup, training prints the exact `train.py` source file, config path, data
directory, and checkpoint directory it is using. If edits ever seem ignored,
that first `Training source:` line is the path to check.

Resume from the latest checkpoint:

```bash
python3 scripts/train.py \
  --config configs/tiny.json \
  --data-dir data/jeopardy \
  --checkpoints-dir checkpoints \
  --resume checkpoints/latest.pt
```

The default config is intentionally modest. It is meant to be readable and
trainable on normal hardware, not state of the art.

For fewer checkpoint files while experimenting:

```bash
python3 scripts/train.py \
  --config configs/tiny.json \
  --data-dir data/jeopardy_tiny \
  --checkpoints-dir checkpoints_tiny \
  --no-numbered-checkpoints
```

For verbose metrics and plots:

```bash
python3 scripts/train.py \
  --config configs/tiny.json \
  --data-dir data/jeopardy_tiny \
  --checkpoints-dir checkpoints_tiny \
  --verbose
```

Verbose mode writes:

- `checkpoints_tiny/metrics/history.csv`
- `checkpoints_tiny/metrics/history.jsonl`
- `checkpoints_tiny/metrics/loss_vs_epoch.png`
- `checkpoints_tiny/metrics/scale_metrics.png`

The scale plot includes weight RMS, gradient RMS, and an estimated update/weight
ratio. Because AdamW rescales updates internally, that ratio is a useful estimate
rather than the exact optimizer update.

## Pick Hyperparameters

Full k-fold cross-validation is usually not the best first tool for language
models because it multiplies training cost by the number of folds. A cheaper
workflow is a validation sweep: keep the same validation set, run several short
training jobs, and pick the config with the lowest validation loss.

Run the included tiny sweep:

```bash
python3 scripts/sweep.py \
  --base-config configs/tiny.json \
  --sweep-config configs/sweep_tiny.json \
  --data-dir data/jeopardy_tiny \
  --out-dir checkpoints/sweeps \
  --verbose
```

The sweep writes `checkpoints/sweeps/sweep_results.csv`, sorted by best
validation loss. Each trial also keeps its own `best.pt` and `latest.pt`.

## Generate An Interactive HTML Page

```bash
python3 scripts/generate_html.py \
  --checkpoint checkpoints/latest.pt \
  --out outputs/generated_jeopardy.html \
  --num-clues 12 \
  --open
```

Each generated clue appears as a card. Click the clue or the reveal button to
show the answer.

## Notes On The Tokenizer

This project uses `tiktoken` with the GPT-2 base encoding because it has a
smaller vocabulary than newer OpenAI encodings, which keeps the embedding layer
smaller. The Jeopardy tags are added as true special tokens, so `<PROMPT>` is one
token instead of being split into punctuation and letters.

## Reasonable Storage Defaults

The source code is tiny. Downloaded data, token caches, checkpoints, and HTML
outputs are ignored by git and kept in dedicated folders:

- `data/`
- `checkpoints/`
- `outputs/`

That keeps the project easy to zip, move, and inspect.
