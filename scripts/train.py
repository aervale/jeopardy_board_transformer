"""Convenience wrapper for `python -m jeopardy_transformer.train`."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jeopardy_transformer.train import main


if __name__ == "__main__":
    main()

