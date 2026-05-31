"""Render generated clues as an interactive HTML page."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratedClue:
    """One generated Jeopardy-style clue."""

    round_name: str
    category: str
    dollar_value: str
    prompt: str
    answer: str
    raw_text: str


def _escape(value: str) -> str:
    """HTML-escape user/model text before inserting it into the page."""

    return html.escape(value, quote=True)


def render_generated_clues_html(
    clues: list[GeneratedClue],
    *,
    title: str = "Generated Jeopardy Clues",
) -> str:
    """Create a standalone HTML document with clickable answer reveals."""

    cards = []
    for index, clue in enumerate(clues, start=1):
        cards.append(
            f"""
      <article class="clue-card" tabindex="0" data-card>
        <div class="clue-topline">
          <span class="round">{_escape(clue.round_name or "Generated Round")}</span>
          <span class="value">{_escape(clue.dollar_value or "No value")}</span>
        </div>
        <p class="category">{_escape(clue.category or "Generated Category")}</p>
        <h2>{_escape(clue.prompt or "Generated clue could not be parsed cleanly.")}</h2>
        <div class="answer-row" aria-live="polite">
          <span class="answer-label">Answer</span>
          <p class="answer" hidden>{_escape(clue.answer or "No parsed answer")}</p>
        </div>
        <button class="reveal-button" type="button" data-reveal>
          <span class="show-label">Reveal</span>
          <span class="hide-label">Hide</span>
        </button>
        <details>
          <summary>Raw sample {index}</summary>
          <pre>{_escape(clue.raw_text)}</pre>
        </details>
      </article>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #181a20;
      --muted: #5e6472;
      --paper: #f7f5ef;
      --panel: #ffffff;
      --line: #d8d2c4;
      --blue: #153a8a;
      --blue-2: #244fb0;
      --gold: #d59b2d;
      --green: #26745f;
      --shadow: 0 10px 30px rgba(31, 35, 48, 0.10);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(rgba(255, 255, 255, 0.62), rgba(255, 255, 255, 0.62)),
        repeating-linear-gradient(
          135deg,
          #f4efe2 0,
          #f4efe2 14px,
          #eee7d6 14px,
          #eee7d6 28px
        );
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    header {{
      padding: 32px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(12px);
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 4.25rem);
      line-height: 0.95;
      letter-spacing: 0;
    }}

    main {{
      width: min(1180px, calc(100% - 28px));
      margin: 24px auto 48px;
    }}

    .clue-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      align-items: stretch;
    }}

    .clue-card {{
      display: grid;
      grid-template-rows: auto auto 1fr auto auto;
      gap: 14px;
      min-height: 330px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      outline: none;
      transition: transform 160ms ease, border-color 160ms ease;
    }}

    .clue-card:hover,
    .clue-card:focus-visible {{
      transform: translateY(-2px);
      border-color: var(--blue-2);
    }}

    .clue-topline {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      min-height: 32px;
    }}

    .round,
    .value,
    .answer-label {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      white-space: nowrap;
    }}

    .round {{
      color: #ffffff;
      background: var(--blue);
    }}

    .value {{
      color: #221604;
      background: var(--gold);
    }}

    .category {{
      margin: 0;
      color: var(--green);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }}

    h2 {{
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.38;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }}

    .answer-row {{
      min-height: 56px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}

    .answer-label {{
      color: var(--muted);
      background: #ece8de;
    }}

    .answer {{
      margin: 10px 0 0;
      color: var(--blue);
      font-size: 1.05rem;
      font-weight: 800;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }}

    .reveal-button {{
      width: 100%;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      color: #ffffff;
      background: var(--blue-2);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}

    .reveal-button:hover {{
      background: var(--blue);
    }}

    .hide-label,
    .clue-card.is-revealed .show-label {{
      display: none;
    }}

    .clue-card.is-revealed .hide-label {{
      display: inline;
    }}

    details {{
      color: var(--muted);
      font-size: 0.85rem;
    }}

    summary {{
      cursor: pointer;
    }}

    pre {{
      max-height: 180px;
      overflow: auto;
      padding: 10px;
      border-radius: 6px;
      background: #f0ede6;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{_escape(title)}</h1>
  </header>
  <main>
    <section class="clue-grid">
{''.join(cards)}
    </section>
  </main>
  <script>
    function setRevealed(card, revealed) {{
      const answer = card.querySelector(".answer");
      card.classList.toggle("is-revealed", revealed);
      answer.hidden = !revealed;
    }}

    for (const card of document.querySelectorAll("[data-card]")) {{
      const button = card.querySelector("[data-reveal]");

      button.addEventListener("click", (event) => {{
        event.stopPropagation();
        setRevealed(card, !card.classList.contains("is-revealed"));
      }});

      card.addEventListener("click", (event) => {{
        if (event.target.closest("details")) return;
        setRevealed(card, !card.classList.contains("is-revealed"));
      }});

      card.addEventListener("keydown", (event) => {{
        if (event.key !== "Enter" && event.key !== " ") return;
        if (event.target.closest("button, details")) return;
        event.preventDefault();
        setRevealed(card, !card.classList.contains("is-revealed"));
      }});
    }}
  </script>
</body>
</html>
"""


def write_generated_clues_html(
    clues: list[GeneratedClue],
    out_path: Path,
    *,
    title: str = "Generated Jeopardy Clues",
) -> Path:
    """Write the standalone HTML file and return its path."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_generated_clues_html(clues, title=title),
        encoding="utf-8",
    )
    return out_path
