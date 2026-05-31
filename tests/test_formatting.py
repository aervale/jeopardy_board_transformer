from jeopardy_transformer.data import format_jeopardy_row
from jeopardy_transformer.generate import parse_generated_text
from jeopardy_transformer.special_tokens import (
    ANSWER,
    CATEGORY,
    DOLLAR_VALUE,
    END_OF_EXAMPLE,
    PROMPT,
    ROUND_NAME,
)


def test_format_jeopardy_row_uses_special_tokens():
    """Formatted rows should expose Jeopardy fields as special-token lines."""

    row = {
        "round": "Jeopardy!",
        "og-category": "HISTORY",
        "value": "$200",
        "question": "This scientist studied gravity.",
        "answer": "Newton",
        "air_date": "2004-12-31",
        "show_number": "4680",
    }

    text = format_jeopardy_row(row)

    assert f"{ROUND_NAME} Jeopardy!" in text
    assert f"{CATEGORY} HISTORY" in text
    assert f"{DOLLAR_VALUE} $200" in text
    assert f"{PROMPT} This scientist studied gravity." in text
    assert f"{ANSWER} Newton" in text
    assert text.endswith(f"{END_OF_EXAMPLE}\n")


def test_parse_generated_text_round_trips_structured_sample():
    """Generated structured text should parse back into one clue object."""

    text = (
        f"{ROUND_NAME} Jeopardy!\n"
        f"{CATEGORY} SCIENCE\n"
        f"{DOLLAR_VALUE} $400\n"
        f"{PROMPT} This planet is known as the Red Planet.\n"
        f"{ANSWER} Mars\n"
        f"{END_OF_EXAMPLE}\n"
    )

    clues = parse_generated_text(text)

    assert len(clues) == 1
    assert clues[0].round_name == "Jeopardy!"
    assert clues[0].category == "SCIENCE"
    assert clues[0].dollar_value == "$400"
    assert clues[0].prompt == "This planet is known as the Red Planet."
    assert clues[0].answer == "Mars"
