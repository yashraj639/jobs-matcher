from pipeline.llm import _strip_fences

EXPECTED = '{"a": 1}'


def test_strip_fences_plain_json() -> None:
    assert _strip_fences(EXPECTED) == EXPECTED


def test_strip_fences_with_language() -> None:
    text = f"```json\n{EXPECTED}\n```"
    assert _strip_fences(text) == EXPECTED


def test_strip_fences_without_language() -> None:
    text = f"```\n{EXPECTED}\n```"
    assert _strip_fences(text) == EXPECTED


def test_strip_fences_without_newline() -> None:
    text = f"```json{EXPECTED}```"
    assert _strip_fences(text) == EXPECTED
