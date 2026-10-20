import pytest

from pipeline import draft as draft_module
from pipeline.llm import LLMParseError

PROFILE = "Software engineer building TypeScript developer tools."
COMPANY = {
    "name": "Windmill",
    "one_liner": "Turn scripts into internal apps",
    "long_description": "Windmill turns scripts into internal apps and workflows.",
}
FOUNDER = {"full_name": "Ruben Fiszel"}
VALID_DRAFT = {
    "subject": "Windmill workflow tooling",
    "body": (
        "I build TypeScript developer tools for internal teams. "
        "Your script-to-app workflow overlaps with the tooling I have built. "
        "Would you be open to a short chat?"
    ),
}


def test_validate_bans_both_hope_phrases() -> None:
    phrases = [
        "I hope this finds you well",
        "I hope this email finds you well",
    ]
    for phrase in phrases:
        reasons = draft_module._validate(
            {
                "subject": "Windmill workflow tooling",
                "body": f"{phrase}. I build developer tools. Want to chat?",
            }
        )
        assert any("hope this" in reason for reason in reasons)


def test_draft_retries_after_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_generate_json(prompt: str) -> dict:
        calls.append(prompt)
        if len(calls) == 1:
            raise LLMParseError("Gemini did not return valid JSON")
        return dict(VALID_DRAFT)

    monkeypatch.setattr(draft_module, "generate_json", fake_generate_json)

    result = draft_module.draft(PROFILE, COMPANY, FOUNDER, "Yash", "")

    assert result["subject"] == VALID_DRAFT["subject"]
    assert len(calls) == 2
    assert "Previous attempt failed" in calls[1]


def test_draft_retries_when_hook_marker_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    responses = [
        {
            "subject": "Windmill workflow tooling",
            "body": (
                f"{draft_module.HOOK_MARKER} I build developer tools. "
                "Want to chat?"
            ),
        },
        dict(VALID_DRAFT),
    ]

    def fake_generate_json(prompt: str) -> dict:
        calls.append(prompt)
        return responses[len(calls) - 1]

    monkeypatch.setattr(draft_module, "generate_json", fake_generate_json)

    result = draft_module.draft(PROFILE, COMPANY, FOUNDER, "Yash", "")

    assert draft_module.HOOK_MARKER not in result["body"]
    assert result["body"] == VALID_DRAFT["body"]
    assert len(calls) == 2
    assert "Previous attempt failed" in calls[1]


def test_draft_raises_when_hook_marker_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_generate_json(prompt: str) -> dict:
        calls.append(prompt)
        return {
            "subject": "Windmill workflow tooling",
            "body": (
                f"{draft_module.HOOK_MARKER} I build developer tools. "
                "Want to chat?"
            ),
        }

    monkeypatch.setattr(draft_module, "generate_json", fake_generate_json)

    with pytest.raises(draft_module.DraftError, match="failed validation"):
        draft_module.draft(PROFILE, COMPANY, FOUNDER, "Yash", "")

    assert len(calls) == 2
