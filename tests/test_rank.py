import pytest

from pipeline.rank import MAX_PROFILE_CHARS, load_profile


def test_load_profile_success(tmp_path):
    p = tmp_path / "profile.txt"
    p.write_text("  Software engineer with 5 years in backend systems.  ")
    result = load_profile(p)
    assert result == "Software engineer with 5 years in backend systems."


def test_load_profile_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nonexistent.txt")


def test_load_profile_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n  \t  ")
    with pytest.raises(ValueError, match="empty"):
        load_profile(p)


def test_load_profile_too_long(tmp_path):
    p = tmp_path / "long.txt"
    p.write_text("x" * (MAX_PROFILE_CHARS + 1))
    with pytest.raises(ValueError, match="too long"):
        load_profile(p)
