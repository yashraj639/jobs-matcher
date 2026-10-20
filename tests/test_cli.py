import pytest

import cli
from pipeline import clean as clean_module


def test_build_parser_rank() -> None:
    args = cli.build_parser().parse_args(["rank", "profile.txt"])

    assert args.command == "rank"
    assert args.profile == "profile.txt"
    assert args.func is cli.cmd_rank


def test_build_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_check_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YC_JOBS_TEST_MISSING_KEY", raising=False)

    with pytest.raises(SystemExit, match="missing env vars"):
        cli.check_env(["YC_JOBS_TEST_MISSING_KEY"])


def test_check_env_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YC_JOBS_TEST_PRESENT_KEY", "value")

    cli.check_env(["YC_JOBS_TEST_PRESENT_KEY"])


def test_cmd_clean_maps_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run() -> None:
        raise clean_module.CleanError("run fetch first")

    monkeypatch.setattr(clean_module, "run", fake_run)
    args = cli.build_parser().parse_args(["clean"])

    with pytest.raises(SystemExit, match="run fetch first"):
        cli.cmd_clean(args)
