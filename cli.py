import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline import clean, draft, embed, rank, rerank
from pipeline.fetch import fetch, print_samples
from pipeline.founders import FoundersError, get_founders, get_founders_with_source
from pipeline.llm import LLMError

ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = logging.getLogger(__name__)


def check_env(keys: list[str]) -> None:
    """Verify required env vars are set; raise SystemExit if not."""
    load_dotenv(ENV_PATH)
    missing = [k for k in keys if not os.environ.get(k, "").strip()]
    if missing:
        raise SystemExit(
            f"error: missing env vars: {', '.join(missing)}\n"
            "Add them to .env (see SPEC.md for details)."
        )


def cmd_fetch(args: argparse.Namespace) -> None:
    records = fetch()
    logger.info(f"Saved {len(records)} records to data/raw.json")
    print_samples(records, args.samples)


def cmd_clean(args: argparse.Namespace) -> None:
    clean.run()


def cmd_embed(args: argparse.Namespace) -> None:
    embed.run()


def cmd_rank(args: argparse.Namespace) -> None:
    try:
        profile_text = rank.load_profile(args.profile)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")
    try:
        results = rank.rank(profile_text)
    except rank.RankError as err:
        raise SystemExit(f"error: {err}")
    for r in results[:20]:
        logger.info(
            f"{r['name']} | {r['one_liner']} | team {r['team_size']} | "
            f"{r['batch']} | hiring {rank.format_hiring(r['is_hiring'])} | "
            f"{r['score']:.4f}"
        )


def cmd_rerank(args: argparse.Namespace) -> None:
    check_env(["GEMINI_API_KEY", "GEMINI_MODEL"])
    try:
        profile_text = rank.load_profile(args.profile)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")

    try:
        candidates = rank.rank(profile_text, hiring_only=args.hiring_only)
    except rank.RankError as err:
        raise SystemExit(f"error: {err}")

    try:
        results, rejected = rerank.rerank_with_stats(profile_text, candidates)
    except RuntimeError as err:
        raise SystemExit(f"error: {err}")

    logger.info(
        f"candidates sent: {len(candidates)} | "
        f"picks requested: {rerank.DEFAULT_TOP_N} | "
        f"picks returned: {len(results)} | rejected: {rejected}"
    )
    for r in results:
        team = r["team_size"] if r["team_size"] else "unknown"
        logger.info(
            f"{r['name']} | {r['one_liner']} | team {team} | "
            f"{r['batch']} | hiring {rank.format_hiring(r['is_hiring'])} | "
            f"{r['reason']}"
        )


def cmd_founders(args: argparse.Namespace) -> None:
    for slug in args.slugs:
        try:
            founders, source = get_founders_with_source(slug)
        except FoundersError as err:
            raise SystemExit(f"error: {err}")

        if not founders:
            logger.info(f"{slug} | (no active founders) | - | - | - | {source}")
            continue

        for f in founders:
            linkedin = f["linkedin_url"] or "-"
            twitter = f["twitter_url"] or "-"
            logger.info(
                f"{slug} | {f['full_name']} | {f['title']} | "
                f"{linkedin} | {twitter} | {f['source']}"
            )


def cmd_draft(args: argparse.Namespace) -> None:
    check_env(["GEMINI_API_KEY", "GEMINI_MODEL", "SENDER_NAME"])
    sender_name = os.environ["SENDER_NAME"].strip()
    links = os.environ.get("SENDER_LINKS", "").strip()

    try:
        profile_text = rank.load_profile(args.profile)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")

    try:
        company = draft.load_company(args.slug)
        founders = get_founders(args.slug)
    except (draft.DraftError, FoundersError) as err:
        raise SystemExit(f"error: {err}")

    if not founders:
        raise SystemExit(f"error: no active founders for '{args.slug}'")
    founder = founders[0]

    try:
        result = draft.draft(
            profile_text, company, founder, sender_name, links
        )
    except (draft.DraftError, LLMError) as err:
        raise SystemExit(f"error: {err}")

    linkedin = founder["linkedin_url"] or "-"
    twitter = founder["twitter_url"] or "-"
    logger.info(result["subject"])
    logger.info(result["body"])
    logger.info(f"To: {founder['full_name']} | {linkedin} | {twitter}")
    logger.info(f"words: {len(result['body'].split())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download the raw YC dataset")
    fetch_parser.add_argument("--samples", type=int, default=5)
    fetch_parser.set_defaults(func=cmd_fetch)

    clean_parser = sub.add_parser("clean", help="Filter raw data into SQLite")
    clean_parser.set_defaults(func=cmd_clean)

    embed_parser = sub.add_parser("embed", help="Embed companies into numpy files")
    embed_parser.set_defaults(func=cmd_embed)

    rank_parser = sub.add_parser("rank", help="Rank companies against a profile file")
    rank_parser.add_argument("profile", help="Path to a plain-text profile file")
    rank_parser.set_defaults(func=cmd_rank)

    rerank_parser = sub.add_parser("rerank", help="Re-rank top hits with the LLM")
    rerank_parser.add_argument("profile", help="Path to a plain-text profile file")
    rerank_parser.add_argument(
        "--hiring-only",
        action="store_true",
        help="Only consider companies that are hiring",
    )
    rerank_parser.set_defaults(func=cmd_rerank)

    founders_parser = sub.add_parser("founders", help="Look up founders for company slugs")
    founders_parser.add_argument("slugs", nargs="+", help="One or more YC company slugs")
    founders_parser.set_defaults(func=cmd_founders)

    draft_parser = sub.add_parser("draft", help="Draft an outreach email")
    draft_parser.add_argument("profile", help="Path to a plain-text profile file")
    draft_parser.add_argument("slug", help="YC company slug")
    draft_parser.set_defaults(func=cmd_draft)

    return parser


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
