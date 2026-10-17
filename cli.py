import argparse

from pipeline import clean, embed, rank, rerank
from pipeline.fetch import fetch, print_samples
from pipeline.founders import FoundersError, get_founders


def cmd_fetch(args: argparse.Namespace) -> None:
    records = fetch()
    print(f"Saved {len(records)} records to data/raw.json")
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
    results = rank.rank(profile_text)
    for r in results[:20]:
        print(
            f"{r['name']} | {r['one_liner']} | team {r['team_size']} | "
            f"{r['batch']} | hiring {r['is_hiring']} | {r['score']:.4f}"
        )


def cmd_rerank(args: argparse.Namespace) -> None:
    try:
        profile_text = rank.load_profile(args.profile)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")

    if args.hiring_only:
        hiring = [c for c in rank.rank(profile_text, top_n=300) if c["is_hiring"]]
        candidates = hiring[:50]
    else:
        candidates = rank.rank(profile_text)

    try:
        results, rejected = rerank.rerank_with_stats(profile_text, candidates)
    except RuntimeError as err:
        raise SystemExit(f"error: {err}")

    print(
        f"candidates sent: {len(candidates)} | "
        f"picks requested: {rerank.DEFAULT_TOP_N} | "
        f"picks returned: {len(results)} | rejected: {rejected}"
    )
    for r in results:
        team = r["team_size"] if r["team_size"] else "unknown"
        print(
            f"{r['name']} | {r['one_liner']} | team {team} | "
            f"{r['batch']} | hiring {r['is_hiring']} | {r['reason']}"
        )


def cmd_founders(args: argparse.Namespace) -> None:
    for slug in args.slugs:
        try:
            founders = get_founders(slug)
        except FoundersError as err:
            raise SystemExit(f"error: {err}")

        if not founders:
            print(f"{slug} | (no active founders) | - | - | - | fetched")
            continue

        for f in founders:
            linkedin = f["linkedin_url"] or "-"
            twitter = f["twitter_url"] or "-"
            print(
                f"{slug} | {f['full_name']} | {f['title']} | "
                f"{linkedin} | {twitter} | {f['source']}"
            )


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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
