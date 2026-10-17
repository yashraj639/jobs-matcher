import argparse

from pipeline import clean, embed, rank
from pipeline.fetch import fetch, print_samples


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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
