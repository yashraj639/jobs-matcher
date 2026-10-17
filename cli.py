import argparse

from pipeline.fetch import fetch, print_samples


def cmd_fetch(args: argparse.Namespace) -> None:
    records = fetch()
    print(f"Saved {len(records)} records to data/raw.json")
    print_samples(records, args.samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Download the raw YC dataset")
    fetch_parser.add_argument("--samples", type=int, default=5)
    fetch_parser.set_defaults(func=cmd_fetch)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
