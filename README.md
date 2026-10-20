# yc-jobs

Finds YC startups that fit a developer's profile, shows the founders, and drafts a short outreach message. Drafts only, it never sends anything.

## How it works

1. **fetch**: downloads the YC company dataset (`all.json`) and saves it to `data/raw.json`.
2. **clean**: keeps active companies with a description, drops any flagged as not hiring, and stores the result in SQLite (`data/yc.db`).
3. **embed**: builds one text string per company and embeds them locally with `all-MiniLM-L6-v2`.
4. **rank**: embeds the profile and ranks companies by cosine similarity, top 50.
5. **rerank**: sends the top candidates and the profile to Gemini in one call, asks for the best picks with a one-sentence reason each.
6. **founders**: looks up active founders from the company's YC page (name, title, LinkedIn, X).
7. **draft**: sends the profile, company, and founder to Gemini in one call to draft a short email.

Embeddings are local, so ranking costs nothing. Only rerank and draft call the LLM.

## Setup

```
uv sync
```

The first `embed` run downloads a ~90MB model, and `uv sync` installs PyTorch, so setup takes a few minutes.

Create a `.env` file in the project root:

```
GEMINI_API_KEY=
GEMINI_MODEL=
SENDER_NAME=
SENDER_LINKS=
```

- `GEMINI_API_KEY`: key from Google AI Studio.
- `GEMINI_MODEL`: model name exactly as shown in Google AI Studio.
- `SENDER_NAME`: your name, used in the draft.
- `SENDER_LINKS`: your links, used in the draft.

Then restrict the key file to your user: `chmod 600 .env`.

Write your profile as plain text in `profile.txt` (skills, projects, what you're looking for). The rank, rerank, and draft steps read this file. It is capped at 15,000 characters. Profile files (`profile*.txt`) are gitignored.

`rerank` needs `GEMINI_API_KEY` and `GEMINI_MODEL`; `draft` needs those plus `SENDER_NAME`. The commands check for them up front and exit with an error if any are missing.

## Usage

Run the pipeline in order. `rank` and `rerank` need the database from `clean` and the embeddings from `embed`; `draft` needs the database. If you re-run `clean` and the company ids change, `rank` fails with a clear error telling you to re-run `embed`. Missing data files also name the command to run first.

```
uv run python cli.py fetch
uv run python cli.py clean
uv run python cli.py embed
uv run python cli.py rank profile.txt
uv run python cli.py rerank profile.txt
uv run python cli.py rerank profile.txt --hiring-only
uv run python cli.py founders <slug> [<slug> ...]
uv run python cli.py draft profile.txt <slug>
```

- `fetch` takes `--samples N` (default 5).
- `rank` logs the top 20 matches. `rerank` logs the LLM's picks (up to 15) with a reason each.
- `rerank --hiring-only` only considers companies flagged as hiring.
- `founders` takes one or more YC company slugs.
- `draft` logs the subject, body, founder contact, and word count.
- Status and result lines go through Python `logging` at INFO to stderr. Each module uses its own logger, so lines look like `INFO pipeline.clean: ...` or `INFO pipeline.fetch: ...`; the CLI itself logs as `INFO __main__:` when run as `python cli.py`.

## Example output

`rerank --hiring-only` on a full-stack TypeScript profile (from an earlier run, before the switch to logging; the `INFO cli:` prefixes are omitted). Truncated to three rows:

```
candidates sent: 50 | picks requested: 15 | picks returned: 12 | rejected: 0
Windmill | Open-source platform to turn scripts into internal apps & workflows | team 7 | Summer 2022 | hiring 1 | Windmill is a 7-person team building an open-source developer tool where users write scripts in TypeScript and SQL to auto-generate UIs and workflows.
Alguna | AI-Native CPQ & Billing for Modern Pricing Models | team 10 | Summer 2023 | hiring 1 | Alguna is an early-stage 10-person startup building billing and payment collection infrastructure for B2B SaaS companies.
Lago | The AI Native Billing Platform | team 27 | Summer 2021 | hiring 1 | Lago builds an open-source billing platform to manage pricing models and billing infrastructure for SaaS companies.
```

## Numbers (as of Oct 17, 2026)

- 6,307 companies fetched.
- 4,334 active with a description.
- The clean step also drops companies flagged as not hiring, leaving 1,482 in the database.

## Design notes

- One LLM call per rerank and one per draft.
- Company embeddings are cached to disk and only recomputed when the set of company IDs changes.
- Founder lookups are cached in SQLite for 7 days, empty results included, and rate-limited (2s between requests).
- Drafts are validated in code: word limit, subject length, and banned phrases. Validation or JSON parse failures retry once.
- Nothing is ever sent automatically. The output is text you copy yourself.
- Your profile text and company descriptions are sent to the Gemini API. On the free tier, Google may use requests to improve its products, so keep contact info out of `profile.txt`.

## Limitations

- The dataset goes stale. There is no refresh job; re-run `fetch` and `clean` to update.
- `isHiring` can lag reality, so a flagged company may not actually be hiring.
- MiniLM truncates long text, so only the start of a long description affects the ranking.
- The profile quality drives the results; a thin profile gives generic matches.
- Founder pages are scraped lightly, for personal use, with a cache and delay.

## Development

```
uv run pytest
uv run mypy .
```

## Roadmap

Not built yet:

- Streamlit UI (upload, ranked list, editable draft).
- Indian startups beyond the YC dataset.
- An eval set to score match quality.