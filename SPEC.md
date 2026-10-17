# Project: Job Matcher

Upload a resume, get YC startups ranked by fit, see the founder, get a short outreach draft.
Built for personal use first, shareable later. Keep it simple.

## Rules for the AI (read every time)

- Do ONE step at a time. Never build ahead of the step I asked for.
- Python only. No new dependencies without asking.
- Pipeline code and UI code stay separate. The UI only calls functions from `pipeline/`.
- No giant files. One job per module, functions under ~40 lines.
- After each step, tell me how to run it and what output I should see.
- Never auto-send emails. Drafts only.
- Don't invent fields. If the dataset doesn't have something, say so.
- Don't "fix" things I didn't ask about.

## Stack

- Python 3.11+
- SQLite (stdlib `sqlite3`), no ORM
- `sentence-transformers` (local embeddings, model: `all-MiniLM-L6-v2`)
- `numpy` for cosine similarity (6k rows doesn't need a vector DB)
- LLM API for re-rank and drafts, behind one function so the model is swappable
- Streamlit for the UI (Phase 2)
- `pypdf` for resume parsing

## Folder layout

```
yc-matcher/
  SPEC.md
  data/                # raw dumps + the sqlite file (gitignored)
  pipeline/
    fetch.py           # download dataset
    clean.py           # filter + normalize
    embed.py           # company + resume embeddings
    rank.py            # cosine similarity, top N
    rerank.py          # LLM re-rank of top N
    founders.py        # founder lookup
    draft.py           # outreach email drafting
    llm.py             # single wrapper around whatever LLM we call
  app.py               # Streamlit UI (Phase 2)
  cli.py               # run the whole pipeline from terminal (Phase 1)
  tests/
```

## Phase 1: CLI, just me

Goal: `python cli.py resume.pdf` prints the top 20 companies with a reason and a draft email for each.

### Step 1: Fetch
- Pull the YC company dataset (yc-oss GitHub dump, `all.json`). Save raw JSON to `data/raw.json`.
- Print the total count and 5 sample records, with ALL fields visible.
- **Checkpoint:** I look at the fields. Is there a long description? Tags? Team size? Hiring flag? Founders? Write down what's missing.

### Step 2: Clean
- Keep only companies that are active. Keep `isHiring == true` if that field exists.
- Drop records with an empty description.
- Store in SQLite table `companies` (id, name, slug, one_liner, long_description, tags, industries, team_size, batch, website, is_hiring).
- Print: count before, count after.
- **Checkpoint:** The after-count should be a sane fraction of the total. If it dropped to 12 or didn't drop at all, the filter is wrong.

### Step 3: Embed
- Build one text string per company: name + one_liner + long_description + tags.
- Embed all of them once, store as a numpy file or a BLOB column. Never recompute unless data changes.
- **Checkpoint:** Embedding count equals company count. Second run should be instant (cached).

### Step 4: Rank
- Parse resume PDF to text, embed it, cosine similarity against all companies, return top 50.
- Print top 20 with name, one-liner, score.
- **Checkpoint:** Do these make sense for MY resume? If it's garbage, fix this step before moving on. Common bug: sorted ascending instead of descending.

### Step 5: Re-rank with LLM
- Send resume summary + top 50 company blurbs to the LLM. Ask it to pick the best 15-20 and give a one-sentence reason each, based only on what's in the data.
- Output must be JSON. Validate it, retry once if it fails to parse.
- **Checkpoint:** Reasons are specific, not "this company aligns with your skills."

### Step 6: Founders
- Check whether the dataset has founder names/links. If not, fetch from the company's YC page, politely: cache results, delay between requests, no hammering.
- Store founder name + LinkedIn/X if available.
- **Checkpoint:** Spot check 5 companies by hand.

### Step 7: Draft
- Inputs: resume summary, company blurb, founder name.
- Output: under 120 words, plain tone, references ONE specific thing about the company, ends with one clear ask. No "I hope this email finds you well," no fake flattery.
- **Checkpoint:** Would I actually send this? If it reads like a LinkedIn bot, fix the prompt.

## Phase 2: Streamlit UI

- Upload resume, show ranked list, click a company to see founder, reason, draft.
- Draft is editable and has a copy button. No send button.
- Resume is held in memory only, never written to disk. Say that on the page.
- Filters: batch, industry, team size.

## Phase 3: Only if people actually want it

- Auth, per-user rate limits, cost caps on LLM calls
- Daily data refresh job
- Swap Streamlit for FastAPI + Next.js if needed (pipeline is already separate, so this is painless)
- A real angle beyond "cloned the tweet": roles you fit, better drafts, or non-YC companies

## Known risks

- Dataset goes stale. Record the fetch date and show it in the UI.
- Dead startups slipping through. Don't trust the status field blindly, spot check.
- Drafts sounding like AI slop. Prompt quality matters more than model size here.
- Resumes are sensitive. Minimal retention, no logging resume text.
- LLM cost per user. Embeddings are free (local), the re-rank and drafts are the only paid calls.

## Definition of done for Phase 1

I run the CLI on my own resume, the top 10 are companies I'd genuinely apply to, and at least 3 drafts are good enough to send with minor edits.