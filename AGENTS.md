Project spec is in SPEC.md. Read it before starting any step, and do only the step I name.

## Rules for the AI

- Do ONE step at a time. Never build ahead of the step I asked for.
- Python only. Use `uv` for everything (`uv add`, `uv run`), never `pip`. No new dependencies without asking.
- Pipeline code and UI code stay separate. The UI only calls functions from `pipeline/`.
- No giant files. One job per module, functions under ~40 lines.
- After each step, tell me how to run it and what output I should see.
- Never auto-send emails. Drafts only.
- Don't invent fields. If the dataset doesn't have something, say so.
- Don't "fix" things I didn't ask about.

## Git rules

- Never run `git commit`, `git push`, `git reset`, `git checkout`, `git clean`, or `git rebase`. I handle all of that myself or ill tell you when and what to do.
- You may run `git status` and `git diff` to check what you changed.
- Never delete or overwrite files I didn't ask you to touch.
- Never commit or output secrets. API keys go in `.env`, read via environment variables.
- `.gitignore` must always include: `.venv/`, `data/`, `__pycache__/`, `.env`.
- When a step is done, list the files you created or changed so I can review the diff.
- Suggest a commit message in conventional commit format (feat/fix/refactor/docs/chore), but don't run the commit.