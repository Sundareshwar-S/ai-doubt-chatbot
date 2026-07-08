# CLAUDE.md

## Git workflow

- All work happens on the **`develop`** branch. Commit changes there, not on `main`.
- `main` is reserved for stable/released snapshots only — do not commit directly to `main` unless the user explicitly asks for a release merge.
- After every commit to `develop`, push it to `origin/develop` right away — don't wait to be asked.

See `plan.md` for the project architecture/phased plan and `tasks.md` for the current task checklist.
