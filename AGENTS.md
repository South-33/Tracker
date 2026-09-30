This is the project's AGENTS.md

# Project rules

- Read `tracker.md` first.
- Optimize for the next real experiment, not framework completeness.
- Treat the current tracker bet as the incumbent. Do not change it merely to make progress: first measure it, identify a concrete weakness, and only replace or complicate it when a focused experiment gives evidence that the change is meaningfully better. "No change" is a valid result.
- Work with broad research freedom inside that constraint: inspect data and failure cases, research papers/implementations, question the current formulation, rewrite or remove parts of the approach, and run ambitious experiments when they have a clear hypothesis. Be conservative about what gets promoted, not about what gets investigated.
- Keep the active repo small. Delete stale code and use Git history as the archive.
- No legacy compatibility unless the current experiment needs it.
- Prefer simple direct code and few files.
- Heavy/generated data, weights and runs stay untracked.
- A result is only real when its settings, data and metrics are reproducible.
- Use exposed DanceTrack `0016` and `0020` for development. Keep `0096` and `0004/0005/0007/0010` untouched until a model is frozen.
