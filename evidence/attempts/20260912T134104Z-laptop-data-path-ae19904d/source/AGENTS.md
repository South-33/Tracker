This is the project's AGENTS.md

- Start with `GOAL.md`, `tracker.md`, and `.\lab.ps1 status`. The goal owns intent/acceptance; tracker owns current scientific facts; the ledger owns active execution state. Do not duplicate the experiment graveyard into startup context.
- The priority is a substantial path to the useful 80% regime, not a stream of safe 1% gains. At most three exploratory experiments, including train-only diagnostics, then a substantial bet or an explicit new hypothesis; `lab` enforces the shared budget.
- `strict_visible_segment_survival` is a legacy metric that rewards silence. Acceptance uses version-2 covered identity metrics plus recall, precision and official TrackEval, as defined in the goal. Old numbers need replay before comparison.
- `dancetrack0012` is repeatedly exposed architecture-development data, even when omitted from gradients. Never train on gate `0096`; final `0004/0005/0007/0010` remain for frozen final evaluation.
- Clean detector ancestry is `runs/expanded-0002/last.pt`; the clean training/dev manifest is `data/manifest-train-full-dense-no0096.json`. Verify the hashes in the ledger. Some dense sequences are prefixes despite the manifest name.
- Research source belongs in `experiments/<name>/`; the lab snapshots it and shared Python source under `evidence/attempts/` before execution. Commit source/config/results before removing a recipe. A retained JSON score without generating source is only a historical observation.
- Context cleanup means deleting stale instructions and narrowing unsupported claims, while preserving evidence in Git. A negative local experiment does not permanently ban a whole family. Explain what changed before revisiting it.
- Ordinary `rg` searches exclude archived source via `.ignore`; use `rg --no-ignore` on a specific evidence directory when reproducing an old experiment.
- Use `.venv\Scripts\python.exe` and `lab.ps1` for heavy work. Keep one GPU-heavy process and the activity-aware resource guard; laptop throughput never establishes Nano performance.
- Preserve strict checkpoint loading and nonpersistent positional buffers. Frozen-feature caches require deterministic preprocessing and frozen eval-mode image features; disable/re-key them when those assumptions change.
- The two-second closed roster and immutable canonical state are existing experimental choices, not the product contract. Continuous arrivals, short re-entry, bounded memory and never-recycled public IDs remain required.
