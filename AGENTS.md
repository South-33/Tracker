This is the project's AGENTS.md

- Read RESULTS.md first for evidence, checkpoint and next action; GOAL.md owns the full target and the user's substantial-experiment/reassessment instructions.
- Bootstrap owns upstream patches. Preserve strict checkpoint loading and the loader's nonpersistent position-embedding buffers.
- Frozen-feature caching assumes deterministic preprocessing and frozen eval-mode image features; disable or redesign it when adding augmentation or unfreezing them.
- Checkpoint runs/unroll12/last.pt is a 400-step pilot. Development results are mixed; four holdout sequences are untouched. Nano/SmartCampus has not been changed or benchmarked.
