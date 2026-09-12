# Current tracker research

<!-- cycle-status: decision_required -->

Updated 2026-09-12. Read the acceptance and research policy in [GOAL.md](GOAL.md). No current tracker has passed the revised acceptance criteria. There is no validated incumbent under version-2 metrics.

## Next decision

Restore a complete runnable reference, score the already-exposed full dev sequences, then choose a substantial model/state/training bet. The present repo has reusable detector/state infrastructure and diagnostic results, but the previous full training/replay entry points were removed. Git commit `92da962` retains the earlier runnable implementation; do not reset the current tree to it. The accumulated later work is preserved at `e6dae0b`.

Reference restoration is a bounded prerequisite, not permission to spend the next night on infrastructure. Once it runs, the next work interval must produce or launch a substantial causal experiment. At most three exploratory tests can precede that commitment. A substantial bet may fail; its useful output is a reproducible measurement that changes the next decision.

The paused joint-bundle idea remains an option, not the default command. Its next proposed question was victim-only entry into anonymous group state. Compare its potential with an actual temporal-perception learner, a faithfully adapted published tracker, or teacher-guided training before choosing. A tiny frozen-feature adaptation does not stand in for every version of those approaches.

## What the evidence supports

Historical artifacts below are preserved under `evidence/legacy/`. Most generating scripts were removed before archival. Treat these as observations with limited reproducibility, not established ceilings or current acceptance scores. Source completeness is recorded in [the index](evidence/legacy/index.json).

| Observation | Evidence and limitation | Useful inference |
|---|---|---|
| Identity errors compound through carried state | A 3,000-step local-recovery system on full `0016/0020` reported 97.46% detection precision but 36.23% recall and 925 switches. [Result](evidence/legacy/local-recovery-dev.json) | Good publication precision does not establish stable tracking. Evaluate the full feedback loop. |
| Conservatism can create impressive but useless scores | A hard-freeze diagnostic reported 98.35% old switch-free survival with 0.46% recall. [Result](evidence/legacy/confidence-weighted-anchor-train-diagnostic.json) | Require sustained coverage and identity together; silence is failure, not survival. |
| Joint state has a conditional signal | GT selected pair membership and the correct two proposals every frame; 93/140 selected continuous eight-frame episodes recovered the outgoing identity permutation. [Result](evidence/legacy/joint-bundle-oracle-train-diagnostic.json) | 66.43% on this subset is worth understanding, but it is not a whole-system jump or general upper bound. |
| Observable grouping did not preserve that signal | The tested top-2 grouping recovered 24/140 targets, suppressing 26.87% of safe rows; initialization still used clean GT boxes. [Result](evidence/legacy/expanding-bundle-train-diagnostic.json) | This particular grouping rule is negative. Membership, self-owned state and missing detections remain unresolved. |
| Some larger local learners failed to generalize | A separate 1.33M decoder trained for 10k steps reported 18.95% hard recovery and 76.54% absent-owner false claims on repeatedly exposed `0012`. [Result](evidence/legacy/tracking-pointer-train-diagnostic.json) | The tested objective/representation is insufficient. This does not rule out jointly trained temporal perception or a faithful published baseline. |

Previous prose asserted a 99.27% candidate-availability audit, but its K-worldline artifact did not survive the September 12 archive. Do not use that number as an established premise without reproducing it. The older 73-74% baseline claim also lacks surviving evidence. Candidate availability is not equivalent to deployable detector recall.

No existing result has been recalculated under version 2. The legacy strict score counts unseen segments as switch-free; version 2 retains it only for compatibility and adds coverage-aware success. Thresholded matching also changed, so even historical FP/recall figures require replay for direct comparison. The reported numbers above are not new measurements from the context reset.

## Assets and data exposure

Machine-readable hashes live once in `research-cycle.json.inputs`. The verified local detector is `runs/expanded-0002/last.pt`; load through `ancestry.py`. Full checkpoint load of unrelated obsolete heads is not needed to extract the clean detector.

| Local sequence group | Availability and role |
|---|---|
| `0001/0002/0006/0008/0012/0015` | Full training sequences; `0012` repeatedly inspected for architecture selection |
| `0082/0083` | 120/603 frames each; training prefixes |
| `0016/0020` | Full dev: 2,163 and 583 frames; repeatedly exposed |
| `0096` | Architecture gate; never training data, historically inspected in earlier diagnostics |
| `0004/0005/0007/0010` | Reserved final holdouts according to the prior record; no final evaluation in this reset |

Use actual frame lists and timestamps. Do not call the current set a full DanceTrack benchmark. New-arrival and independently reviewed exit/re-entry evaluation still need implementation and suitable event coverage.

## Working code and known gaps

- `ancestry.py`, `detector.py`, `features.py`, `detections.py`, `proposals.py`: reusable clean RT-DETR perception and caches.
- `enrollment.py`, `local_continuity.py`, `state_split.py`: experimental local/identity state. Enrollment still closes births after two seconds, so this is not a general-arrival tracker.
- `tracking_metrics.py`, `episodes.py`, `scripts/evaluate.py`: coverage diagnostics, episode scoring and official metrics. `lab score` joins saved MOT predictions with the first and third. Visual review and full re-entry integration remain outstanding.
- `lab.py`, `research_cycle.py`, `experiment_archive.py`: resource guard, shared exploration budget and automatic source/result preservation. `doctor` checks declared mechanics; it cannot prove scientific validity, no GT leakage, or correct experimental interpretation.

Keep the active source small. Historical code belongs in Git/evidence, fetched only when reproducing a relevant observation. Do not erase the last runnable reference when replacing a mechanism. Follow [README.md](README.md) for commands.
