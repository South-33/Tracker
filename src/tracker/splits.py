"""Single source of truth for DanceTrack data roles.

TRAIN may update learned weights.
CALIBRATION may tune thresholds and choose between hypotheses.
DEV is the normal online tracking hill-climb set.
CONSUMED_HOLDOUT is historical final evidence and must not guide new changes.
RESERVED_HOLDOUT is sealed for the next frozen candidate.
"""
from __future__ import annotations

TRAIN = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0006",
    "dancetrack0008",
    "dancetrack0015",
)

CALIBRATION = ("dancetrack0012",)

DEV = (
    "dancetrack0016",
    "dancetrack0020",
)

CONSUMED_HOLDOUT_LEGACY = (
    "dancetrack0004",
    "dancetrack0005",
    "dancetrack0007",
    "dancetrack0010",
    "dancetrack0096",
)

CONSUMED_HOLDOUT_2026_10_01 = (
    "dancetrack0014",
    "dancetrack0019",
    "dancetrack0035",
    "dancetrack0047",
    "dancetrack0063",
    "dancetrack0073",
    "dancetrack0077",
    "dancetrack0081",
    "dancetrack0090",
    "dancetrack0097",
)

CONSUMED_HOLDOUT = CONSUMED_HOLDOUT_LEGACY + CONSUMED_HOLDOUT_2026_10_01

# Empty after the 2026-10-01 frozen-candidate evaluation. Define a new sealed
# set before the next final comparison, before running inference or metrics.
RESERVED_HOLDOUT = ()

RESEARCH_EVAL = CALIBRATION + DEV
ALL_KNOWN = (
    TRAIN
    + CALIBRATION
    + DEV
    + CONSUMED_HOLDOUT
    + RESERVED_HOLDOUT
)


def role_of(sequence: str) -> str:
    if sequence in TRAIN:
        return "train"
    if sequence in CALIBRATION:
        return "calibration"
    if sequence in DEV:
        return "dev"
    if sequence in CONSUMED_HOLDOUT:
        return "consumed_holdout"
    if sequence in RESERVED_HOLDOUT:
        return "reserved_holdout"
    return "unknown"


def assert_train_only(sequences) -> None:
    invalid = [sequence for sequence in sequences if sequence not in TRAIN]
    if invalid:
        raise ValueError(
            "training is restricted to TRAIN sequences; refused: "
            + ", ".join(invalid)
        )


def assert_research_eval(sequences) -> None:
    invalid = [sequence for sequence in sequences if sequence not in RESEARCH_EVAL]
    if invalid:
        raise ValueError(
            "research evaluation is restricted to CALIBRATION/DEV; refused: "
            + ", ".join(invalid)
        )


def assert_reserved_holdout(sequences) -> None:
    invalid = [sequence for sequence in sequences if sequence not in RESERVED_HOLDOUT]
    if invalid:
        raise ValueError(
            "final holdout mode is restricted to RESERVED_HOLDOUT; refused: "
            + ", ".join(invalid)
        )


def assert_not_holdout(sequences) -> None:
    blocked = [
        sequence
        for sequence in sequences
        if sequence in CONSUMED_HOLDOUT or sequence in RESERVED_HOLDOUT
    ]
    if blocked:
        raise ValueError(
            "holdout sequences are sealed from normal research; refused: "
            + ", ".join(blocked)
        )
