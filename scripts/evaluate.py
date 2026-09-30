"""Score MOT-format predictions against local DanceTrack ground truth."""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import shutil

import numpy as np
import trackeval

from tracker.splits import (
    CONSUMED_HOLDOUT,
    RESERVED_HOLDOUT,
    assert_research_eval,
    assert_reserved_holdout,
)

ROOT = Path(__file__).resolve().parents[1]


def frame_count(sequence_dir: Path) -> int:
    config = configparser.ConfigParser()
    config.read(sequence_dir / "seqinfo.ini")
    return int(config["Sequence"]["seqLength"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Folder containing <sequence>.txt MOT predictions")
    parser.add_argument("sequences", nargs="+", help="DanceTrack sequence names")
    parser.add_argument(
        "--final-holdout",
        action="store_true",
        help="Score only the sealed RESERVED_HOLDOUT after a candidate is frozen.",
    )
    parser.add_argument(
        "--historical-holdout",
        action="store_true",
        help="Reproduce already-consumed holdout metrics. Never use for tuning.",
    )
    args = parser.parse_args()

    if args.final_holdout and args.historical_holdout:
        raise ValueError("choose only one holdout mode")
    if args.final_holdout:
        assert_reserved_holdout(args.sequences)
    elif args.historical_holdout:
        invalid = [
            sequence
            for sequence in args.sequences
            if sequence not in CONSUMED_HOLDOUT
        ]
        if invalid:
            raise ValueError(
                "historical holdout mode only accepts CONSUMED_HOLDOUT: "
                + ", ".join(invalid)
            )
    else:
        if any(sequence in RESERVED_HOLDOUT for sequence in args.sequences):
            raise ValueError(
                "RESERVED_HOLDOUT is sealed; use --final-holdout only after freeze"
            )
        if any(sequence in CONSUMED_HOLDOUT for sequence in args.sequences):
            raise ValueError(
                "CONSUMED_HOLDOUT cannot guide new research; "
                "use --historical-holdout only to reproduce old metrics"
            )
        assert_research_eval(args.sequences)

    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    eval_gt = run / "_eval_gt"
    if eval_gt.exists():
        shutil.rmtree(eval_gt)

    seq_info = {}
    for sequence in args.sequences:
        source = ROOT / "data" / "dancetrack" / sequence
        gt = source / "gt" / "gt.txt"
        prediction = run / f"{sequence}.txt"
        if not gt.exists():
            raise FileNotFoundError(gt)
        if not prediction.exists():
            raise FileNotFoundError(prediction)
        frames = frame_count(source)
        seq_info[sequence] = frames
        target = eval_gt / sequence / "gt"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt, target / "gt.txt")

    evaluator = trackeval.Evaluator({
        "USE_PARALLEL": False,
        "PLOT_CURVES": False,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "TIME_PROGRESS": False,
        "BREAK_ON_ERROR": True,
    })
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(eval_gt),
        "TRACKERS_FOLDER": str(run.parent),
        "TRACKERS_TO_EVAL": [run.name],
        "TRACKER_SUB_FOLDER": "",
        "OUTPUT_FOLDER": str(run / "metrics"),
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": seq_info,
        "DO_PREPROC": False,
        "PRINT_CONFIG": False,
        "BENCHMARK": "DanceTrack",
        "SPLIT_TO_EVAL": "dev",
    })
    result, _ = evaluator.evaluate([
        dataset
    ], [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ])

    combined = result[dataset.get_name()][run.name]["COMBINED_SEQ"]["pedestrian"]
    summary = {
        "sequences": args.sequences,
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"])) * 100,
        "DetA": float(np.mean(combined["HOTA"]["DetA"])) * 100,
        "AssA": float(np.mean(combined["HOTA"]["AssA"])) * 100,
        "IDF1": float(combined["Identity"]["IDF1"]) * 100,
        "IDSW": int(combined["CLEAR"]["IDSW"]),
        "Frag": int(combined["CLEAR"]["Frag"]),
        "Recall": float(combined["CLEAR"]["CLR_Re"]) * 100,
        "Precision": float(combined["CLEAR"]["CLR_Pr"]) * 100,
    }
    (run / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
