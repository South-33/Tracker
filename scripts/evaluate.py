"""Score MOT-format predictions against local DanceTrack ground truth."""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from tracker.data import load_mot
from tracker.metrics import score_mot_rows

ROOT = Path(__file__).resolve().parents[1]


def frame_count(sequence_dir: Path) -> int:
    config = configparser.ConfigParser()
    config.read(sequence_dir / "seqinfo.ini")
    return int(config["Sequence"]["seqLength"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Folder containing <sequence>.txt MOT predictions")
    parser.add_argument("sequences", nargs="+", help="DanceTrack sequence names")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    eval_gt = run / "_eval_gt"
    if eval_gt.exists():
        shutil.rmtree(eval_gt)

    coverage = {}
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
        truth_rows = load_mot(gt)
        prediction_rows = load_mot(prediction)
        coverage[sequence] = score_mot_rows(truth_rows, prediction_rows, frames)
        target = eval_gt / sequence / "gt"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt, target / "gt.txt")

    trackeval_root = ROOT / "third_party" / "TrackEval"
    if not trackeval_root.exists():
        raise FileNotFoundError("third_party/TrackEval is missing")
    sys.path.insert(0, str(trackeval_root))
    import trackeval

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
        "coverage": coverage,
    }
    (run / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
