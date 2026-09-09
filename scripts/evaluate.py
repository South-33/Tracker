"""Official TrackEval metrics for an explicitly labelled, contiguous DanceTrack subset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest-120.json")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    directory = Path(args.run).resolve()
    metadata = json.loads((directory / "run.json").read_text())
    record = next(record for record in json.loads(Path(args.manifest).read_text())["records"] if record["sequence"] == metadata["sequence"])
    frames = record["available_frames"]
    if frames != list(range(1, len(frames) + 1)):
        raise ValueError("This evaluator accepts contiguous prefixes only; sparse labels need an explicit protocol")
    if metadata["frames"] != len(frames):
        raise ValueError("Prediction run does not cover the full frozen subset")
    sequence = record["sequence"]
    gt_path = directory / "eval-inputs" / sequence / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.loadtxt(ROOT / "data/dancetrack" / sequence / "gt/gt.txt", delimiter=",", ndmin=2)
    np.savetxt(gt_path, rows[rows[:, 0] <= len(frames)], delimiter=",", fmt="%.6f")
    sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
    import trackeval
    evaluator = trackeval.Evaluator({"USE_PARALLEL": False, "PLOT_CURVES": False, "PRINT_CONFIG": False,
                                     "PRINT_RESULTS": True, "TIME_PROGRESS": False, "BREAK_ON_ERROR": True})
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(directory / "eval-inputs"), "TRACKERS_FOLDER": str(directory.parent),
        "TRACKERS_TO_EVAL": [directory.name], "TRACKER_SUB_FOLDER": "", "OUTPUT_FOLDER": str(directory / "metrics"),
        "SKIP_SPLIT_FOL": True, "SEQ_INFO": {sequence: len(frames)}, "DO_PREPROC": False,
        "PRINT_CONFIG": False, "BENCHMARK": "DanceTrack", "SPLIT_TO_EVAL": record["split"],
    })
    result, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    combined = result[dataset.get_name()][directory.name]["COMBINED_SEQ"]["pedestrian"]
    summary = {"scope": "DanceTrack contiguous sequence prefix subset; not official full-benchmark scores",
               "split": record["split"], "sequence": sequence, "frames": len(frames), "source_frames": record["source_frames"],
               "HOTA": float(np.mean(combined["HOTA"]["HOTA"])) * 100,
               "DetA": float(np.mean(combined["HOTA"]["DetA"])) * 100,
               "AssA": float(np.mean(combined["HOTA"]["AssA"])) * 100,
               "IDF1": float(combined["Identity"]["IDF1"]) * 100,
               "IDSW": int(combined["CLEAR"]["IDSW"]), "Frag": int(combined["CLEAR"]["Frag"]),
               "Recall": float(combined["CLEAR"]["CLR_Re"]) * 100, "Precision": float(combined["CLEAR"]["CLR_Pr"]) * 100}
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
