"""Package the current causal person tracker into one deployable artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracker.system import TrackerConfig, build_tracker_bundle

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-head", default="runs/identity-head-v2/head.pt")
    parser.add_argument("--owner-head", default="runs/owner-head-v1/head.pt")
    parser.add_argument(
        "--output",
        default="runs/person-tracker.pt",
        help="Packaged tracker artifact.",
    )
    args = parser.parse_args()

    bundle = build_tracker_bundle(
        ROOT / args.identity_head,
        ROOT / args.owner_head,
        ROOT / args.output,
    )
    path = ROOT / args.output
    print(
        json.dumps(
            {
                "output": args.output,
                "bytes": path.stat().st_size,
                "format": bundle["format"],
                "config": TrackerConfig().metadata(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
