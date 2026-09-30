"""Train the temporal YOLO track-slot head on contiguous human video clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tracker.data import MOTClipDataset, collate_clips
from tracker.model import TemporalYOLO, temporal_slot_loss
from tracker.splits import TRAIN, assert_train_only

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="DanceTrack training sequence. Repeat for multiple.",
    )
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overfit-one",
        action="store_true",
        help="Repeat one clip. This is the required Stage-0 smoke test.",
    )
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--output", default="runs/temporal-smoke")
    args = parser.parse_args()

    torch.manual_seed(0)
    sequences = tuple(args.sequences or [TRAIN[0]])
    assert_train_only(sequences)

    dataset = MOTClipDataset(
        ROOT / "data" / "dancetrack",
        sequences,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        frame_step=args.frame_step,
    )
    if not 0 <= args.clip_index < len(dataset):
        raise ValueError(
            f"--clip-index must be between 0 and {len(dataset) - 1}"
        )

    if args.overfit_one:
        fixed_batch = collate_clips(
            [dataset[args.clip_index] for _ in range(args.batch_size)]
        )
        loader = None
    else:
        fixed_batch = None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_clips,
        )
        iterator = iter(loader)

    device = torch.device(args.device)
    model = TemporalYOLO(
        ROOT / "weights" / "yolo26n.pt",
        slots=args.slots,
        memory_dim=args.memory_dim,
        freeze_visual=True,
    ).to(device)
    optimizer = AdamW(
        model.temporal.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print(
        json.dumps(
            {
                "mode": "overfit-one" if args.overfit_one else "train",
                "sequences": sequences,
                "clips": len(dataset),
                "clip_length": args.clip_length,
                "frame_step": args.frame_step,
                "slots": args.slots,
                "memory_dim": args.memory_dim,
            }
        )
    )

    history = []
    model.train()
    for step in range(1, args.steps + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

        frames = batch["frames"].to(device, non_blocking=True)
        outputs = model(frames)
        loss, report = temporal_slot_loss(
            outputs,
            batch["boxes"],
            batch["identities"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.temporal.parameters(), 5.0)
        optimizer.step()

        history.append(report)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step} "
                f"loss={report['loss']:.4f} "
                f"presence={report['presence_loss']:.4f} "
                f"box={report['box_loss']:.4f} "
                f"targets={int(report['visible_targets'])}"
            )

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "temporal-yolo-slots-v0",
            "yolo": "weights/yolo26n.pt",
            "temporal": model.temporal.state_dict(),
            "slots": args.slots,
            "memory_dim": args.memory_dim,
            "clip_length": args.clip_length,
            "frame_step": args.frame_step,
            "training_sequences": list(sequences),
            "overfit_one": args.overfit_one,
            "clip_index": args.clip_index,
            "steps": args.steps,
            "lr": args.lr,
            "initial_loss": history[0]["loss"],
            "final_loss": history[-1]["loss"],
        },
        output / "model.pt",
    )
    (output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
