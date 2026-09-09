"""Small sequence-supervised pilot with fixed identity assignments and resumable checkpoints."""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import load_sequences
from .detector import load_detector
from .geometry import preprocess
from .loss import assign, supervised_loss
from .model import RecurrentTracker, expire_memory, fill_slot


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    sequences = load_sequences(args.manifest, "train")[:2] if args.debug else load_sequences(args.manifest, "train")
    model = RecurrentTracker(load_detector(device=args.device, size=args.size), args.slots).to(args.device).train()
    detector_parameters = [p for p in model.detector.parameters() if p.requires_grad]
    added_parameters = [p for name, p in model.named_parameters() if not name.startswith("detector.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": detector_parameters, "lr": args.lr * 0.1}, {"params": added_parameters, "lr": args.lr}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=args.device.startswith("cuda"))
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    metadata = {**vars(args), "manifest_sha256": manifest_hash, "torch": torch.__version__,
                "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
                "upstream": "55fefaaed7efe2a5f72d0a18fd4e05965e35c292",
                "loss_scope": "person varifocal + L1/GIoU + resident continuity + IoU quality; no FDR local/auxiliary losses in this pilot",
                "teacher": "none; pretrained detector retains its authors' DINOv3 training benefit"}
    iteration = 0
    elapsed_previous = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        for key in ("manifest_sha256", "slots", "size", "seed"):
            if checkpoint["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
        iteration = checkpoint["iteration"]
        elapsed_previous = checkpoint["elapsed_seconds"]
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()

    def save():
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "iteration": iteration, "elapsed_seconds": elapsed_previous + time.monotonic() - start, "metadata": metadata,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []}
        temporary = directory / "last.tmp.pt"
        torch.save(payload, temporary)
        temporary.replace(directory / "last.pt")

    try:
        while iteration < args.steps:
            sequence = sequences[iteration % len(sequences)] if args.debug else random.choice(sequences)
            stride = 1 if args.debug else random.choice([1, 2, 4])
            length = args.unroll
            if len(sequence.frames) <= stride * (length - 1):
                raise ValueError("Not enough available frames for selected unroll")
            offset = (iteration // len(sequences) * length) % (len(sequence.frames) - length + 1) if args.debug else random.randrange(len(sequence.frames) - stride * (length - 1))
            frame_numbers = sequence.frames[offset:offset + stride * length:stride]
            memory = model.empty_memory()
            slot_to_gt = {}
            optimizer.zero_grad(set_to_none=True)
            losses, reports = [], []
            last_number = frame_numbers[0] - stride
            for number in frame_numbers:
                rgb, ids, boxes = sequence.read(number)
                image, transform = preprocess(rgb, args.size, args.device)
                target = torch.as_tensor(transform.normalize_xywh(boxes), device=args.device)
                dt = torch.tensor([(number - last_number) / sequence.fps], device=args.device)
                last_number = number
                memory = expire_memory(memory, 10)
                slot_to_gt = {slot: identity for slot, identity in slot_to_gt.items() if memory.valid[0, slot, 0].item() > 0.5}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                    prediction = model(image, *memory, dt)
                    assignments, births = assign(prediction, ids, target, slot_to_gt, args.slots)
                    loss, parts = supervised_loss(prediction, assignments, target, memory, args.slots)
                    losses.append(loss)
                    reports.append(parts)
                    observed = torch.zeros(args.slots, device=args.device)
                    for query, _ in assignments:
                        if query < args.slots:
                            observed[query] = 1
                    memory = model.update_memory(memory, prediction, observed, dt)
                    for query, gt_index in births:
                        available = [slot for slot in range(args.slots) if slot not in slot_to_gt]
                        if not available:
                            raise RuntimeError("Training clip exceeds configured resident capacity")
                        slot = available[0]
                        slot_to_gt[slot] = int(ids[gt_index])
                        memory = fill_slot(memory, slot, prediction.features[0, query], prediction.boxes[0, query])
            combined = torch.stack(losses).mean()
            if not torch.isfinite(combined):
                raise FloatingPointError(f"Nonfinite training loss at iteration {iteration}")
            scaler.scale(combined).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite gradient at iteration {iteration}")
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            record = {"iteration": iteration, "loss": float(combined.detach()), "gradient_norm": float(gradient_norm),
                "sequence": sequence.name, "first_frame": frame_numbers[0], "last_frame": frame_numbers[-1],
                "elapsed_seconds": elapsed_previous + time.monotonic() - start,
                **{name: float(torch.stack([p[name].detach() for p in reports]).mean()) for name in reports[0]}}
            with (directory / "loss.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            if iteration % 10 == 0 or iteration == 1:
                print(json.dumps(record), flush=True)
            if iteration % args.save_every == 0:
                save()
        save()
    except KeyboardInterrupt:
        # Checkpoints represent completed optimizer steps, with fresh sequence memory on resume.
        optimizer.zero_grad(set_to_none=True)
        save()
        print("Interrupted; checkpoint preserved.", flush=True)
        raise
