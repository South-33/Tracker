"""Prepare frozen sequence splits, optionally using HTTP ranges for an early pilot."""
import argparse
from collections import OrderedDict
import configparser
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from fetch_data import DEFAULT_ARCHIVES, ROOT, SOURCES


class RemoteArchive(io.RawIOBase):
    """Bounded read cache lets zipfile read selected members before a full download finishes."""
    def __init__(self, name, block_size=4 * 1024 * 1024):
        self.name = name
        self.length = SOURCES[name]
        self.url = f"https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/{name}"
        self.position = 0
        self.block_size = block_size
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, size=-1):
        end = self.length if size < 0 else min(self.position + size, self.length)
        output = bytearray()
        while self.position < end:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                stop = min(start + self.block_size, self.length) - 1
                for attempt in range(4):
                    try:
                        request = urllib.request.Request(
                            self.url + f"?download=true&block={index}&t={int(time.time())}",
                            headers={"Range": f"bytes={start}-{stop}"},
                        )
                        with urllib.request.urlopen(request, timeout=90) as response:
                            expected = f"bytes {start}-{stop}/{self.length}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise RuntimeError("Source did not honor exact byte range")
                            block = response.read()
                        if len(block) != stop - start + 1:
                            raise RuntimeError("Incomplete archive range")
                        self.cache[index] = block
                        while len(self.cache) > 4:
                            self.cache.popitem(last=False)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
            self.cache.move_to_end(index)
            block = self.cache[index]
            offset = self.position % self.block_size
            count = min(len(block) - offset, end - self.position)
            output.extend(block[offset:offset + count])
            self.position += count
        return bytes(output)


def prepare(archive_name, frame_limit, remote, requested_sequences=None, forced_split=None):
    path = ROOT / "data" / "archives" / archive_name
    stream = path if path.exists() else RemoteArchive(archive_name) if remote else None
    if stream is None:
        raise FileNotFoundError(f"Archive not downloaded: {path}; use --remote for selected range reads")
    with zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        sequences = sorted({part for name in names for part in name.split("/") if part.startswith("dancetrack") and part[10:].isdigit()})
        if requested_sequences:
            missing = sorted(set(requested_sequences) - set(sequences))
            if missing:
                raise RuntimeError(f"Requested sequences not present in {archive_name}: {missing}")
            chosen = [sequence for sequence in requested_sequences if sequence in sequences]
        else:
            if archive_name == "train2.zip":
                raise ValueError("train2.zip requires explicit --sequence selections and --split train")
            chosen = sequences[:8] if archive_name == "train1.zip" else sequences[:4]
            if len(chosen) < (8 if archive_name == "train1.zip" else 4):
                raise RuntimeError(f"Unexpected sequence layout: {sequences}")
        records = []
        for index, sequence in enumerate(chosen):
            split = forced_split or ("train" if archive_name == "train1.zip" and index < 6 else "dev" if archive_name == "train1.zip" else "holdout")
            # All choices depend on source names, before any detector predictions are inspected.
            members = []
            for entry in archive.infolist():
                parts = Path(entry.filename).parts
                if sequence not in parts or entry.is_dir():
                    continue
                relative = Path(*parts[parts.index(sequence):])
                if "img1" in relative.parts and frame_limit and int(relative.stem) > frame_limit:
                    continue
                members.append((entry, relative))
            for entry, relative in sorted(members, key=lambda item: item[0].header_offset):
                target = (ROOT / "data" / "dancetrack" / relative).resolve()
                if not target.is_relative_to((ROOT / "data" / "dancetrack").resolve()):
                    raise RuntimeError("Unsafe archive member")
                if not target.exists() or target.stat().st_size != entry.file_size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(entry))  # zipfile validates each member CRC.
            directory = ROOT / "data" / "dancetrack" / sequence
            config = configparser.ConfigParser()
            config.read(directory / "seqinfo.ini")
            info = config["Sequence"]
            gt = directory / "gt" / "gt.txt"
            frames = sorted(int(p.stem) for p in (directory / "img1").glob("*.jpg"))
            if frame_limit:
                frames = [frame for frame in frames if frame <= frame_limit]
            expected = list(range(1, min(frame_limit or int(info["seqLength"]), int(info["seqLength"])) + 1))
            if frames != expected:
                raise RuntimeError(f"Incomplete sequence extraction: {sequence}")
            records.append({"sequence": sequence, "split": split, "archive": archive_name,
                "fps": int(info["frameRate"]), "width": int(info["imWidth"]), "height": int(info["imHeight"]),
                "source_frames": int(info["seqLength"]), "available_frames": frames,
                "gt_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(), "annotation_cadence": "every frame",
                "box_format": "one-based frame; pixel left,top,width,height", "class": "person",
                "visibility": "unknown; trailing constants are not measured visibility"})
            print(f"{split}: {sequence}, {len(frames)}/{info['seqLength']} frames available", flush=True)
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Leading frames per sequence; zero selects full sequences")
    parser.add_argument("--archive", choices=list(SOURCES), action="append")
    parser.add_argument("--output", help="Manifest path; useful when preparing only one archive")
    parser.add_argument("--sequence", action="append", help="Explicit sequence(s) to extract from a single archive")
    parser.add_argument("--split", choices=["train", "dev", "holdout"], help="Force split for explicit sequences")
    parser.add_argument("--base-manifest", help="Copy records from an existing manifest before adding prepared records")
    args = parser.parse_args()
    archives = args.archive or DEFAULT_ARCHIVES
    if args.sequence and len(archives) != 1:
        raise ValueError("--sequence requires exactly one --archive")
    if args.split and not args.sequence:
        raise ValueError("--split is only valid with --sequence")
    records = []
    if args.base_manifest:
        base = Path(args.base_manifest)
        if not base.is_absolute():
            base = ROOT / base
        records.extend(json.loads(base.read_text())["records"])
    for name in archives:
        records.extend(prepare(name, args.frames, args.remote, args.sequence, args.split))
    names = [record["sequence"] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("Combined manifest contains duplicate sequence names")
    manifest = {"source": "https://huggingface.co/datasets/noahcao/dancetrack", "split_rule": "lexicographic whole-sequence split, frozen before inference",
                "frame_limit": args.frames, "records": records}
    path = Path(args.output) if args.output else ROOT / "data" / f"manifest-{args.frames or 'full'}.json"
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError(f"Refusing to silently change existing manifest: {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest {path}; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
