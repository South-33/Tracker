"""Fetch the two initial official DanceTrack archives without third-party packages."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "train1.zip": 3606300312,
    "val.zip": 4209785614,
}


def fetch(name: str) -> dict:
    url = f"https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/{name}"
    target = ROOT / "data" / "archives" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".zip.part")
    expected = SOURCES[name]
    if target.exists() and target.stat().st_size != expected:
        raise RuntimeError(f"Existing archive has unexpected size: {target}")
    if not target.exists():
        for attempt in range(8):
            start = part.stat().st_size if part.exists() else 0
            if start == expected:
                part.rename(target)
                break
            if start > expected:
                raise RuntimeError(f"Partial file exceeds expected size: {part}")
            headers = {"User-Agent": "remember-tracker-research/0.1"}
            if start:
                headers["Range"] = f"bytes={start}-"
            try:
                request = urllib.request.Request(url + f"?download=true&t={int(time.time())}", headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if start and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                        raise RuntimeError("Server did not honor resume range; retaining partial file")
                    last = time.monotonic()
                    done = start
                    with part.open("ab" if start else "wb") as output:
                        while block := response.read(4 * 1024 * 1024):
                            output.write(block)
                            done += len(block)
                            if time.monotonic() - last >= 15:
                                print(f"{name}: {done / expected:.1%} ({done / 1e9:.2f} GB)", flush=True)
                                last = time.monotonic()
                if part.stat().st_size != expected:
                    raise RuntimeError(f"Size mismatch: {part.stat().st_size} != {expected}")
                part.rename(target)
                break
            except Exception as error:
                print(f"Attempt {attempt + 1}: {error}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {"source": url, "bytes": target.stat().st_size, "sha256": digest}
    target.with_suffix(".zip.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {name}: {digest}", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    arguments = parser.parse_args()
    for archive in arguments.archives:
        fetch(archive)
