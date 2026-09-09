"""Resumable bounded parallel ranges for large public files with verified range support."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request


def fetch(url, destination, workers=8, block_size=16 * 1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise RuntimeError("Source must support exact HTTP ranges")
        total = int(content_range.split("/")[-1])
    if destination.exists() and destination.stat().st_size == total:
        print(f"Already downloaded: {destination}", flush=True)
        return
    parts = destination.with_name(destination.name + ".parts")
    parts.mkdir(exist_ok=True)

    def download(index):
        start, end = index * block_size, min((index + 1) * block_size, total) - 1
        part = parts / f"{index:05d}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part.stat().st_size
        for attempt in range(5):
            try:
                separator = "&" if "?" in url else "?"
                request = urllib.request.Request(url + f"{separator}range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Source did not honor requested range")
                    with part.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete range")
                return part.stat().st_size
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    count = (total + block_size - 1) // block_size
    done, last = 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(download, index) for index in range(count)]):
            done += future.result()
            if time.monotonic() - last > 15:
                print(f"{destination.name}: {done / total:.1%}", flush=True)
                last = time.monotonic()
    temporary = destination.with_name(destination.name + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:05d}").open("rb") as part:
                while block := part.read(4 * 1024 * 1024):
                    digest.update(block)
                    output.write(block)
    temporary.replace(destination)
    # Only remove exact part files created for this destination after successful assembly.
    for index in range(count):
        (parts / f"{index:05d}").unlink()
    parts.rmdir()
    print(f"Downloaded {destination}, {total} bytes, SHA256 {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fetch(args.url, args.destination, args.workers)
