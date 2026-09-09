"""Pin upstream code, apply the small feature-output patch, and fetch detector weights."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--evaluation", action="store_true", help="Also fetch pinned TrackEval")
    parser.add_argument("--baseline", action="store_true", help="Fetch only the official ByteTrack association files")
    args = parser.parse_args()
    if args.baseline:
        baseline = ROOT / "third_party" / "ByteTrack-source"
        revision = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"
        files = ["LICENSE", "yolox/tracker/byte_tracker.py", "yolox/tracker/basetrack.py", "yolox/tracker/kalman_filter.py", "yolox/tracker/matching.py"]
        for relative in files:
            path = baseline / relative
            original_path = baseline / "original" / relative
            if not original_path.exists():
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/ifzhang/ByteTrack/{revision}/{relative}", timeout=60) as response:
                    content = response.read()
                original_path.parent.mkdir(parents=True, exist_ok=True)
                original_path.write_bytes(content)
            original = original_path.read_text(encoding="utf-8")
            patched = re.sub(r"\bnp\.(float|int)\b", r"\1", original) if relative.endswith(".py") else original
            if path.exists() and path.read_text(encoding="utf-8") not in (original, patched):
                raise RuntimeError(f"Refusing to overwrite baseline edits: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(patched, encoding="utf-8", newline="\n")
        for relative in ("yolox/__init__.py", "yolox/tracker/__init__.py"):
            (baseline / relative).touch()
        (baseline / "source.json").write_text(json.dumps({"repository": "https://github.com/ifzhang/ByteTrack", "revision": revision,
            "files": files, "adaptation": "NumPy removed float/int aliases replaced; package initializers omit unrelated detector imports."}, indent=2) + "\n")
    if args.evaluation:
        evaluation = ROOT / "third_party" / "TrackEval"
        evaluation_revision = "12c8791b303e0a0b50f753af204249e622d0281a"
        if not evaluation.exists():
            subprocess.run(["git", "clone", "https://github.com/JonathonLuiten/TrackEval.git", str(evaluation)], check=True)
            subprocess.run(["git", "-C", str(evaluation), "checkout", evaluation_revision], check=True)
        if subprocess.check_output(["git", "-C", str(evaluation), "rev-parse", "HEAD"], text=True).strip() != evaluation_revision:
            raise RuntimeError("TrackEval revision mismatch")
        for relative in ("trackeval/datasets/mot_challenge_2d_box.py", "trackeval/metrics/hota.py", "trackeval/metrics/identity.py"):
            path = evaluation / relative
            original = subprocess.check_output(["git", "-C", str(evaluation), "show", f"{evaluation_revision}:{relative}"], text=True)
            patched = re.sub(r"\bnp\.(float|int)\b", r"\1", original)
            if path.read_text(encoding="utf-8") not in (original, patched):
                raise RuntimeError(f"Refusing to overwrite independent TrackEval edits: {relative}")
            path.write_text(patched, encoding="utf-8", newline="\n")
    source = ROOT / "third_party" / "RT-DETRv4"
    if not source.exists():
        subprocess.run(["git", "clone", "https://github.com/RT-DETRs/RT-DETRv4.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Expected upstream {REVISION}, found {actual}")
    decoder = source / "engine" / "rtv4" / "dfine_decoder.py"
    original = subprocess.check_output(["git", "-C", str(source), "show", f"{REVISION}:engine/rtv4/dfine_decoder.py"], text=True)
    patched = original.replace("                dn_meta=None):", "                dn_meta=None, return_features=False):", 1)
    old_return = "        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \\\n               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores"
    new_return = "        result = (torch.stack(dec_out_bboxes), torch.stack(dec_out_logits),\n                  torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores)\n        return (*result, output) if return_features else result"
    if old_return not in patched:
        raise RuntimeError("Pinned decoder no longer matches the audited feature-output patch")
    patched = patched.replace(old_return, new_return, 1)
    existing = decoder.read_text(encoding="utf-8")
    if existing not in (original, patched):
        raise RuntimeError("Refusing to overwrite independent upstream decoder edits")
    decoder.write_text(patched, encoding="utf-8", newline="\n")
    misc = source / "engine" / "misc" / "__init__.py"
    original_misc = subprocess.check_output(["git", "-C", str(source), "show", f"{REVISION}:engine/misc/__init__.py"], text=True)
    patched_misc = original_misc.replace("from .profiler_utils import stats", "def stats(*args, **kwargs):\n    from .profiler_utils import stats as profile_stats\n    return profile_stats(*args, **kwargs)")
    if misc.read_text(encoding="utf-8") not in (original_misc, patched_misc):
        raise RuntimeError("Refusing to overwrite independent upstream misc edits")
    misc.write_text(patched_misc, encoding="utf-8", newline="\n")
    print(f"Upstream {actual}, optional final decoder feature output enabled.")
    if args.skip_weights:
        return
    import gdown
    weights = ROOT / "weights" / "rtv4_s.pth"
    weights.parent.mkdir(exist_ok=True)
    if not weights.exists():
        result = gdown.download(id="1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z", output=str(weights), resume=True)
        if not result:
            raise RuntimeError("Official RT-DETRv4-S checkpoint download failed")
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    weights.with_suffix(".json").write_text(json.dumps({
        "url": "https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z",
        "sha256": digest, "bytes": weights.stat().st_size, "upstream_revision": REVISION,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Detector checkpoint SHA256 {digest}")


if __name__ == "__main__":
    main()
