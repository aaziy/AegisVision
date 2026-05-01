"""Fetch the pinned traffic sample video and verify its SHA-256.

Run via `make fetch` or directly: `uv run python scripts/fetch_sample.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "samples"
OUTPUT_NAME = "traffic_sample.mp4"

# Pin the source video here. Bumping either is a deliberate change.
# After the first successful download, update EXPECTED_SHA256 to the printed value.
SAMPLE_URL = "https://www.youtube.com/watch?v=wWLAc6mdJrs&list=PLJKyZ_NuOhJQzif2-6-Kq9OiOj_UjJWvi&index=9"
EXPECTED_SHA256 = "9493023bc5db4dcfe7f77358e80afe4285f7e66ead75c61c2ee5a8dcd41ba8d8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        # Cap at 1080p for stable benchmarking input.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(dest),
        "noplaylist": True,
        "quiet": False,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the pinned traffic sample.")
    parser.add_argument("--url", default=SAMPLE_URL, help="YouTube URL to download")
    parser.add_argument(
        "--expected-sha256", default=EXPECTED_SHA256, help="Expected SHA-256 of the file"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    dest = OUTPUT_DIR / OUTPUT_NAME

    if dest.exists() and not args.force:
        print(f"Sample already on disk at {dest.relative_to(REPO_ROOT)}.")
    else:
        if not args.url:
            print(
                "No URL configured. Pass --url <youtube-url>, or pin SAMPLE_URL in this script.",
                file=sys.stderr,
            )
            return 2
        download(args.url, dest)

    actual = sha256(dest)

    if not args.expected_sha256:
        print(f"Computed sha256: {actual}")
        print("No expected hash configured. Pin this value in scripts/fetch_sample.py")
        print("and data/README.md to enable integrity verification on subsequent runs.")
        return 0

    if actual != args.expected_sha256:
        print(
            f"Hash mismatch.\n  expected {args.expected_sha256}\n  actual   {actual}",
            file=sys.stderr,
        )
        return 1

    print(f"Verified {dest.relative_to(REPO_ROOT)} (sha256 {actual[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
