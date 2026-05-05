"""Replay a pipeline events.jsonl into a count time-series.

Demonstrates that the JSON event log captures everything needed to
reconstruct counter state without re-running inference. Output is a
markdown table of cumulative per-class in/out counts bucketed by time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path, help="path to events.jsonl")
    p.add_argument("--bucket", type=int, default=10, help="bucket width in seconds (default: 10)")
    args = p.parse_args(argv)

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2

    events = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    if not events:
        print("log is empty", file=sys.stderr)
        return 1

    starts = [e for e in events if e.get("event") == "start"]
    crossings = [e for e in events if e.get("event") == "line_cross"]
    summaries = [e for e in events if e.get("event") == "summary"]
    ends = [e for e in events if e.get("event") == "end"]

    print(f"events: {len(events)} total | {len(starts)} start, "
          f"{len(crossings)} line_cross, {len(summaries)} summary, {len(ends)} end")
    if not crossings:
        print("no line crossings to replay")
        return 0

    t0 = _parse_ts((starts[-1] if starts else crossings[0])["ts"])

    classes: set[str] = set()
    bucket_in: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bucket_out: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for c in crossings:
        cls = c["class"]
        classes.add(cls)
        secs = (_parse_ts(c["ts"]) - t0).total_seconds()
        b = int(max(0, secs) // args.bucket)
        if c["direction"] == "in":
            bucket_in[b][cls] += 1
        else:
            bucket_out[b][cls] += 1

    cls_sorted = sorted(classes)
    headers = ["t (s)"] + [f"{c}↓in" for c in cls_sorted] + [f"{c}↑out" for c in cls_sorted] + ["row total"]
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    cum_in: dict[str, int] = defaultdict(int)
    cum_out: dict[str, int] = defaultdict(int)
    buckets = sorted(set(bucket_in) | set(bucket_out))
    for b in buckets:
        row_total = 0
        for cls in cls_sorted:
            cum_in[cls] += bucket_in[b].get(cls, 0)
            cum_out[cls] += bucket_out[b].get(cls, 0)
            row_total += bucket_in[b].get(cls, 0) + bucket_out[b].get(cls, 0)
        cells = [f"{b * args.bucket}–{(b + 1) * args.bucket}"]
        cells += [str(cum_in[c]) for c in cls_sorted]
        cells += [str(cum_out[c]) for c in cls_sorted]
        cells += [str(row_total)]
        print("| " + " | ".join(cells) + " |")

    print()
    print(f"final cumulative: in={sum(cum_in.values())}, out={sum(cum_out.values())}, "
          f"total={sum(cum_in.values()) + sum(cum_out.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
