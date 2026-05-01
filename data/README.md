# Sample Data

Holds input video for AegisVision benchmarks. Videos are gitignored; only the URL and hash live in source control.

## Pinned source

| Field | Value |
| --- | --- |
| URL | <https://www.youtube.com/watch?v=wWLAc6mdJrs&list=PLJKyZ_NuOhJQzif2-6-Kq9OiOj_UjJWvi&index=9> |
| SHA-256 | `9493023bc5db4dcfe7f77358e80afe4285f7e66ead75c61c2ee5a8dcd41ba8d8` |
| Resolution | 1920×1080 |
| Frame rate | 30.000 fps |
| Duration | 158.1 s (~2.6 min, 4742 frames) |
| File size | 67.35 MB |
| Local path | `data/samples/traffic_sample.mp4` |

The clip was selected for: fixed-camera angle suitable for line-crossing counting, mixed vehicle classes, daytime visibility, and 1080p/30fps as a clean benchmarking baseline.

## How to fetch

```sh
make fetch
# or:  uv run python scripts/fetch_sample.py
```

The script downloads to `data/samples/traffic_sample.mp4` and verifies the SHA-256 against the pinned value above. If the hash mismatches, the script exits non-zero. Reproducibility depends on the URL staying live — if it goes dead, replace it and re-pin both the URL above and `SAMPLE_URL` / `EXPECTED_SHA256` in `scripts/fetch_sample.py`.
