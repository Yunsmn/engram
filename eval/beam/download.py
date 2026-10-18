"""Download BEAM (Hugging Face Mohammadta/BEAM, CC BY-SA 4.0) as JSON rows through the datasets-server API, so no
Parquet reader is needed. Default: the "100K" split (the 128K-token chats, 20 conversations) -> data/beam/raw/.
Usage: python -m eval.beam.download [--split 100K] [--out data/beam/raw] [--count 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

API = "https://datasets-server.huggingface.co/rows"
PAGE = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="100K")
    parser.add_argument("--out", default="data/beam/raw")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for offset in range(0, args.count, PAGE):
        params = {"dataset": "Mohammadta/BEAM", "config": "default", "split": args.split,
                  "offset": offset, "length": min(PAGE, args.count - offset)}
        r = httpx.get(API, params=params, timeout=120)
        r.raise_for_status()
        (out / f"rows_{offset}.json").write_text(r.text)
        print(f"rows {offset}-{offset + params['length'] - 1}", flush=True)


if __name__ == "__main__":
    main()
