"""Print BEAM question ids (default) or conversation ids (--conversations), space-separated, for shell use."""

from __future__ import annotations

import argparse
import json

from eval.data import INDEX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", action="store_true")
    args = parser.parse_args()
    ids = [r["question_id"] for r in json.loads(INDEX.read_text())]
    if args.conversations:
        ids = list(dict.fromkeys(q.split("_")[0] for q in ids))
    print(" ".join(ids))


if __name__ == "__main__":
    main()
