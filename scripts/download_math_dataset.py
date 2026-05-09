#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import argparse
import pathlib
import urllib.request

BASE = "https://huggingface.co/datasets/dianetc/OBLIQ-Bench/resolve/main"
FILES = {
    "corpus.jsonl": "analogues/math/corpus/corpus.jsonl",
    "queries.jsonl": "analogues/math/queries%2Bqrels/queries.jsonl",
    "qrels.tsv": "analogues/math/queries%2Bqrels/qrels.tsv",
    "qrels_pool.tsv": "analogues/math/queries%2Bqrels/qrels_pool.tsv",
    "per_query_excluded_ids.json": "analogues/math/queries%2Bqrels/per_query_excluded_ids.json",
}


def download(url: str, target: pathlib.Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"exists {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"download {url} -> {target}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".benchmarks/obliq/data")
    args = parser.parse_args()
    out = pathlib.Path(args.data_dir) / "math"
    for name, remote in FILES.items():
        download(f"{BASE}/{remote}", out / name)


if __name__ == "__main__":
    main()
