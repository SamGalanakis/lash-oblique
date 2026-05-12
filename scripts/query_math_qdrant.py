#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "qdrant-client[fastembed]>=1.14",
#   "fastembed>=0.6",
#   "requests>=2.32",
# ]
# ///
import argparse
import json
import os
import pathlib
import sys
import time
import uuid

import requests
from qdrant_client import QdrantClient, models

try:
    from fastembed import LateInteractionTextEmbedding
except ImportError as exc:
    raise SystemExit(
        "missing fastembed; run `python3 -m pip install -r requirements.txt`"
    ) from exc

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DENSE_MODEL = "perplexity/pplx-embed-v1-4b"
SPARSE_MODEL = "Qdrant/bm25"
LATE_MODEL = "colbert-ir/colbertv2.0"
POINT_NAMESPACE = uuid.UUID("47c85e95-6a1c-48dd-bfd9-4b7b4a61f487")
DEFAULT_LIMIT = 300
MAX_LIMIT = 300


def point_id(subset: str, doc_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{subset}/{doc_id}"))


def limit_value(payload: dict, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    value = int(payload.get("limit", default))
    return max(1, min(value, maximum))


def candidate_pool_value(payload: dict, default: int = 1000, maximum: int = 2000) -> int:
    value = int(payload.get("candidate_pool", payload.get("prefetch_limit", default)))
    return max(1, min(value, maximum))


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hits(points):
    out = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        out.append(
            {
                "rank": rank,
                "doc_id": payload.get("doc_id") or str(point.id),
                "score": point.score,
                "text": payload.get("text") or "",
                "metadata": payload.get("metadata") or {},
            }
        )
    return out


class Searcher:
    def __init__(self, url: str, collection: str, subset: str, dense_model: str, api_key: str):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.subset = subset
        self.dense_model = dense_model
        self.api_key = api_key
        self._late = None
        self._late_available = None

    def subset_filter(self, *conditions):
        must = [
            models.FieldCondition(
                key="subset", match=models.MatchValue(value=self.subset)
            )
        ]
        must.extend(conditions)
        return models.Filter(must=must)

    def dense_embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.dense_model, "input": texts},
            timeout=120,
        )
        if response.status_code == 429:
            time.sleep(5)
            response = requests.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.dense_model, "input": texts},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item.get("index", 0))
        return [[float(v) for v in item["embedding"]] for item in data]

    @property
    def late(self):
        if self._late is None:
            self._late = LateInteractionTextEmbedding(model_name=LATE_MODEL)
        return self._late

    def late_available(self) -> bool:
        if self._late_available is not None:
            return self._late_available
        info = self.client.get_collection(self.collection)
        config = info.config.params.vectors
        self._late_available = "late" in getattr(config, "keys", lambda: [])()
        return self._late_available

    def bm25(self, query: str, limit: int):
        result = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(text=query, model=SPARSE_MODEL),
            using="bm25",
            query_filter=self.subset_filter(),
            limit=limit,
            with_payload=True,
        )
        return hits(result.points)

    def dense_search(self, query: str, limit: int):
        vector = self.dense_embed([query])[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using="dense",
            query_filter=self.subset_filter(),
            limit=limit,
            with_payload=True,
        )
        return hits(result.points)

    def late_search(self, query: str, prefetch_limit: int, limit: int):
        if not self.late_available():
            raise RuntimeError(
                "collection was built without late vectors; rerun setup with --enable-late"
            )
        dense = self.dense_embed([query])[0]
        late = [[float(v) for v in token] for token in next(self.late.embed([query]))]
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=models.Prefetch(query=dense, using="dense", limit=prefetch_limit),
            query=late,
            using="late",
            query_filter=self.subset_filter(),
            limit=limit,
            with_payload=True,
        )
        return hits(result.points)

    def discover(
        self,
        target_query: str,
        context_pairs: list[dict],
        limit: int,
    ):
        if not context_pairs:
            raise RuntimeError("context_pairs must not be empty")
        positive_texts = [pair["positive_text"] for pair in context_pairs]
        negative_texts = [pair["negative_text"] for pair in context_pairs]
        embeddings = self.dense_embed([target_query, *positive_texts, *negative_texts])
        target = embeddings[0]
        positives = embeddings[1 : 1 + len(context_pairs)]
        negatives = embeddings[1 + len(context_pairs) :]
        result = self.client.query_points(
            collection_name=self.collection,
            query=models.DiscoverQuery(
                discover=models.DiscoverInput(
                    target=target,
                    context=[
                        models.ContextPair(
                            positive=positive,
                            negative=negative,
                        )
                        for positive, negative in zip(positives, negatives, strict=True)
                    ],
                )
            ),
            using="dense",
            query_filter=self.subset_filter(),
            limit=limit,
            with_payload=True,
        )
        return hits(result.points)

    def fetch(self, doc_ids: list[str]):
        result = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=self.subset_filter(
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))
            ),
            limit=len(doc_ids),
            with_payload=True,
            with_vectors=False,
        )
        by_id = {}
        for point in result[0]:
            payload = point.payload or {}
            by_id[payload.get("doc_id") or str(point.id)] = payload
        return [
            {
                "doc_id": doc_id,
                "found": doc_id in by_id,
                "text": by_id.get(doc_id, {}).get("text"),
                "metadata": by_id.get(doc_id, {}).get("metadata") or {},
            }
            for doc_id in doc_ids
        ]


def corpus_stats(data_dir: pathlib.Path, searcher: Searcher):
    subset_dir = data_dir / searcher.subset
    counts = {}
    for name in ["corpus.jsonl", "queries.jsonl"]:
        counts[name] = sum(1 for _ in read_jsonl(subset_dir / name))
    qrels = subset_dir / "qrels.tsv"
    counts["qrels.tsv"] = sum(1 for line in qrels.open() if line.strip()) - 1
    info = searcher.client.get_collection(searcher.collection)
    return {
        "subset": searcher.subset,
        "files": counts,
        "collection": searcher.collection,
        "points_count": info.points_count,
        "status": str(info.status),
        "late_available": searcher.late_available(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--data-dir", default=".benchmarks/obliq/data")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="obliq_analogues")
    parser.add_argument("--subset", default="math")
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--env-file", default="/home/sam/code/lash/.env")
    args = parser.parse_args()
    load_env_file(pathlib.Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is required; set it or provide --env-file /home/sam/code/lash/.env"
        )
    payload = json.load(sys.stdin)
    searcher = Searcher(args.qdrant_url, args.collection, args.subset, args.dense_model, api_key)
    data_dir = pathlib.Path(args.data_dir)

    try:
        if args.op == "corpus_stats":
            output = corpus_stats(data_dir, searcher)
        elif args.op == "fetch_docs":
            output = {"docs": searcher.fetch(payload["doc_ids"])}
        elif args.op == "bm25_search":
            output = {"matches": searcher.bm25(payload["query"], limit_value(payload))}
        elif args.op == "dense_search":
            output = {"matches": searcher.dense_search(payload["query"], limit_value(payload))}
        elif args.op == "late_search":
            output = {
                "matches": searcher.late_search(
                    payload["query"],
                    candidate_pool_value(payload),
                    limit_value(payload),
                )
            }
        elif args.op == "discover_docs":
            output = {
                "matches": searcher.discover(
                    payload["target_query"],
                    payload["context_pairs"],
                    limit_value(payload),
                )
            }
        elif args.op == "hybrid_search":
            limit = limit_value(payload)
            prefetch_limit = candidate_pool_value(payload)
            prefetches = []
            for query in payload["queries"]:
                prefetches.append(
                    models.Prefetch(
                        query=models.Document(text=query, model=SPARSE_MODEL),
                        using="bm25",
                        limit=prefetch_limit,
                    )
                )
                prefetches.append(
                    models.Prefetch(
                        query=searcher.dense_embed([query])[0],
                        using="dense",
                        limit=prefetch_limit,
                    )
                )
                if searcher.late_available():
                    prefetches.append(
                        models.Prefetch(
                            prefetch=models.Prefetch(
                                query=searcher.dense_embed([query])[0],
                                using="dense",
                                limit=prefetch_limit,
                            ),
                            query=[
                                [float(v) for v in token]
                                for token in next(searcher.late.embed([query]))
                            ],
                            using="late",
                            limit=prefetch_limit,
                        )
                    )
            result = searcher.client.query_points(
                collection_name=searcher.collection,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=searcher.subset_filter(),
                limit=limit,
                with_payload=True,
            )
            output = {"matches": hits(result.points)}
        else:
            raise SystemExit(f"unknown op: {args.op}")
    except Exception as exc:
        message = str(exc).lower()
        if ("404" in message and "not found" in message) or (
            "collection" in message and "exist" in message
        ):
            raise SystemExit(
                f"Qdrant collection `{args.collection}` was not found at {args.qdrant_url}. "
                "Run `scripts/setup_math.sh --recreate` or pass the correct `--collection`."
            ) from None
        raise
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
