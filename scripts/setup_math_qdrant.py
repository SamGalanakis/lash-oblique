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
import time
import uuid
from typing import Iterable

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
BATCH_SIZE = 16
POINT_NAMESPACE = uuid.UUID("47c85e95-6a1c-48dd-bfd9-4b7b4a61f487")


def read_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def point_id(subset: str, doc_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{subset}/{doc_id}"))


def batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class OpenRouterEmbedder:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
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
                json={"model": self.model, "input": texts},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item.get("index", 0))
        return [[float(v) for v in item["embedding"]] for item in data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".benchmarks/obliq/data")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="obliq_analogues")
    parser.add_argument("--subsets", default="math,writing")
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--env-file", default="/home/sam/code/lash/.env")
    parser.add_argument("--enable-late", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_env_file(pathlib.Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is required; set it or provide --env-file /home/sam/code/lash/.env"
        )

    data_root = pathlib.Path(args.data_dir)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    docs = []
    for subset in subsets:
        subset_dir = data_root / subset
        subset_docs = list(read_jsonl(subset_dir / "corpus.jsonl"))
        docs.extend((subset, doc) for doc in subset_docs)
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit(f"no docs found for subsets={subsets} in {data_root}")

    dense_model = OpenRouterEmbedder(args.dense_model, api_key)
    late_model = LateInteractionTextEmbedding(model_name=LATE_MODEL) if args.enable_late else None

    dense_size = len(dense_model.embed(["probe"])[0])
    late_size = len(next(late_model.embed(["probe"]))[0]) if late_model else None

    client = QdrantClient(url=args.qdrant_url)
    if args.recreate or not client.collection_exists(args.collection):
        vectors_config = {
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE,
            ),
        }
        if late_size is not None:
            vectors_config["late"] = models.VectorParams(
                size=late_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            )
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    for offset, batch in enumerate(batched(docs, BATCH_SIZE)):
        texts = [doc["text"] for _, doc in batch]
        dense_vectors = dense_model.embed(texts)
        late_vectors = list(late_model.embed(texts)) if late_model else [None] * len(batch)
        points = []
        for (subset, doc), dense, late in zip(batch, dense_vectors, late_vectors):
            doc_id = doc["_id"]
            payload = {
                "subset": subset,
                "doc_id": doc_id,
                "text": doc["text"],
                "metadata": {k: v for k, v in doc.items() if k not in {"_id", "text"}},
            }
            points.append(
                models.PointStruct(
                    id=point_id(subset, doc_id),
                    vector={
                        "dense": [float(v) for v in dense],
                        "bm25": models.Document(text=doc["text"], model=SPARSE_MODEL),
                    },
                    payload=payload,
                )
            )
            if late is not None:
                points[-1].vector["late"] = [[float(v) for v in token] for token in late]
        client.upsert(collection_name=args.collection, points=points)
        print(f"upserted {(offset * BATCH_SIZE) + len(batch)}/{len(docs)}")

    client.create_payload_index(
        collection_name=args.collection,
        field_name="subset",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=args.collection,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print(f"ready collection={args.collection} subsets={','.join(subsets)} docs={len(docs)}")


if __name__ == "__main__":
    main()
