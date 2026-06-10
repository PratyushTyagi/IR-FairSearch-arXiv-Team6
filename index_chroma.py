"""
index_chroma.py
Embed the 50K sample with each dense encoder and store the vectors in a
persistent ChromaDB collection (one collection per model), using cosine space.

Run:  python src/index_chroma.py            # both models
      python src/index_chroma.py minilm     # one model
"""
import json
import sys

import chromadb

import config as C
from encoders import get_encoder


def load_sample():
    with open(C.SAMPLE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_model(model_name, sample, client):
    enc = get_encoder(model_name)
    print(f"[{model_name}] encoding {len(sample):,} papers ...")
    embs = enc.encode([r["title"] for r in sample], [r["abstract"] for r in sample])

    col = client.get_or_create_collection(
        name=model_name, metadata={"hnsw:space": "cosine"}
    )
    print(f"[{model_name}] adding to ChromaDB ...")
    B = 2000
    for i in range(0, len(sample), B):
        chunk = sample[i:i + B]
        col.add(
            ids=[r["id"] for r in chunk],
            embeddings=embs[i:i + B].tolist(),
            documents=[f'{r["title"]}. {r["abstract"]}' for r in chunk],
            metadatas=[{"category": r["primary_category"], "year": r["year"]} for r in chunk],
        )
    print(f"[{model_name}] done -> collection has {col.count():,} vectors")


def main():
    models = sys.argv[1:] or ["minilm", "specter2"]
    sample = load_sample()
    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    for m in models:
        index_model(m, sample, client)


if __name__ == "__main__":
    main()
