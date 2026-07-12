"""Naive RAG baseline over the enriched arXiv corpus.

Document text: title + "\n\n" + abstract.
Embed with all-MiniLM-L6-v2. Build a FAISS IndexFlatIP over L2-normalized
embeddings (== cosine similarity). Retrieve top-k.

Relevance scheme (a): same-category. Each arXiv paper has a `categories`
field of space-separated category tokens (e.g., "math.OA math.FA"). A
retrieved doc is RELEVANT to a query doc iff they share >=1 category token.

Eval: sample N=1000 query papers from the corpus (seeded). Query text =
the paper's title. Exclude the query paper itself from retrieved hits.
Report P@k and R@k for k in {1, 5, 10}, with mean across queries.

Outputs:
  data/rag_index.faiss
  data/rag_docids.parquet
  data/rag_eval.csv
  data/rag_README.md
"""
import json
import os
import random
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = "/Users/pavanijain/Desktop/Kaggle_Preprocessing_Project"
FINAL_JSONL = os.path.join(ROOT, "data", "final_enriched.jsonl")
INDEX_PATH = os.path.join(ROOT, "data", "rag_index.faiss")
DOCIDS_PARQUET = os.path.join(ROOT, "data", "rag_docids.parquet")
EVAL_CSV = os.path.join(ROOT, "data", "rag_eval.csv")
README = os.path.join(ROOT, "data", "rag_README.md")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_QUERIES = 1000
K_VALUES = [1, 5, 10]
SEED = 42
BATCH = 256


def load_corpus():
    recs = []
    with open(FINAL_JSONL) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def doc_text(r):
    title = (r.get("title") or "").strip()
    abstract = (r.get("abstract") or "").strip()
    return f"{title}\n\n{abstract}"


def categories_set(r):
    cats = r.get("categories") or ""
    return set(re.split(r"\s+", cats.strip())) - {""}


def build_index(corpus):
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [doc_text(r) for r in corpus]
    print(f"Encoding {len(texts):,} docs (batch={BATCH})...")
    emb = model.encode(
        texts,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    print(f"  embeddings shape: {emb.shape}")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, INDEX_PATH)
    print(f"Wrote {INDEX_PATH}")

    docids = pd.DataFrame({
        "row": np.arange(len(corpus)),
        "arxiv_id": [r["id"] for r in corpus],
        "categories": [r.get("categories") or "" for r in corpus],
    })
    docids.to_parquet(DOCIDS_PARQUET, index=False)
    print(f"Wrote {DOCIDS_PARQUET}")
    return model, index, emb


def evaluate(model, index, emb, corpus):
    rng = random.Random(SEED)
    n = len(corpus)
    q_idx = rng.sample(range(n), min(N_QUERIES, n))

    # for recall denominator: count relevant docs per query in the whole corpus
    cats_per_row = [categories_set(r) for r in corpus]

    queries = [(corpus[i].get("title") or "").strip() for i in q_idx]
    print(f"Encoding {len(queries)} queries...")
    q_emb = model.encode(
        queries,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    K_MAX = max(K_VALUES)
    # retrieve K_MAX + 1 to leave room for excluding the query doc itself
    D, I = index.search(q_emb, K_MAX + 1)

    per_query_rows = []
    p_sums = {k: 0.0 for k in K_VALUES}
    r_sums = {k: 0.0 for k in K_VALUES}
    n_eval = 0

    for q_pos, qi in enumerate(q_idx):
        q_cats = cats_per_row[qi]
        if not q_cats:
            continue  # skip queries with no category
        n_total_relevant = sum(
            1 for j, cs in enumerate(cats_per_row)
            if j != qi and (cs & q_cats)
        )
        if n_total_relevant == 0:
            continue

        # filter self
        retrieved = [int(r) for r in I[q_pos] if int(r) != qi][:K_MAX]

        row = {"query_arxiv_id": corpus[qi]["id"], "n_relevant_in_corpus": n_total_relevant}
        for k in K_VALUES:
            topk = retrieved[:k]
            rel = sum(1 for r in topk if cats_per_row[r] & q_cats)
            p = rel / k
            r = rel / n_total_relevant
            row[f"P@{k}"] = p
            row[f"R@{k}"] = r
            p_sums[k] += p
            r_sums[k] += r
        per_query_rows.append(row)
        n_eval += 1

    print(f"\n=== RAG eval (N={n_eval} queries) ===")
    summary_rows = []
    for k in K_VALUES:
        mP = p_sums[k] / n_eval
        mR = r_sums[k] / n_eval
        print(f"  k={k:>2}  meanP@k={mP:.4f}  meanR@k={mR:.4f}")
        summary_rows.append({"k": k, "mean_P@k": mP, "mean_R@k": mR, "n_queries": n_eval})

    pd.DataFrame(summary_rows).to_csv(EVAL_CSV, index=False)
    pd.DataFrame(per_query_rows).to_csv(
        EVAL_CSV.replace(".csv", "_per_query.csv"), index=False
    )
    print(f"Wrote {EVAL_CSV}")


def write_readme():
    text = f"""# Naive RAG Baseline

## Corpus
- Source: `data/final_enriched.jsonl` (resolved arXiv papers after OpenAlex enrichment)
- Document text: `title + "\\n\\n" + abstract`

## Pipeline
1. Embed each document with `{MODEL_NAME}` (L2-normalized).
2. Build a FAISS `IndexFlatIP` (inner product on normalized vectors == cosine similarity).
3. At query time, embed the query, retrieve top-(k+1), drop the query doc, take top-k.

## Relevance definition (scheme (a) — same-category proxy)
arXiv papers carry a `categories` field of space-separated subject tags
(e.g. `math.OA math.FA`). A retrieved doc is considered *relevant* to a
query doc iff they share at least one category tag.

## Eval protocol
- Sample {N_QUERIES} query papers from the corpus (seed={SEED}).
- Query text = the paper's title (we exclude the paper itself from results).
- For each query, compute:
  - **P@k** = (#relevant in top-k) / k
  - **R@k** = (#relevant in top-k) / (#relevant in the full corpus, excluding self)
- Skip queries with no categories or no relevant docs in the corpus.
- Reported for k = {K_VALUES}.

## Reproduce
```bash
.venv/bin/python scripts/06_rag_baseline.py
```
Outputs land in `data/rag_*`.

## Files
- `rag_index.faiss`         FAISS index
- `rag_docids.parquet`      row → arxiv_id mapping + categories
- `rag_eval.csv`            aggregate P@k / R@k
- `rag_eval_per_query.csv`  per-query breakdown
"""
    with open(README, "w") as f:
        f.write(text)
    print(f"Wrote {README}")


def main():
    print("Loading enriched corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus):,} documents")

    if os.path.exists(INDEX_PATH):
        print(f"FAISS index exists at {INDEX_PATH}. Rebuild? Set FORCE_REBUILD=1.")
        if os.environ.get("FORCE_REBUILD") != "1":
            print("Loading existing index...")
            index = faiss.read_index(INDEX_PATH)
            model = SentenceTransformer(MODEL_NAME)
            emb = None
        else:
            model, index, emb = build_index(corpus)
    else:
        model, index, emb = build_index(corpus)

    evaluate(model, index, emb, corpus)
    write_readme()


if __name__ == "__main__":
    main()
