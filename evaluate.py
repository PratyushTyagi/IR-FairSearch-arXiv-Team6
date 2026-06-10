"""
evaluate.py
Run the baseline comparison and compute retrieval-quality metrics.

Retrievers:
  - BM25 (lexical, relevance-matched control for RQ1) over the same 50K sample.
  - Dense: each ChromaDB collection (minilm, specter2).

Metrics (macro-averaged over queries): Precision@k, Recall@k, plus nDCG@k
and MRR (nDCG is handy now and required for the RQ3 fairness-utility curve).

Outputs:
  - results/baseline_metrics.csv
  - a Markdown table printed to stdout -> paste into the report (Table 1) and
    Slide 6 of the deck.

Run:  python src/evaluate.py
"""
import json
import math

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

import config as C


# ---------- data ----------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- metrics (binary relevance) ----------
def precision_at_k(ranked, relevant, k):
    return len([d for d in ranked[:k] if d in relevant]) / k

def recall_at_k(ranked, relevant, k):
    return len([d for d in ranked[:k] if d in relevant]) / len(relevant)

def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0

def reciprocal_rank(ranked, relevant):
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------- retrievers ----------
class BM25Retriever:
    name = "BM25 (baseline)"

    def __init__(self, sample):
        self.ids = [r["id"] for r in sample]
        corpus = [f'{r["title"]}. {r["abstract"]}'.lower().split() for r in sample]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k):
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in top]


class DenseRetriever:
    def __init__(self, model_name, client, label):
        self.name = label
        self.model_name = model_name
        self.col = client.get_collection(model_name)
        from encoders import get_encoder
        self.enc = get_encoder(model_name)

    def search(self, query, k):
        # encode the query the same way documents were encoded (title empty)
        vec = self.enc.encode([""], [query])[0].tolist()
        res = self.col.query(query_embeddings=[vec], n_results=k, include=[])
        return res["ids"][0]


# ---------- driver ----------
def evaluate(retriever, queries, ks):
    maxk = max(ks)
    agg = {f"P@{k}": [] for k in ks}
    agg.update({f"R@{k}": [] for k in ks})
    agg.update({f"nDCG@{k}": [] for k in ks})
    agg["MRR"] = []
    for q in queries:
        rel = set(q["relevant_ids"])
        ranked = retriever.search(q["query"], maxk)
        for k in ks:
            agg[f"P@{k}"].append(precision_at_k(ranked, rel, k))
            agg[f"R@{k}"].append(recall_at_k(ranked, rel, k))
            agg[f"nDCG@{k}"].append(ndcg_at_k(ranked, rel, k))
        agg["MRR"].append(reciprocal_rank(ranked, rel))
    return {m: float(np.mean(v)) for m, v in agg.items()}


def main():
    sample = load_jsonl(C.SAMPLE_FILE)
    queries = load_jsonl(C.QUERIES_FILE)
    client = chromadb.PersistentClient(path=C.CHROMA_DIR)

    retrievers = [
        BM25Retriever(sample),
        DenseRetriever("specter2", client, "SPECTER2 (dense)"),
        DenseRetriever("minilm", client, "MiniLM (dense)"),
    ]

    rows = {}
    for r in retrievers:
        print(f"Evaluating {r.name} ...")
        rows[r.name] = evaluate(r, queries, C.K_VALUES)

    # ---- CSV ----
    cols = [f"P@{k}" for k in C.K_VALUES] + [f"R@{k}" for k in C.K_VALUES] \
        + [f"nDCG@{k}" for k in C.K_VALUES] + ["MRR"]
    csv_path = C.RESULTS_DIR / "baseline_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("retriever," + ",".join(cols) + "\n")
        for name, m in rows.items():
            f.write(name + "," + ",".join(f"{m[c]:.4f}" for c in cols) + "\n")
    print(f"\nWrote {csv_path}")

    # ---- Markdown table for the report / slide ----
    show = [f"P@{k}" for k in C.K_VALUES] + [f"R@{k}" for k in C.K_VALUES]
    print("\nPaste into the report (Table 1) and Slide 6:\n")
    print("| Retriever | " + " | ".join(show) + " |")
    print("|" + "---|" * (len(show) + 1))
    for name, m in rows.items():
        print(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in show) + " |")


if __name__ == "__main__":
    main()
