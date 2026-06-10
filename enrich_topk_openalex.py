"""
enrich_topk_openalex.py
Cheap variant of enrich_affiliations_openalex.py: only enrich the papers that
were actually retrieved in the top-k by any retriever for any query.

For 200 queries x K=10 x 3 retrievers, the unique union is typically <2K
papers (vs 50K for full-corpus enrichment), runnable in ~15 minutes vs
overnight. This is what powers Slide 7 "early fairness observations": we
need institution/country labels on retrieved papers, not on the whole
corpus, to talk about retrieval bias.

Set MAILTO before running (OpenAlex polite pool).

Run AFTER index_chroma.py and build_queries.py.
Run:  python src/enrich_topk_openalex.py
"""
import json
import time

import numpy as np
import requests
import chromadb
from rank_bm25 import BM25Okapi

import config as C
from encoders import get_encoder


MAILTO = "your-email@northeastern.edu"   # <-- set this before running
OPENALEX = "https://api.openalex.org/works/arxiv:{arxiv_id}"
K = 10
DENSE_RETRIEVERS = ["specter2", "minilm"]


def fetch_affiliations(arxiv_id):
    """Return list of {author, institution, ror, country} or None on miss."""
    url = OPENALEX.format(arxiv_id=arxiv_id) + f"?mailto={MAILTO}"
    try:
        r = requests.get(url, timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    data = r.json()
    out = []
    for a in data.get("authorships", []):
        for inst in (a.get("institutions") or [{}]):
            out.append({
                "author": (a.get("author") or {}).get("display_name"),
                "institution": inst.get("display_name"),
                "ror": inst.get("ror"),
                "country": inst.get("country_code"),
            })
    return out


def collect_topk_ids(sample, queries):
    """Union of paper ids retrieved in top-K by any retriever for any query."""
    ids_union = set()

    # BM25
    print("Collecting BM25 top-K ...")
    corpus = [f'{r["title"]}. {r["abstract"]}'.lower().split() for r in sample]
    ids_list = [r["id"] for r in sample]
    bm25 = BM25Okapi(corpus)
    for q in queries:
        scores = bm25.get_scores(q["query"].lower().split())
        for i in np.argsort(scores)[::-1][:K]:
            ids_union.add(ids_list[i])

    # Dense retrievers
    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    for model_name in DENSE_RETRIEVERS:
        print(f"Collecting {model_name} top-K ...")
        col = client.get_collection(model_name)
        enc = get_encoder(model_name)
        for q in queries:
            vec = enc.encode([""], [q["query"]])[0].tolist()
            res = col.query(query_embeddings=[vec], n_results=K, include=[])
            ids_union.update(res["ids"][0])

    return ids_union


def main():
    if MAILTO.startswith("your-email"):
        raise SystemExit("Set MAILTO at the top of this file before running.")

    sample = [json.loads(l) for l in open(C.SAMPLE_FILE, "r", encoding="utf-8")]
    queries = [json.loads(l) for l in open(C.QUERIES_FILE, "r", encoding="utf-8")]

    target_ids = sorted(collect_topk_ids(sample, queries))
    print(f"Unique retrieved papers across all retrievers: {len(target_ids):,}")

    out_path = C.DATA_DIR / "topk_affiliations.jsonl"
    resolved = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, pid in enumerate(target_ids):
            affs = fetch_affiliations(pid)
            if affs:
                resolved += 1
            f.write(json.dumps({"id": pid, "affiliations": affs}) + "\n")
            time.sleep(0.1)  # polite ~10 req/s
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(target_ids):,}  resolved={resolved:,}")
    print(f"Done. Resolved {resolved:,}/{len(target_ids):,} "
          f"({resolved/len(target_ids):.1%}) -> {out_path}")
    print("This resolution % is the verifiable number for Slide 7 / Dataset.")


if __name__ == "__main__":
    main()
