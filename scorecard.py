"""Pure logic for the FairSearch-arXiv Streamlit fairness scorecard (app.py).

No Streamlit, no torch, no LLM, no embeddings — everything runs on the small
cached files (scorecard_docs.csv, specter_retrieved_topk.csv, queries_100.jsonl)
so it is unit-testable and the demo runs from a fresh clone.

Grouping is the QS Top-20 proxy: Privileged vs Underrepresented. Fair-Top-K is
recomputed live (DetGreedy floor quota protecting Underrepresented), so the app
can expose the representation target as an interactive slider; MMR is loaded
from its precomputed ordering if present (it needs embeddings to recompute).
"""
import csv
import json
import math
import os

# Corpus constants (from baseline_bias_scores_proxy20.json)
N_DOCS = 49652
N_PRIV = 1638
N_UND = N_DOCS - N_PRIV
CORPUS_PRIV_RATE = N_PRIV / N_DOCS            # 0.0330
CORPUS_UNDER_RATE = N_UND / N_DOCS           # 0.9670 (Fair-Top-K default target)


def data_dir():
    """Locate the folder holding the cached files, flat repo or nested data/."""
    env = os.environ.get("FAIRSEARCH_ROOT")
    here = os.path.dirname(os.path.abspath(__file__))
    for base in [b for b in (env, here, os.path.dirname(here)) if b]:
        if os.path.isfile(os.path.join(base, "scorecard_docs.csv")):
            return base
        if os.path.isfile(os.path.join(base, "data", "scorecard_docs.csv")):
            return os.path.join(base, "data")
    return os.path.join(os.path.dirname(here), "data")


def _clean_inst(s):
    s = (s or "").strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s


def load(dd=None):
    """Return (queries, pools, docs, mmr_orders).

    queries : list of {qid, text}
    pools   : {qid: [ {doc_id, rank, score, relevant, is_priv, group,
                       title, institution, categories} ... ]}  (SPECTER top-100)
    docs    : {doc_id: {...meta...}}
    mmr     : {qid: [doc_id,...]} or {} if the precomputed file is absent
    """
    dd = dd or data_dir()
    docs = {}
    with open(os.path.join(dd, "scorecard_docs.csv")) as f:
        for r in csv.DictReader(f):
            docs[r["doc_arxiv_id"]] = {
                "title": r["title"] or "(untitled)",
                "institution": _clean_inst(r["institution"]),
                "categories": r["categories"],
                "group": r["proxy_group"],
                "is_priv": r["is_privileged"].strip().lower() == "true",
                "tier": r.get("inst_prestige_tier", ""),
            }

    pools = {}
    with open(os.path.join(dd, "specter_retrieved_topk.csv")) as f:
        for r in csv.DictReader(f):
            qid = r["query_arxiv_id"]
            d = docs.get(r["doc_arxiv_id"], {})
            pools.setdefault(qid, []).append({
                "doc_id": r["doc_arxiv_id"],
                "rank": int(r["rank"]),
                "score": float(r["score"]),
                "relevant": r["relevant"].strip().lower() == "true",
                "is_priv": bool(d.get("is_priv", False)),
                "group": d.get("group", "Underrepresented"),
                "title": d.get("title", r["doc_arxiv_id"]),
                "institution": d.get("institution", ""),
                "categories": d.get("categories", ""),
            })
    for qid in pools:
        pools[qid].sort(key=lambda x: x["rank"])

    queries = []
    with open(os.path.join(dd, "queries_100.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                queries.append({"qid": r["query_arxiv_id"],
                                "text": r["query_text"]})

    mmr = {}
    mmr_path = os.path.join(dd, "mmr_topk.csv")
    if os.path.isfile(mmr_path):
        with open(mmr_path) as f:
            for r in csv.DictReader(f):
                mmr.setdefault(r["query_arxiv_id"], []).append(
                    (int(r["rank"]), r["doc_arxiv_id"]))
        mmr = {q: [d for _, d in sorted(v)] for q, v in mmr.items()}
    return queries, pools, docs, mmr


# --------------------------------------------------------------------------- #
def rerank(pool, method, min_share=None, mmr_order=None):
    """Return the pool reordered by `method` ('baseline'|'fairtopk'|'mmr').

    fairtopk: DetGreedy floor quota — every prefix of length p holds at least
    floor(min_share * p) Underrepresented (protected) docs. min_share defaults
    to the corpus Underrepresented rate (statistical-parity-to-corpus target).
    """
    if min_share is None:
        min_share = CORPUS_UNDER_RATE
    if method == "baseline":
        return list(pool)
    if method == "mmr" and mmr_order:
        by_id = {d["doc_id"]: d for d in pool}
        ordered = [by_id[i] for i in mmr_order if i in by_id]
        seen = {d["doc_id"] for d in ordered}
        return ordered + [d for d in pool if d["doc_id"] not in seen]
    # fairtopk
    prot = [d for d in pool if not d["is_priv"]]      # Underrepresented
    other = [d for d in pool if d["is_priv"]]         # Privileged
    res, pi, oi, have = [], 0, 0, 0
    n = len(pool)
    for pos in range(1, n + 1):
        if have < math.floor(min_share * pos) and pi < len(prot):
            res.append(prot[pi]); pi += 1; have += 1
            continue
        hp = prot[pi] if pi < len(prot) else None
        ho = other[oi] if oi < len(other) else None
        if hp is None and ho is None:
            break
        if ho is None or (hp is not None and hp["score"] >= ho["score"]):
            res.append(hp); pi += 1; have += 1
        else:
            res.append(ho); oi += 1
    return res


def scorecard(pool, ordered, k):
    """Fairness + utility metrics for the top-k of `ordered`."""
    sel = ordered[:k]
    ret_priv = sum(1 for d in sel if d["is_priv"])
    n_rel = sum(1 for d in sel if d["relevant"])
    pool_rel = sum(1 for d in pool if d["relevant"])

    priv_share = ret_priv / k if k else 0.0
    spd = ret_priv / N_PRIV - (k - ret_priv) / N_UND
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(sel, 1) if d["relevant"])
    ideal = min(pool_rel, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "priv_in_topk": ret_priv,
        "priv_share": priv_share,
        "spd": spd,
        "ndcg": ndcg,
        "precision": n_rel / k if k else 0.0,
        "n_relevant": n_rel,
        "corpus_priv_rate": CORPUS_PRIV_RATE,
    }
