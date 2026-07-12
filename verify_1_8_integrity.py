"""Integrity check for the steps 1-8 baseline artifacts that step 09 depends on.

Verifies the outputs of 08_specter_fairness_baseline.py are complete,
self-consistent, and reproducible from the corpus, WITHOUT re-running SPECTER:

  A. Shapes/counts       corpus lines == embeddings == n_docs; top_uni count
  B. Embeddings          float32, unit-norm (cosine == dot, as MMR assumes)
  C. Query set           100 unique ids, in corpus, self_top flag correct,
                         query_text == first-30-abstract-tokens heuristic,
                         AND the exact seeded sample is reproducible (seed=42)
  D. Retrieved topk      100 queries x contiguous ranks 1..K, self excluded,
                         scores sorted desc & in [-1,1], ids in corpus
  E. Tags reproducible   top_uni and relevant flags re-derived from the corpus
                         match the CSV (relevant := shares >=1 category token)
  F. Per-query metrics   P@10 recomputed from the CSV matches fairness_per_query;
                         rel-set sizes are k-invariant (@10 == @100)
  G. Aggregate scores    mean SPD/EO over fairness_per_query match
                         baseline_bias_scores.json

Prints PASS/FAIL per check and exits non-zero if any FAIL.

Usage:
  python3 scripts/verify_1_8_integrity.py
"""
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd


QUERY_ABSTRACT_TOKENS = 30          # step 08 constant
SEED = 42                           # step 08 default


def query_text(title, abstract):
    """Reproduce step 08 query_text(): first 30 abstract tokens, else title."""
    abstract = re.sub(r"\s+", " ", (abstract or "")).strip()
    toks = abstract.split()
    if toks:
        return " ".join(toks[:QUERY_ABSTRACT_TOKENS])
    return re.sub(r"\s+", " ", (title or "")).strip()


def categories_set(cats):
    return set((cats or "").strip().split()) - {""}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(root, "data")
    corpus_path = os.path.join(data, "final_enriched.jsonl")

    checks = []                       # (name, ok, detail)

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    # ---- load baseline deliverable (source of truth for sizes) ------------- #
    with open(os.path.join(data, "baseline_bias_scores.json")) as f:
        baseline = json.load(f)
    n_docs_ref = baseline["config"]["n_docs"]
    n_top_ref = baseline["config"]["n_top_uni"]

    # ---- stream corpus once: id->row, top_uni, category sets, query fields -- #
    print("Reading corpus ...")
    ids, id2row, top_uni_list, cat_sets = [], {}, [], []
    title_by_row, abstract_by_row = {}, {}
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r["id"])
            id2row[r["id"]] = i
            top_uni_list.append(bool(r.get("top_uni")))
            cat_sets.append(categories_set(r.get("categories")))
    n_docs = len(ids)
    top_uni = np.array(top_uni_list)

    print("\n== A. shapes / counts ==")
    emb = np.load(os.path.join(data, "specter_doc_emb.npy"), mmap_mode="r")
    check("corpus lines == baseline n_docs", n_docs == n_docs_ref,
          f"{n_docs} vs {n_docs_ref}")
    check("embedding rows == n_docs", emb.shape[0] == n_docs,
          f"{emb.shape} (dim={emb.shape[1]})")
    check("corpus top_uni count == baseline n_top_uni",
          int(top_uni.sum()) == n_top_ref, f"{int(top_uni.sum())} vs {n_top_ref}")
    check("unique corpus ids", len(set(ids)) == n_docs,
          f"{len(set(ids))} unique of {n_docs}")

    print("\n== B. embeddings ==")
    check("dtype float32", emb.dtype == np.float32, str(emb.dtype))
    sample_rows = np.linspace(0, n_docs - 1, 500).astype(int)
    norms = np.linalg.norm(np.asarray(emb[sample_rows], dtype="float64"), axis=1)
    check("unit-norm rows (cosine==dot)", np.allclose(norms, 1.0, atol=1e-3),
          f"norm range [{norms.min():.4f}, {norms.max():.4f}] over 500 rows")

    print("\n== C. query set ==")
    q = [json.loads(l) for l in open(os.path.join(data, "queries_100.jsonl"))]
    qids = [r["query_arxiv_id"] for r in q]
    check("exactly 100 queries", len(q) == 100, str(len(q)))
    check("unique query ids", len(set(qids)) == len(qids))
    check("all query ids in corpus", all(qid in id2row for qid in qids))
    # self_top flag matches corpus
    self_ok = all(bool(r["source_paper_top_uni"]) == top_uni[id2row[r["query_arxiv_id"]]]
                  for r in q)
    check("source_paper_top_uni matches corpus", self_ok)
    # query_text heuristic + categories: needs the query papers' text
    want_rows = {id2row[qid] for qid in qids}
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i in want_rows:
                r = json.loads(line)
                title_by_row[i] = r.get("title")
                abstract_by_row[i] = r.get("abstract")
    qt_ok = cats_ok = True
    for r in q:
        row = id2row[r["query_arxiv_id"]]
        if query_text(title_by_row[row], abstract_by_row[row]) != r["query_text"]:
            qt_ok = False
        if set(r["categories"]) != cat_sets[row]:
            cats_ok = False
    check("query_text == first-30-token heuristic", qt_ok)
    check("query categories match corpus", cats_ok)
    # reproduce the exact seeded sample (step 08: sorted(Random(42).sample(...)))
    repro_rows = sorted(random.Random(SEED).sample(range(n_docs), 100))
    repro_ids = [ids[i] for i in repro_rows]
    check("seeded query selection reproducible (seed=42)", repro_ids == qids,
          "exact id+order match" if repro_ids == qids else "MISMATCH")

    print("\n== D. retrieved top-k ==")
    df = pd.read_csv(os.path.join(data, "specter_retrieved_topk.csv"))
    per_q = df.groupby("query_arxiv_id")
    kmax = int(df["rank"].max())
    check("queries covered == query set", set(df.query_arxiv_id) == set(qids),
          f"{df.query_arxiv_id.nunique()} queries, K={kmax}")
    ranks_ok = all(list(g.sort_values("rank")["rank"]) == list(range(1, kmax + 1))
                   for _, g in per_q)
    check("contiguous ranks 1..K per query", ranks_ok)
    check("no self-retrieval", not (df["doc_arxiv_id"] == df["query_arxiv_id"]).any())
    check("all retrieved ids in corpus", df["doc_arxiv_id"].isin(id2row).all())
    check("scores in [-1, 1]", df["score"].between(-1.0, 1.0).all(),
          f"[{df.score.min():.4f}, {df.score.max():.4f}]")
    desc_ok = all((g.sort_values("rank")["score"].diff().dropna() <= 1e-6).all()
                  for _, g in per_q)
    check("scores non-increasing with rank", desc_ok)

    print("\n== E. tags reproducible from corpus ==")
    qcat = {r["query_arxiv_id"]: cat_sets[id2row[r["query_arxiv_id"]]] for r in q}
    rows = df.to_dict("records")
    top_bad = rel_bad = 0
    for rec in rows:
        drow = id2row[rec["doc_arxiv_id"]]
        if bool(top_uni[drow]) != bool(rec["top_uni"]):
            top_bad += 1
        overlap = len(cat_sets[drow] & qcat[rec["query_arxiv_id"]]) > 0
        if overlap != bool(rec["relevant"]):
            rel_bad += 1
    check("top_uni flags match corpus", top_bad == 0, f"{top_bad} mismatches")
    check("relevant flags == category overlap", rel_bad == 0,
          f"{rel_bad} mismatches / {len(rows)} rows")

    print("\n== F. per-query metrics ==")
    pq = pd.read_csv(os.path.join(data, "fairness_per_query.csv"))
    check("fairness_per_query has 100 rows", len(pq) == 100, str(len(pq)))
    check("rel-set sizes k-invariant (@10==@100)",
          (pq["n_relevant_top@10"].equals(pq["n_relevant_top@100"]) and
           pq["n_relevant_non@10"].equals(pq["n_relevant_non@100"])))
    # recompute P@10 from the retrieved CSV and compare
    top10 = df[df["rank"] <= 10]
    p10 = top10.groupby("query_arxiv_id")["relevant"].mean().rename("p10_recompute")
    merged = pq.set_index("query_arxiv_id").join(p10)
    p10_ok = np.allclose(merged["P@10"], merged["p10_recompute"], atol=1e-9)
    check("P@10 recomputed == fairness_per_query P@10", p10_ok,
          f"max abs diff {np.abs(merged['P@10']-merged['p10_recompute']).max():.2e}")

    print("\n== G. aggregate scores ==")
    for k in (10, 100):
        ref = baseline["metrics"][f"k={k}"]
        spd = float(np.nanmean(pq[f"spd@{k}"]))
        eo = max(abs(float(np.nanmean(pq[f"tpr_gap@{k}"]))),
                 abs(float(np.nanmean(pq[f"fpr_gap@{k}"]))))
        p = float(pq[f"P@{k}"].mean())
        check(f"k={k}: mean SPD matches json",
              abs(spd - ref["statistical_parity_difference"]["mean"]) < 1e-9,
              f"{spd:+.3e} vs {ref['statistical_parity_difference']['mean']:+.3e}")
        check(f"k={k}: EO matches json",
              abs(eo - ref["equalized_odds"]["value"]) < 1e-9,
              f"{eo:.3e} vs {ref['equalized_odds']['value']:.3e}")
        check(f"k={k}: mean P@k matches json", abs(p - ref["mean_P@k"]) < 5e-4,
              f"{p:.4f} vs {ref['mean_P@k']:.4f}")

    # ---- summary ----------------------------------------------------------- #
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{'='*60}\n{len(checks)-n_fail}/{len(checks)} checks PASSED, "
          f"{n_fail} FAILED")
    if n_fail:
        print("FAILED:", ", ".join(n for n, ok, _ in checks if not ok))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()