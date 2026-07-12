"""Step 08 — SPECTER retrieval fairness baseline (FairSearch-arXiv Team 6).

Implements the 8-step baseline bias audit:
  1. Pick 100 standardized research queries (seeded sample of corpus papers;
     query text = first 30 tokens of the abstract, the known-item heuristic
     from the project deck).
  2. Run SPECTER on each query, retrieve top-K papers (FAISS cosine over
     L2-normalized embeddings; the query's own paper is excluded).
  3. Tag each retrieved paper with its top_uni flag (from 05_rank_and_finalize).
  4. Tag each retrieved paper with relevance: shares >=1 arXiv category token
     with the query paper (same scheme as 06_rag_baseline).
  5. Statistical Parity Difference per query:
       SPD = P(retrieved | top_uni) - P(retrieved | non_top_uni)
     where P(retrieved | g) = (#group-g papers in top-K) / (#group-g papers
     in corpus, excluding the query paper). Mean across queries.
  6. Equalized Odds: TPR gap = TPR_top - TPR_non and FPR gap = FPR_top -
     FPR_non computed per query over the full corpus, then averaged;
     EO = max(|mean TPR gap|, |mean FPR gap|).
  7. Bootstrap 95% CIs by resampling the 100 queries (B=1000, percentile).
  8. Save baseline bias scores to data/baseline_bias_scores.json.

Doc text for embedding follows the SPECTER convention: title [SEP] abstract.
Document embeddings are cached to data/specter_doc_emb.npy so re-runs skip
the expensive encode.

Usage:
  .venv/bin/python scripts/08_specter_fairness_baseline.py
  # options:
  #   --model  sentence-transformers model (default sentence-transformers/allenai-specter;
  #            swap for a SPECTER2 checkpoint once the adapters env is set up)
  #   --k      report cutoffs (default: 10 100)
  #   --n-queries 100  --seed 42  --bootstrap 1000

Outputs (into <root>/data):
  queries_100.jsonl              the standardized query set
  specter_retrieved_topk.csv     per (query, rank) retrieved doc + tags
  fairness_per_query.csv         per-query SPD / TPR / FPR components
  baseline_bias_scores.json      aggregate scores + bootstrap CIs (deliverable)
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/allenai-specter"
QUERY_ABSTRACT_TOKENS = 30


def load_corpus(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def doc_text(r):
    """SPECTER convention: title [SEP] abstract."""
    title = re.sub(r"\s+", " ", (r.get("title") or "")).strip()
    abstract = re.sub(r"\s+", " ", (r.get("abstract") or "")).strip()
    return f"{title} [SEP] {abstract}"


def query_text(r):
    """Known-item heuristic from the deck: first 30 tokens of the abstract."""
    abstract = re.sub(r"\s+", " ", (r.get("abstract") or "")).strip()
    toks = abstract.split()
    if toks:
        return " ".join(toks[:QUERY_ABSTRACT_TOKENS])
    return re.sub(r"\s+", " ", (r.get("title") or "")).strip()


def categories_set(r):
    cats = (r.get("categories") or "").strip()
    return set(cats.split()) - {""}


def embed(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def get_doc_embeddings(model, corpus, cache_path, batch_size, chunk_size):
    """Encode the whole corpus, resumable.

    Embeddings are streamed into an on-disk float32 memmap so a crash costs at
    most one chunk. Progress (# docs completed) is tracked in a sidecar
    <cache>.progress file. A finished, contiguous .npy is written at the end.
    """
    n = len(corpus)
    if os.path.exists(cache_path):
        emb = np.load(cache_path, mmap_mode="r")
        if emb.shape[0] == n:
            print(f"Loaded cached doc embeddings {emb.shape} from {cache_path}")
            return np.asarray(emb)
        print(f"  cache has {emb.shape[0]} rows != corpus {n}; re-encoding")

    memmap_path = cache_path + ".partial"
    progress_path = cache_path + ".progress"

    # Determine embedding dim from a single probe encode.
    dim = int(embed(model, [doc_text(corpus[0])], 1).shape[1])

    done = 0
    if os.path.exists(memmap_path) and os.path.exists(progress_path):
        try:
            done = int(open(progress_path).read().strip())
        except ValueError:
            done = 0
        mm = np.memmap(memmap_path, dtype="float32", mode="r+", shape=(n, dim))
        print(f"Resuming encode from doc {done:,}/{n:,}")
    else:
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n, dim))
        done = 0
        print(f"Encoding {n:,} docs (batch={batch_size}, chunk={chunk_size})...")

    texts = [doc_text(r) for r in corpus]
    t0 = time.monotonic()
    for start in range(done, n, chunk_size):
        end = min(start + chunk_size, n)
        mm[start:end] = embed(model, texts[start:end], batch_size)
        mm.flush()
        with open(progress_path, "w") as f:
            f.write(str(end))
        elapsed = time.monotonic() - t0
        rate = (end - done) / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        print(f"  {end:,}/{n:,} docs  ({rate:.0f} docs/s, ETA {eta/60:.1f} min)",
              flush=True)

    emb = np.array(mm)
    np.save(cache_path, emb)
    del mm
    os.remove(memmap_path)
    os.remove(progress_path)
    print(f"Cached doc embeddings {emb.shape} to {cache_path}")
    return emb


def bootstrap_ci(values_per_query, agg_fn, n_boot, seed):
    """Percentile 95% CI of agg_fn applied to resampled query rows.

    values_per_query: dict of name -> np.array (len = n queries); agg_fn takes
    the resampled dict and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(values_per_query.values())))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = agg_fn({k: v[idx] for k, v in values_per_query.items()})
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 100],
                    help="top-K cutoffs to evaluate (first one is primary)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2048,
                    help="docs per checkpoint flush during encoding")
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="truncate doc/query tokens (SPECTER default 512; "
                         "256 ~halves encode time with little retrieval loss)")
    args = ap.parse_args()

    data_dir = os.path.join(args.root, "data")
    corpus_path = os.path.join(data_dir, "final_enriched.jsonl")
    emb_cache = os.path.join(data_dir, "specter_doc_emb.npy")
    queries_out = os.path.join(data_dir, f"queries_{args.n_queries}.jsonl")
    retrieved_out = os.path.join(data_dir, "specter_retrieved_topk.csv")
    per_query_out = os.path.join(data_dir, "fairness_per_query.csv")
    scores_out = os.path.join(data_dir, "baseline_bias_scores.json")

    print(f"Loading corpus from {corpus_path} ...")
    corpus = load_corpus(corpus_path)
    n_docs = len(corpus)
    top_uni = np.array([bool(r.get("top_uni")) for r in corpus])
    cats = [categories_set(r) for r in corpus]
    n_top_total = int(top_uni.sum())
    print(f"  {n_docs:,} docs | top_uni=True: {n_top_total:,} "
          f"({100*n_top_total/n_docs:.2f}%)")

    # inverted index: category -> doc rows (for fast relevant-set construction)
    cat_index = defaultdict(list)
    for i, cs in enumerate(cats):
        for c in cs:
            cat_index[c].append(i)

    # ---- Step 1: pick the standardized queries ----
    import random
    rng = random.Random(args.seed)
    q_rows = sorted(rng.sample(range(n_docs), args.n_queries))
    with open(queries_out, "w") as f:
        for qi in q_rows:
            r = corpus[qi]
            f.write(json.dumps({
                "query_arxiv_id": r["id"],
                "query_text": query_text(r),
                "categories": sorted(categories_set(r)),
                "source_paper_top_uni": bool(top_uni[qi]),
            }) + "\n")
    print(f"Step 1: wrote {args.n_queries} queries to {queries_out}")

    # ---- Step 2: SPECTER encode + retrieve ----
    from sentence_transformers import SentenceTransformer
    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    print(f"  device={model.device}, max_seq_length={model.max_seq_length}")

    doc_emb = get_doc_embeddings(model, corpus, emb_cache, args.batch_size,
                                 args.chunk_size)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    q_texts = [query_text(corpus[qi]) for qi in q_rows]
    print(f"Encoding {len(q_texts)} queries...")
    q_emb = embed(model, q_texts, args.batch_size)

    k_values = sorted(args.k)
    k_max = max(k_values)
    D, I = index.search(q_emb, k_max + 1)  # +1 leaves room to drop self
    print(f"Step 2: retrieved top-{k_max} for {len(q_rows)} queries")

    # ---- Steps 3+4: tag retrieved docs; Steps 5+6: per-query metrics ----
    retrieved_rows = []
    per_query = []
    for q_pos, qi in enumerate(q_rows):
        q_cats = cats[qi]
        q_rec = corpus[qi]

        # relevant set over the whole corpus (category overlap), excluding self
        rel_rows = set()
        for c in q_cats:
            rel_rows.update(cat_index[c])
        rel_rows.discard(qi)
        rel_mask = np.zeros(n_docs, dtype=bool)
        rel_mask[list(rel_rows)] = True

        hits = [(int(r), float(s)) for r, s in zip(I[q_pos], D[q_pos])
                if int(r) != qi][:k_max]
        for rank, (row, score) in enumerate(hits, 1):
            retrieved_rows.append({
                "query_arxiv_id": q_rec["id"],
                "rank": rank,
                "score": score,
                "doc_arxiv_id": corpus[row]["id"],
                "top_uni": bool(top_uni[row]),          # Step 3
                "relevant": bool(rel_mask[row]),        # Step 4
            })

        # group sizes excluding the query paper
        n_top = n_top_total - int(top_uni[qi])
        n_non = (n_docs - 1) - n_top

        row_out = {"query_arxiv_id": q_rec["id"]}
        for k in k_values:
            topk = np.array([r for r, _ in hits[:k]], dtype=int)
            ret_mask = np.zeros(n_docs, dtype=bool)
            ret_mask[topk] = True

            ret_top = int((ret_mask & top_uni).sum())
            ret_non = int(ret_mask.sum()) - ret_top

            # Step 5: statistical parity difference
            spd = ret_top / n_top - ret_non / n_non

            # Step 6 components: TPR/FPR per group over the full corpus
            rel_top = int((rel_mask & top_uni).sum()) - 0  # self already excluded
            rel_non = int(rel_mask.sum()) - rel_top
            irr_top = n_top - rel_top
            irr_non = n_non - rel_non

            tp_top = int((ret_mask & rel_mask & top_uni).sum())
            tp_non = int((ret_mask & rel_mask & ~top_uni).sum())
            fp_top = ret_top - tp_top
            fp_non = ret_non - tp_non

            tpr_top = tp_top / rel_top if rel_top else np.nan
            tpr_non = tp_non / rel_non if rel_non else np.nan
            fpr_top = fp_top / irr_top if irr_top else np.nan
            fpr_non = fp_non / irr_non if irr_non else np.nan

            row_out.update({
                f"P@{k}": (tp_top + tp_non) / k,
                f"spd@{k}": spd,
                f"tpr_top@{k}": tpr_top, f"tpr_non@{k}": tpr_non,
                f"fpr_top@{k}": fpr_top, f"fpr_non@{k}": fpr_non,
                f"tpr_gap@{k}": tpr_top - tpr_non,
                f"fpr_gap@{k}": fpr_top - fpr_non,
                f"n_relevant_top@{k}": rel_top, f"n_relevant_non@{k}": rel_non,
            })
        per_query.append(row_out)

    pd.DataFrame(retrieved_rows).to_csv(retrieved_out, index=False)
    print(f"Steps 3-4: wrote {retrieved_out} ({len(retrieved_rows):,} rows)")
    pq = pd.DataFrame(per_query)
    pq.to_csv(per_query_out, index=False)
    print(f"Steps 5-6: wrote {per_query_out}")

    # ---- Step 7: bootstrap CIs; Step 8: save scores ----
    results = {
        "config": {
            "model": args.model,
            "corpus": os.path.basename(corpus_path),
            "n_docs": n_docs,
            "n_top_uni": n_top_total,
            "n_queries": args.n_queries,
            "query_protocol": f"first {QUERY_ABSTRACT_TOKENS} abstract tokens "
                              f"(known-item heuristic), seed={args.seed}",
            "relevance": "shares >=1 arXiv category token with the query paper",
            "group_attribute": "top_uni (top-50 mean-citation institutions "
                               "from 05_rank_and_finalize)",
            "bootstrap_resamples": args.bootstrap,
        },
        "metrics": {},
    }
    n_q = len(pq)
    for k in k_values:
        spd = pq[f"spd@{k}"].to_numpy()
        tpr_gap = pq[f"tpr_gap@{k}"].to_numpy()
        fpr_gap = pq[f"fpr_gap@{k}"].to_numpy()

        mean_spd = float(np.nanmean(spd))
        mean_tpr_gap = float(np.nanmean(tpr_gap))
        mean_fpr_gap = float(np.nanmean(fpr_gap))
        eo = max(abs(mean_tpr_gap), abs(mean_fpr_gap))

        spd_ci = bootstrap_ci({"spd": spd},
                              lambda d: np.nanmean(d["spd"]),
                              args.bootstrap, args.seed)
        eo_ci = bootstrap_ci(
            {"tpr": tpr_gap, "fpr": fpr_gap},
            lambda d: max(abs(np.nanmean(d["tpr"])), abs(np.nanmean(d["fpr"]))),
            args.bootstrap, args.seed)

        results["metrics"][f"k={k}"] = {
            "mean_P@k": float(pq[f"P@{k}"].mean()),
            "statistical_parity_difference": {
                "mean": mean_spd, "ci95": list(spd_ci),
                "definition": "P(retrieved|top_uni) - P(retrieved|non_top_uni), "
                              "mean over queries",
            },
            "equalized_odds": {
                "value": eo, "ci95": list(eo_ci),
                "mean_tpr_gap": mean_tpr_gap,
                "mean_fpr_gap": mean_fpr_gap,
                "definition": "max(|mean TPR gap|, |mean FPR gap|), "
                              "gaps are top_uni minus non_top_uni",
            },
            "n_queries_evaluated": n_q,
        }

    with open(scores_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Steps 7-8: wrote {scores_out}")

    print("\n=== BASELINE BIAS SCORES ===")
    for k in k_values:
        m = results["metrics"][f"k={k}"]
        spd_m = m["statistical_parity_difference"]
        eo_m = m["equalized_odds"]
        print(f"  K={k:<4} P@k={m['mean_P@k']:.3f}")
        print(f"    SPD  = {spd_m['mean']:+.3e}  "
              f"CI95 [{spd_m['ci95'][0]:+.3e}, {spd_m['ci95'][1]:+.3e}]")
        print(f"    EO   = {eo_m['value']:.3e}  "
              f"CI95 [{eo_m['ci95'][0]:.3e}, {eo_m['ci95'][1]:.3e}]  "
              f"(TPRgap {eo_m['mean_tpr_gap']:+.3e}, "
              f"FPRgap {eo_m['mean_fpr_gap']:+.3e})")
    print("\nDone.")


if __name__ == "__main__":
    main()
