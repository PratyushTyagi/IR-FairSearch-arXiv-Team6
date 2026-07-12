"""Step 09 - Fairness-aware re-ranking of the SPECTER baseline (FairSearch-arXiv Team 6).

Continues the 8-step baseline audit (scripts/08_specter_fairness_baseline.py) with
the re-ranking half of the pipeline, steps 9-17:

   9. For each of the 100 standardized queries, take SPECTER's top-50 as the
      candidate pool for re-ranking.
  10. MMR: greedily pick documents that balance relevance to the query against
      dissimilarity from the already-picked documents, controlled by lambda.
        mmr(d) = lambda * rel(q,d) - (1-lambda) * max_{d' in selected} sim(d,d')
      Both rel(q,d) and sim(d,d') are cosine similarities in the SAME (unit-norm)
      SPECTER space, so the two terms are directly comparable and need no rescaling.
      rel(q,d) is the cached query-doc score from step 08; sim(d,d') is the dot
      product of the two cached doc embeddings.
  11. Sweep lambda in {0.3, 0.5, 0.7} and keep the best tradeoff (selection rule
      documented in pick_best_lambda()).
  12. Save MMR top-K per query -> data/mmr_topk.csv (+ full sweep).
  13. Fair-Top-K (DetGreedy minimum-quota): re-rank the same top-50 so every
      prefix of length p_len contains at least floor(min_share * p_len)
      non_top_uni ("protected") papers. Default min_share = corpus non_top_uni
      proportion (statistical-parity-to-corpus target).
  14. Save Fair-Top-K top-K per query -> data/fairtopk_topk.csv.
  15. Re-compute SPD and Equalized Odds on the MMR output.
  16. Re-compute SPD and Equalized Odds on the Fair-Top-K output.
  17. Stack baseline + MMR + Fair-Top-K into one comparison table with bias
      metrics and precision/recall side by side -> data/comparison_table.csv
      (also printed as Markdown).

Metric definitions are IDENTICAL to step 08 so numbers are directly comparable:
  SPD = P(retrieved|top_uni) - P(retrieved|non_top_uni), where
        P(retrieved|g) = (#group-g papers in top-K) / (#group-g papers in the
        corpus, excluding the query paper); mean over queries.
  EO  = max(|mean TPR gap|, |mean FPR gap|), gaps = top_uni minus non_top_uni,
        TPR/FPR computed per query over the full corpus.
Bootstrap 95% CIs resample the 100 queries (B=1000, percentile) - same as step 08.
Precision@k = (#relevant in top-k)/k. Recall@k is pool-relative:
  (#relevant in top-k)/(#relevant in the 50-doc candidate pool) - the natural
  recall for a re-ranking task (all three methods share the same pool).

This script reads ONLY the cached artifacts from step 08 (it does not reload the
model or re-run retrieval):
  data/specter_retrieved_topk.csv   top-100 candidates/query + score,top_uni,relevant
  data/specter_doc_emb.npy          corpus embeddings (unit-norm) for MMR diversity
  data/final_enriched.jsonl         id -> embedding-row map + corpus top_uni
  data/fairness_per_query.csv       per-query corpus relevant-set sizes (rel_top/non)
  data/queries_100.jsonl            query list (source_paper_top_uni)
  data/baseline_bias_scores.json    corpus sizes + baseline scores (for validation)

Usage:
  python3 scripts/09_rerank_fairness.py
  #   --pool 50            candidate-pool size to re-rank
  #   --k 5 10             report cutoffs (must be <= --pool)
  #   --lambdas 0.3 0.5 0.7
  #   --min-nontop-share X  Fair-Top-K min non_top_uni share (default: corpus share)
  #   --bootstrap 1000 --seed 42
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #
def load_id_to_row(corpus_path):
    """Map arXiv id -> row index in the embedding matrix (= line order)."""
    id2row = {}
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                id2row[json.loads(line)["id"]] = i
    return id2row


def bootstrap_ci(values_per_query, agg_fn, n_boot, seed):
    """Percentile 95% CI of agg_fn over query resamples (identical to step 08)."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(values_per_query.values())))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = agg_fn({k: v[idx] for k, v in values_per_query.items()})
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- #
# Per-query fairness / quality metrics for an ordered top-K output            #
# --------------------------------------------------------------------------- #
def per_query_metrics(order, cand, q, k):
    """Metrics at cutoff k for one query.

    order : list of candidate-pool positions in output order
    cand  : list of dicts (the pool) with 'top_uni','relevant','score'
    q     : dict with n_top, n_non, rel_top, rel_non, pool_rel
    """
    sel = [cand[i] for i in order[:k]]
    ret_top = sum(1 for d in sel if d["top_uni"])
    ret_non = len(sel) - ret_top
    n_rel = sum(1 for d in sel if d["relevant"])

    n_top, n_non = q["n_top"], q["n_non"]
    rel_top, rel_non = q["rel_top"], q["rel_non"]
    irr_top, irr_non = n_top - rel_top, n_non - rel_non

    tp_top = sum(1 for d in sel if d["top_uni"] and d["relevant"])
    tp_non = sum(1 for d in sel if (not d["top_uni"]) and d["relevant"])
    fp_top, fp_non = ret_top - tp_top, ret_non - tp_non

    spd = ret_top / n_top - ret_non / n_non if n_top and n_non else np.nan
    tpr_top = tp_top / rel_top if rel_top else np.nan
    tpr_non = tp_non / rel_non if rel_non else np.nan
    fpr_top = fp_top / irr_top if irr_top else np.nan
    fpr_non = fp_non / irr_non if irr_non else np.nan

    return {
        "P": n_rel / k,
        "recall": (n_rel / q["pool_rel"]) if q["pool_rel"] else np.nan,
        "spd": spd,
        "tpr_gap": tpr_top - tpr_non,
        "fpr_gap": fpr_top - fpr_non,
        "top_uni_share": ret_top / len(sel) if sel else np.nan,
    }


def aggregate(rows, n_boot, seed):
    """Aggregate per-query metric dicts into means, EO and bootstrap CIs."""
    spd = np.array([r["spd"] for r in rows])
    tprg = np.array([r["tpr_gap"] for r in rows])
    fprg = np.array([r["fpr_gap"] for r in rows])
    P = np.array([r["P"] for r in rows])
    R = np.array([r["recall"] for r in rows])
    tus = np.array([r["top_uni_share"] for r in rows])

    mean_spd = float(np.nanmean(spd))
    mtpr, mfpr = float(np.nanmean(tprg)), float(np.nanmean(fprg))
    eo = max(abs(mtpr), abs(mfpr))

    spd_ci = bootstrap_ci({"spd": spd}, lambda d: np.nanmean(d["spd"]),
                          n_boot, seed)
    eo_ci = bootstrap_ci(
        {"tpr": tprg, "fpr": fprg},
        lambda d: max(abs(np.nanmean(d["tpr"])), abs(np.nanmean(d["fpr"]))),
        n_boot, seed)

    return {
        "mean_P": float(np.nanmean(P)),
        "mean_recall": float(np.nanmean(R)),
        "mean_top_uni_share": float(np.nanmean(tus)),
        "spd_mean": mean_spd, "spd_ci95": list(spd_ci),
        "eo": eo, "eo_ci95": list(eo_ci),
        "mean_tpr_gap": mtpr, "mean_fpr_gap": mfpr,
    }


# --------------------------------------------------------------------------- #
# Re-rankers                                                                   #
# --------------------------------------------------------------------------- #
def mmr_order(cand, sim, lam, out_len):
    """Greedy Maximal Marginal Relevance ordering of the candidate pool.

    cand : pool (relevance/score attached); sim : pool x pool cosine matrix.
    Returns a list of pool positions in MMR order (length out_len).
    """
    rel = np.array([c["score"] for c in cand], dtype=float)
    selected, remaining = [], list(range(len(cand)))
    while remaining and len(selected) < out_len:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])       # first = most relevant
        else:
            sub = sim[np.ix_(remaining, selected)]            # |rem| x |sel|
            max_sim = sub.max(axis=1)
            scores = lam * rel[remaining] - (1 - lam) * max_sim
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


def fair_topk_order(cand, min_share, out_len):
    """DetGreedy minimum-quota re-ranking (Fair-Top-K).

    Protected group = non_top_uni. Relevance order within a group = pool rank
    order (candidates arrive already sorted by SPECTER score). At every prefix
    of length pos, require at least floor(min_share * pos) protected items;
    otherwise take the higher-scoring head of the two group queues.
    Returns pool positions in Fair-Top-K order.
    """
    prot = [i for i, c in enumerate(cand) if not c["top_uni"]]   # non_top_uni, ranked
    other = [i for i, c in enumerate(cand) if c["top_uni"]]      # top_uni, ranked
    result, pi, oi, have_prot = [], 0, 0, 0
    for pos in range(1, out_len + 1):
        need_prot = math.floor(min_share * pos)
        if have_prot < need_prot and pi < len(prot):
            result.append(prot[pi]); pi += 1; have_prot += 1
            continue
        head_p = prot[pi] if pi < len(prot) else None
        head_o = other[oi] if oi < len(other) else None
        if head_p is None and head_o is None:
            break
        if head_o is None or (head_p is not None
                              and cand[head_p]["score"] >= cand[head_o]["score"]):
            result.append(head_p); pi += 1; have_prot += 1
        else:
            result.append(head_o); oi += 1
    return result


# --------------------------------------------------------------------------- #
# lambda selection (step 11)                                                   #
# --------------------------------------------------------------------------- #
def pick_best_lambda(sweep, primary_k):
    """Best tradeoff = max of a min-max-normalized composite at the primary K:
        0.50 * P_norm  +  0.25 * (1 - |SPD|_norm)  +  0.25 * (1 - EO_norm)
    i.e. reward precision, penalize absolute bias (|SPD| and EO). Each term is
    min-max scaled across the swept lambdas so the very different magnitudes are
    comparable. Ties break toward lower |SPD|. Fully documented so it can be
    overridden. Returns (best_lambda, table) where table has the composite.
    """
    ks = list(sweep.values())
    P = np.array([s[primary_k]["mean_P"] for s in ks])
    aspd = np.array([abs(s[primary_k]["spd_mean"]) for s in ks])
    eo = np.array([s[primary_k]["eo"] for s in ks])

    def mm(x, higher_better):
        rng = x.max() - x.min()
        if rng == 0:
            return np.ones_like(x)               # all equal -> neutral 1.0
        z = (x - x.min()) / rng
        return z if higher_better else 1 - z

    composite = 0.50 * mm(P, True) + 0.25 * mm(aspd, False) + 0.25 * mm(eo, False)
    lams = list(sweep.keys())
    # tie-break toward lower |SPD|
    best_idx = max(range(len(lams)), key=lambda i: (composite[i], -aspd[i]))
    table = {lams[i]: {"composite": float(composite[i]),
                       "P": float(P[i]), "abs_spd": float(aspd[i]),
                       "eo": float(eo[i])} for i in range(len(lams))}
    return lams[best_idx], table


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=default_root)
    ap.add_argument("--pool", type=int, default=50, help="candidate-pool size")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10],
                    help="report cutoffs (first is primary; must be <= --pool)")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--min-nontop-share", type=float, default=None,
                    help="Fair-Top-K min non_top_uni share "
                         "(default: corpus non_top_uni proportion)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = os.path.join(args.root, "data")
    corpus_path = os.path.join(data, "final_enriched.jsonl")
    ks = sorted(args.k)
    primary_k = ks[0]
    if max(ks) > args.pool:
        raise SystemExit(f"--k {max(ks)} exceeds --pool {args.pool}")

    # ---- corpus sizes (from the baseline deliverable) ---------------------- #
    with open(os.path.join(data, "baseline_bias_scores.json")) as f:
        baseline = json.load(f)
    n_docs = baseline["config"]["n_docs"]
    n_top_uni = baseline["config"]["n_top_uni"]
    corpus_nontop_share = (n_docs - n_top_uni) / n_docs
    min_share = args.min_nontop_share
    if min_share is None:
        min_share = corpus_nontop_share
    print(f"Corpus: {n_docs:,} docs | top_uni={n_top_uni:,} "
          f"({100*n_top_uni/n_docs:.2f}%) | non_top_uni share={corpus_nontop_share:.4f}")
    print(f"Fair-Top-K min non_top_uni share = {min_share:.4f} "
          f"(floor quota; allows <= {args.pool - math.floor(min_share*args.pool)} "
          f"top_uni in the full pool)")

    # ---- per-query corpus relevant-set sizes (from step 08) ---------------- #
    pq = pd.read_csv(os.path.join(data, "fairness_per_query.csv"))
    rel_top_by_q = dict(zip(pq["query_arxiv_id"], pq["n_relevant_top@10"]))
    rel_non_by_q = dict(zip(pq["query_arxiv_id"], pq["n_relevant_non@10"]))

    # ---- query self-membership (from step 08 query set) -------------------- #
    self_top = {}
    q_order = []
    with open(os.path.join(data, "queries_100.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            self_top[r["query_arxiv_id"]] = bool(r["source_paper_top_uni"])
            q_order.append(r["query_arxiv_id"])

    # ---- candidate pools (step 09) ----------------------------------------- #
    print(f"Loading candidates and building top-{args.pool} pools ...")
    df = pd.read_csv(os.path.join(data, "specter_retrieved_topk.csv"))
    df = df[df["rank"] <= args.pool].sort_values(["query_arxiv_id", "rank"])
    pools = {}
    for qid, g in df.groupby("query_arxiv_id"):
        pools[qid] = [
            {"doc_id": row.doc_arxiv_id, "rank": int(row.rank),
             "score": float(row.score), "top_uni": bool(row.top_uni),
             "relevant": bool(row.relevant)}
            for row in g.itertuples(index=False)
        ]

    # ---- embeddings for MMR diversity -------------------------------------- #
    print("Mapping doc ids -> embedding rows ...")
    id2row = load_id_to_row(corpus_path)
    doc_emb = np.load(os.path.join(data, "specter_doc_emb.npy"), mmap_mode="r")

    # per-query context (group sizes + pool relevant count) ------------------ #
    qctx = {}
    for qid in q_order:
        n_top = n_top_uni - (1 if self_top[qid] else 0)
        n_non = (n_docs - 1) - n_top
        pool_rel = sum(1 for c in pools[qid] if c["relevant"])
        qctx[qid] = {"n_top": n_top, "n_non": n_non,
                     "rel_top": int(rel_top_by_q[qid]),
                     "rel_non": int(rel_non_by_q[qid]), "pool_rel": pool_rel}

    # ---- build orderings: baseline, MMR (per lambda), Fair-Top-K ----------- #
    print("Building orderings (baseline / MMR sweep / Fair-Top-K) ...")
    baseline_order = {qid: list(range(len(pools[qid]))) for qid in q_order}  # rank order
    fair_order = {qid: fair_topk_order(pools[qid], min_share, args.pool)
                  for qid in q_order}
    mmr_orders = {lam: {} for lam in args.lambdas}
    for qid in q_order:
        cand = pools[qid]
        rows = [id2row[c["doc_id"]] for c in cand]
        E = np.asarray(doc_emb[rows], dtype="float32")   # pool x dim, unit-norm
        sim = E @ E.T                                     # cosine (dot of unit vecs)
        for lam in args.lambdas:
            mmr_orders[lam][qid] = mmr_order(cand, sim, lam, args.pool)

    # ---- score every ordering at every k ----------------------------------- #
    def score_method(order_by_q):
        out = {}
        for k in ks:
            rows = [per_query_metrics(order_by_q[qid], pools[qid], qctx[qid], k)
                    for qid in q_order]
            agg = aggregate(rows, args.bootstrap, args.seed)
            agg["_rows"] = rows
            out[k] = agg
        return out

    base_scores = score_method(baseline_order)
    fair_scores = score_method(fair_order)
    mmr_sweep = {lam: score_method(mmr_orders[lam]) for lam in args.lambdas}

    # ---- VALIDATION: recomputed baseline must match step 08 ---------------- #
    print("\n[validation] recomputed baseline vs step 08 baseline_bias_scores.json")
    for k in ks:
        ref = baseline["metrics"].get(f"k={k}")
        got = base_scores[k]
        if ref:
            print(f"  k={k}: SPD {got['spd_mean']:+.3e} vs ref "
                  f"{ref['statistical_parity_difference']['mean']:+.3e} | "
                  f"EO {got['eo']:.3e} vs ref {ref['equalized_odds']['value']:.3e} | "
                  f"P@k {got['mean_P']:.3f} vs ref {ref['mean_P@k']:.3f}")
        else:
            print(f"  k={k}: SPD {got['spd_mean']:+.3e}  EO {got['eo']:.3e}  "
                  f"P@k {got['mean_P']:.3f}  (no k={k} in step-08 ref)")

    # ---- step 11: pick best lambda ----------------------------------------- #
    flat_sweep = {lam: {k: {"mean_P": mmr_sweep[lam][k]["mean_P"],
                            "spd_mean": mmr_sweep[lam][k]["spd_mean"],
                            "eo": mmr_sweep[lam][k]["eo"]} for k in ks}
                  for lam in args.lambdas}
    best_lambda, comp_table = pick_best_lambda(flat_sweep, primary_k)
    print(f"\nStep 11: lambda sweep composite @k={primary_k} "
          f"(0.5*P + 0.25*(1-|SPD|) + 0.25*(1-EO), min-max normalized):")
    for lam in args.lambdas:
        c = comp_table[lam]
        mark = "  <-- best" if lam == best_lambda else ""
        print(f"    lambda={lam}: composite={c['composite']:.3f}  "
              f"P={c['P']:.3f}  |SPD|={c['abs_spd']:.3e}  EO={c['eo']:.3e}{mark}")
    mmr_scores = mmr_sweep[best_lambda]

    # ---- steps 12 + 14: save per-query re-ranked top-K --------------------- #
    def save_topk(order_by_q, path, extra=None):
        rows = []
        for qid in q_order:
            for rank, pos in enumerate(order_by_q[qid][:max(ks)], 1):
                c = pools[qid][pos]
                rec = {"query_arxiv_id": qid, "rank": rank, "score": c["score"],
                       "doc_arxiv_id": c["doc_id"], "orig_rank": c["rank"],
                       "top_uni": c["top_uni"], "relevant": c["relevant"]}
                if extra:
                    rec.update(extra)
                rows.append(rec)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    mmr_out = save_topk(mmr_orders[best_lambda],
                        os.path.join(data, "mmr_topk.csv"),
                        extra={"lambda": best_lambda})
    fair_out = save_topk(fair_order, os.path.join(data, "fairtopk_topk.csv"),
                         extra={"min_nontop_share": round(min_share, 4)})
    print(f"\nStep 12: wrote {mmr_out} (lambda={best_lambda})")
    print(f"Step 14: wrote {fair_out}")

    # sweep detail for the report appendix
    sweep_rows = []
    for lam in args.lambdas:
        for k in ks:
            s = mmr_sweep[lam][k]
            sweep_rows.append({"lambda": lam, "k": k, "P": s["mean_P"],
                               "recall": s["mean_recall"], "spd": s["spd_mean"],
                               "eo": s["eo"], "top_uni_share": s["mean_top_uni_share"],
                               "chosen": lam == best_lambda})
    pd.DataFrame(sweep_rows).to_csv(os.path.join(data, "mmr_lambda_sweep.csv"),
                                    index=False)
    print(f"Step 11: wrote {os.path.join(data, 'mmr_lambda_sweep.csv')}")

    # ---- steps 15 + 16 + 8: bias-score deliverable ------------------------- #
    def strip(scores):
        return {f"k={k}": {kk: vv for kk, vv in scores[k].items() if kk != "_rows"}
                for k in ks}

    deliverable = {
        "config": {
            "derived_from": "08_specter_fairness_baseline.py cached artifacts",
            "n_docs": n_docs, "n_top_uni": n_top_uni,
            "n_queries": len(q_order), "pool": args.pool, "k_values": ks,
            "mmr_lambdas": args.lambdas, "mmr_best_lambda": best_lambda,
            "mmr_selection_rule": "argmax 0.5*P + 0.25*(1-|SPD|) + 0.25*(1-EO), "
                                  "min-max normalized across lambdas at primary k",
            "fairtopk_min_nontop_share": min_share,
            "fairtopk_rule": "DetGreedy floor-quota on non_top_uni (protected)",
            "bootstrap_resamples": args.bootstrap, "seed": args.seed,
            "metric_defs": {
                "spd": "P(retrieved|top_uni)-P(retrieved|non_top_uni), mean over queries",
                "eo": "max(|mean TPR gap|,|mean FPR gap|), gaps=top_uni minus non_top_uni",
                "precision": "(#relevant in top-k)/k",
                "recall": "(#relevant in top-k)/(#relevant in the 50-doc pool)",
            },
        },
        "baseline": strip(base_scores),
        "mmr": strip(mmr_scores),
        "fair_topk": strip(fair_scores),
        "mmr_lambda_sweep_composite": comp_table,
    }
    scores_out = os.path.join(data, "rerank_bias_scores.json")
    with open(scores_out, "w") as f:
        json.dump(deliverable, f, indent=2)
    print(f"Steps 15-16: wrote {scores_out}")

    # ---- step 17: stacked comparison table --------------------------------- #
    methods = [("Baseline (SPECTER)", base_scores),
               (f"MMR (lambda={best_lambda})", mmr_scores),
               ("Fair-Top-K", fair_scores)]
    table_rows = []
    for name, sc in methods:
        for k in ks:
            s = sc[k]
            table_rows.append({
                "method": name, "k": k,
                "P@k": round(s["mean_P"], 4),
                "Recall@k(pool)": round(s["mean_recall"], 4),
                "top_uni_share": round(s["mean_top_uni_share"], 4),
                "SPD": s["spd_mean"], "SPD_ci_lo": s["spd_ci95"][0],
                "SPD_ci_hi": s["spd_ci95"][1],
                "EO": s["eo"], "EO_ci_lo": s["eo_ci95"][0],
                "EO_ci_hi": s["eo_ci95"][1],
            })
    comp_df = pd.DataFrame(table_rows)
    comp_csv = os.path.join(data, "comparison_table.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"Step 17: wrote {comp_csv}")

    # binding diagnostics
    changed_fair = sum(1 for qid in q_order
                       if fair_order[qid][:max(ks)] != baseline_order[qid][:max(ks)])
    changed_mmr = sum(1 for qid in q_order
                      if mmr_orders[best_lambda][qid][:max(ks)]
                      != baseline_order[qid][:max(ks)])
    print(f"\nOrderings changed vs baseline (within top-{max(ks)}): "
          f"MMR={changed_mmr}/100, Fair-Top-K={changed_fair}/100")

    # ---- Markdown for the report / slide ----------------------------------- #
    print("\n=== STEP 17 COMPARISON (paste into report Table 2 / Slide 8) ===\n")
    hdr = ["Method", "K", "P@k", "Recall@k", "top_uni%", "SPD [95% CI]", "EO [95% CI]"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for name, sc in methods:
        for k in ks:
            s = sc[k]
            print(f"| {name} | {k} | {s['mean_P']:.3f} | {s['mean_recall']:.3f} "
                  f"| {100*s['mean_top_uni_share']:.2f}% "
                  f"| {s['spd_mean']:+.2e} [{s['spd_ci95'][0]:+.1e}, {s['spd_ci95'][1]:+.1e}] "
                  f"| {s['eo']:.2e} [{s['eo_ci95'][0]:.1e}, {s['eo_ci95'][1]:.1e}] |")
    print("\nDone.")


if __name__ == "__main__":
    main()
