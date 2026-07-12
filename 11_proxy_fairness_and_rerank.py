"""Step 11 - Fairness audit + mitigation on the Top-20 QS proxy (FairSearch-arXiv).

Supersedes the citation-based analysis (08 baseline + 09 re-rank, which grouped
by the top-50 mean-citation `top_uni` flag). This runs the SAME audit + the SAME
two mitigations, but on the assignment's demographic grouping:

    Privileged      = QS-2026 Top-20 institution (proxy_labels.csv, step 10)
    Underrepresented = everyone else

and it adds the two ranking-utility metrics the rubric asks for, NDCG@10 and MRR,
to every method so the fairness-utility tradeoff can be quantified.

It reuses step 08's CACHED retrieval (specter_retrieved_topk.csv) and CACHED
document embeddings (specter_doc_emb.npy). Retrieval is independent of the group
label, so only the per-doc group tag and the metrics are recomputed - no model
reload, no FAISS. The metric code is the one validated in 09 to reproduce step
08 exactly on the original grouping.

Metrics (all macro-averaged over the 100 queries; bootstrap 95% CIs, B=1000):
  Fairness
    SPD = P(retrieved|Privileged) - P(retrieved|Underrepresented), where
          P(retrieved|g) = (#group-g in top-K)/(#group-g in corpus, excl. query).
          SPD > 0  =>  elite institutions over-retrieved per-capita.
    EO  = max(|mean TPR gap|, |mean FPR gap|), gaps = Privileged - Underrep,
          TPR/FPR per query over the full corpus (binary category relevance).
    privileged_share = fraction of top-K slots held by Privileged papers
          (the most legible signal; corpus base rate = 3.30%).
  Utility (binary relevance = shares >=1 arXiv category token with the query)
    P@k      = (#relevant in top-k)/k
    Recall@k = (#relevant in top-k)/(#relevant in the candidate pool)
    NDCG@k   = DCG@k / IDCG@k, binary gains, log2 discount
    MRR      = 1 / rank of the first relevant doc in the (re-)ranked list

Two audits are produced:
  (A) BASELINE retrieval audit over SPECTER top-100 at k in {10, 100}
      -> baseline_bias_scores_proxy20.json, fairness_per_query_proxy20.csv
      (RQ1 / Slide 5 / demographic+bias criterion)
  (B) MITIGATION comparison over the top-50 pool at k in {5, 10}: Baseline vs
      MMR (lambda sweep) vs Fair-Top-K
      -> comparison_table.csv, rerank_bias_scores.json, mmr_topk.csv,
         fairtopk_topk.csv, mmr_lambda_sweep.csv
      (RQ3 / Slide 7 / mitigation criterion)

Usage:
  python3 scripts/11_proxy_fairness_and_rerank.py
  #   --pool 50 --k 5 10 --audit-k 10 100 --lambdas 0.3 0.5 0.7
  #   --min-underrep-share X  (Fair-Top-K target; default: corpus underrep share)
  #   --bootstrap 1000 --seed 42
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def load_corpus_fields(path):
    """Stream corpus -> (ids, id2row, list of category sets)."""
    ids, id2row, cats = [], {}, []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r["id"])
            id2row[r["id"]] = i
            cs = set((r.get("categories") or "").strip().split()) - {""}
            cats.append(cs)
    return ids, id2row, cats


def bootstrap_ci(arrays, agg_fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays.values())))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = agg_fn({k: v[idx] for k, v in arrays.items()})
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- #
def per_query_metrics(order, cand, q, k):
    """Metrics for one query. `order` = full (re-)ranked list of pool positions;
    utility/fairness use the top-k prefix, MRR uses the full list."""
    pref = [cand[i] for i in order[:k]]
    ret_priv = sum(1 for d in pref if d["priv"])
    ret_und = len(pref) - ret_priv
    n_rel = sum(1 for d in pref if d["rel"])

    n_priv, n_und = q["n_priv"], q["n_und"]
    rel_priv, rel_und = q["rel_priv"], q["rel_und"]
    irr_priv, irr_und = n_priv - rel_priv, n_und - rel_und

    tp_priv = sum(1 for d in pref if d["priv"] and d["rel"])
    tp_und = sum(1 for d in pref if (not d["priv"]) and d["rel"])
    fp_priv, fp_und = ret_priv - tp_priv, ret_und - tp_und

    spd = (ret_priv / n_priv - ret_und / n_und) if n_priv and n_und else np.nan
    tpr_priv = tp_priv / rel_priv if rel_priv else np.nan
    tpr_und = tp_und / rel_und if rel_und else np.nan
    fpr_priv = fp_priv / irr_priv if irr_priv else np.nan
    fpr_und = fp_und / irr_und if irr_und else np.nan

    # NDCG@k with binary gains
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, d in enumerate(pref, 1) if d["rel"])
    ideal_n = min(q["pool_rel"], k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    ndcg = dcg / idcg if idcg else 0.0

    # MRR over the full re-ranked list
    rr = 0.0
    for rank, i in enumerate(order, 1):
        if cand[i]["rel"]:
            rr = 1.0 / rank
            break

    return {
        "P": n_rel / k,
        "recall": (n_rel / q["pool_rel"]) if q["pool_rel"] else np.nan,
        "ndcg": ndcg,
        "rr": rr,
        "spd": spd,
        "tpr_gap": tpr_priv - tpr_und,
        "fpr_gap": fpr_priv - fpr_und,
        "priv_share": ret_priv / len(pref) if pref else np.nan,
    }


def aggregate(rows, n_boot, seed):
    arr = {m: np.array([r[m] for r in rows], dtype=float)
           for m in ("P", "recall", "ndcg", "rr", "spd", "tpr_gap",
                     "fpr_gap", "priv_share")}
    mtpr, mfpr = float(np.nanmean(arr["tpr_gap"])), float(np.nanmean(arr["fpr_gap"]))
    out = {
        "mean_P": float(np.nanmean(arr["P"])),
        "mean_recall": float(np.nanmean(arr["recall"])),
        "mean_ndcg": float(np.nanmean(arr["ndcg"])),
        "mean_mrr": float(np.nanmean(arr["rr"])),
        "mean_priv_share": float(np.nanmean(arr["priv_share"])),
        "spd_mean": float(np.nanmean(arr["spd"])),
        "eo": max(abs(mtpr), abs(mfpr)),
        "mean_tpr_gap": mtpr, "mean_fpr_gap": mfpr,
    }
    out["spd_ci95"] = list(bootstrap_ci({"x": arr["spd"]},
                                        lambda d: np.nanmean(d["x"]), n_boot, seed))
    out["eo_ci95"] = list(bootstrap_ci(
        {"t": arr["tpr_gap"], "f": arr["fpr_gap"]},
        lambda d: max(abs(np.nanmean(d["t"])), abs(np.nanmean(d["f"]))),
        n_boot, seed))
    out["ndcg_ci95"] = list(bootstrap_ci({"x": arr["ndcg"]},
                                         lambda d: np.nanmean(d["x"]), n_boot, seed))
    out["mrr_ci95"] = list(bootstrap_ci({"x": arr["rr"]},
                                        lambda d: np.nanmean(d["x"]), n_boot, seed))
    return out


# --------------------------------------------------------------------------- #
def mmr_order(cand, sim, lam, out_len):
    rel = np.array([c["score"] for c in cand], dtype=float)
    sel, rem = [], list(range(len(cand)))
    while rem and len(sel) < out_len:
        if not sel:
            best = max(rem, key=lambda i: rel[i])
        else:
            max_sim = sim[np.ix_(rem, sel)].max(axis=1)
            best = rem[int(np.argmax(lam * rel[rem] - (1 - lam) * max_sim))]
        sel.append(best)
        rem.remove(best)
    return sel


def fair_topk_order(cand, min_share, out_len):
    """DetGreedy floor-quota; protected group = Underrepresented (not priv)."""
    prot = [i for i, c in enumerate(cand) if not c["priv"]]
    other = [i for i, c in enumerate(cand) if c["priv"]]
    res, pi, oi, have = [], 0, 0, 0
    for pos in range(1, out_len + 1):
        if have < math.floor(min_share * pos) and pi < len(prot):
            res.append(prot[pi]); pi += 1; have += 1
            continue
        hp = prot[pi] if pi < len(prot) else None
        ho = other[oi] if oi < len(other) else None
        if hp is None and ho is None:
            break
        if ho is None or (hp is not None and cand[hp]["score"] >= cand[ho]["score"]):
            res.append(hp); pi += 1; have += 1
        else:
            res.append(ho); oi += 1
    return res


def pick_best_lambda(sweep, pk):
    lams = list(sweep)
    P = np.array([sweep[l][pk]["mean_ndcg"] for l in lams])   # utility = NDCG@primary
    aspd = np.array([abs(sweep[l][pk]["spd_mean"]) for l in lams])
    eo = np.array([sweep[l][pk]["eo"] for l in lams])

    def mm(x, hi):
        r = x.max() - x.min()
        if r == 0:
            return np.ones_like(x)
        z = (x - x.min()) / r
        return z if hi else 1 - z

    comp = 0.5 * mm(P, True) + 0.25 * mm(aspd, False) + 0.25 * mm(eo, False)
    best = max(range(len(lams)), key=lambda i: (comp[i], -aspd[i]))
    tbl = {lams[i]: {"composite": float(comp[i]), "ndcg": float(P[i]),
                     "abs_spd": float(aspd[i]), "eo": float(eo[i])}
           for i in range(len(lams))}
    return lams[best], tbl


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=root)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--audit-k", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--min-underrep-share", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = os.path.join(args.root, "data")
    ks = sorted(args.k)
    pk = 10 if 10 in ks else ks[-1]   # primary cutoff = NDCG@10 (rubric headline)
    audit_ks = sorted(args.audit_k)

    # ---- corpus, labels, category index ------------------------------------ #
    print("Loading corpus + proxy labels ...")
    ids, id2row, cats = load_corpus_fields(os.path.join(data, "final_enriched.jsonl"))
    n_docs = len(ids)
    lbl = pd.read_csv(os.path.join(data, "proxy_labels.csv"))
    priv_by_id = dict(zip(lbl["id"].astype(str), lbl["is_privileged"].astype(bool)))
    priv = np.array([bool(priv_by_id.get(str(i), False)) for i in ids])
    n_priv_total = int(priv.sum())
    und_share = (n_docs - n_priv_total) / n_docs
    min_share = args.min_underrep_share if args.min_underrep_share is not None else und_share
    print(f"  {n_docs:,} docs | Privileged={n_priv_total:,} "
          f"({100*n_priv_total/n_docs:.2f}%) | Underrep share={und_share:.4f}")
    print(f"  Fair-Top-K min Underrep share = {min_share:.4f} (floor quota)")

    cat_index = defaultdict(list)
    for i, cs in enumerate(cats):
        for c in cs:
            cat_index[c].append(i)

    # ---- cached retrieval (label-independent) ------------------------------ #
    df = pd.read_csv(os.path.join(data, "specter_retrieved_topk.csv"))
    df = df.sort_values(["query_arxiv_id", "rank"])
    qids = list(dict.fromkeys(df["query_arxiv_id"]))     # preserve first-seen order
    retr = {}
    for qid, g in df.groupby("query_arxiv_id"):
        retr[qid] = [
            {"doc_id": r.doc_arxiv_id, "rank": int(r.rank), "score": float(r.score),
             "rel": bool(r.relevant), "priv": bool(priv[id2row[r.doc_arxiv_id]])}
            for r in g.itertuples(index=False)
        ]
    print(f"Loaded retrieval for {len(qids)} queries (top-{df['rank'].max()})")

    # ---- per-query context (corpus relevant split by group) ---------------- #
    print("Building per-query relevant sets (corpus category overlap) ...")
    qctx = {}
    for qid in qids:
        qrow = id2row[qid]
        qc = cats[qrow]
        rel_rows = set()
        for c in qc:
            rel_rows.update(cat_index[c])
        rel_rows.discard(qrow)
        rel_idx = np.fromiter(rel_rows, dtype=int, count=len(rel_rows))
        rel_priv = int(priv[rel_idx].sum()) if len(rel_idx) else 0
        rel_und = len(rel_rows) - rel_priv
        q_is_priv = bool(priv[qrow])
        n_priv = n_priv_total - (1 if q_is_priv else 0)
        n_und = (n_docs - 1) - n_priv
        qctx[qid] = {"n_priv": n_priv, "n_und": n_und,
                     "rel_priv": rel_priv, "rel_und": rel_und, "q_is_priv": q_is_priv}

    # ======================================================================= #
    # (A) BASELINE retrieval audit over SPECTER top-100                        #
    # ======================================================================= #
    print("\n=== (A) baseline retrieval audit (proxy Top-20) ===")
    audit = {"config": {
        "grouping": "proxy_group: Privileged = QS-2026 Top-20 institution",
        "source": "recomputed from 08 cached retrieval (specter_retrieved_topk.csv)",
        "n_docs": n_docs, "n_privileged": n_priv_total,
        "n_queries": len(qids), "bootstrap": args.bootstrap, "seed": args.seed,
        "relevance": "shares >=1 arXiv category token with the query paper",
    }, "metrics": {}}
    per_q_audit = {qid: {} for qid in qids}
    for k in audit_ks:
        rows = []
        for qid in qids:
            cand = retr[qid]
            q = dict(qctx[qid]); q["pool_rel"] = sum(1 for c in cand if c["rel"])
            m = per_query_metrics(list(range(len(cand))), cand, q, k)
            rows.append(m)
            for key in ("P", "ndcg", "rr", "spd", "tpr_gap", "fpr_gap", "priv_share"):
                per_q_audit[qid][f"{key}@{k}"] = m[key]
        agg = aggregate(rows, args.bootstrap, args.seed)
        audit["metrics"][f"k={k}"] = agg
        print(f"  K={k:<4} P@k={agg['mean_P']:.3f} NDCG@k={agg['mean_ndcg']:.3f} "
              f"MRR={agg['mean_mrr']:.3f} priv%={100*agg['mean_priv_share']:.2f} "
              f"SPD={agg['spd_mean']:+.2e} EO={agg['eo']:.2e}")
    with open(os.path.join(data, "baseline_bias_scores_proxy20.json"), "w") as f:
        json.dump(audit, f, indent=2)
    pd.DataFrame([{"query_arxiv_id": qid, **per_q_audit[qid]} for qid in qids]) \
        .to_csv(os.path.join(data, "fairness_per_query_proxy20.csv"), index=False)
    print("  wrote baseline_bias_scores_proxy20.json + fairness_per_query_proxy20.csv")

    # ======================================================================= #
    # (B) MITIGATION comparison over the top-50 pool                          #
    # ======================================================================= #
    print("\n=== (B) mitigation: Baseline vs MMR vs Fair-Top-K (proxy Top-20) ===")
    pools = {qid: retr[qid][:args.pool] for qid in qids}
    for qid in qids:
        qctx[qid]["pool_rel"] = sum(1 for c in pools[qid] if c["rel"])

    doc_emb = np.load(os.path.join(data, "specter_doc_emb.npy"), mmap_mode="r")
    base_order = {qid: list(range(len(pools[qid]))) for qid in qids}
    fair_order = {qid: fair_topk_order(pools[qid], min_share, args.pool) for qid in qids}
    mmr_orders = {lam: {} for lam in args.lambdas}
    for qid in qids:
        rows_idx = [id2row[c["doc_id"]] for c in pools[qid]]
        E = np.asarray(doc_emb[rows_idx], dtype="float32")
        sim = E @ E.T
        for lam in args.lambdas:
            mmr_orders[lam][qid] = mmr_order(pools[qid], sim, lam, args.pool)

    def score(order_by_q):
        return {k: aggregate([per_query_metrics(order_by_q[qid], pools[qid],
                                                qctx[qid], k) for qid in qids],
                             args.bootstrap, args.seed) for k in ks}

    base_sc = score(base_order)
    fair_sc = score(fair_order)
    mmr_sweep = {lam: score(mmr_orders[lam]) for lam in args.lambdas}
    best_lam, comp_tbl = pick_best_lambda(mmr_sweep, pk)
    mmr_sc = mmr_sweep[best_lam]
    print(f"  MMR lambda sweep, selection @k={pk} (utility=NDCG@{pk}):")
    for lam in args.lambdas:
        c = comp_tbl[lam]
        print(f"    lambda={lam}: composite={c['composite']:.3f} NDCG={c['ndcg']:.3f} "
              f"|SPD|={c['abs_spd']:.2e} EO={c['eo']:.2e}"
              + ("  <-- best" if lam == best_lam else ""))

    # save per-query re-ranked top-K
    def save_topk(order_by_q, path, extra):
        out = []
        for qid in qids:
            for rank, pos in enumerate(order_by_q[qid][:max(ks)], 1):
                c = pools[qid][pos]
                out.append({"query_arxiv_id": qid, "rank": rank, "score": c["score"],
                            "doc_arxiv_id": c["doc_id"], "orig_rank": c["rank"],
                            "privileged": c["priv"], "relevant": c["rel"], **extra})
        pd.DataFrame(out).to_csv(path, index=False)

    save_topk(mmr_orders[best_lam], os.path.join(data, "mmr_topk.csv"),
              {"lambda": best_lam})
    save_topk(fair_order, os.path.join(data, "fairtopk_topk.csv"),
              {"min_underrep_share": round(min_share, 4)})
    sweep_rows = [{"lambda": lam, "k": k, "ndcg": mmr_sweep[lam][k]["mean_ndcg"],
                   "mrr": mmr_sweep[lam][k]["mean_mrr"], "P": mmr_sweep[lam][k]["mean_P"],
                   "spd": mmr_sweep[lam][k]["spd_mean"], "eo": mmr_sweep[lam][k]["eo"],
                   "priv_share": mmr_sweep[lam][k]["mean_priv_share"],
                   "chosen": lam == best_lam}
                  for lam in args.lambdas for k in ks]
    pd.DataFrame(sweep_rows).to_csv(os.path.join(data, "mmr_lambda_sweep.csv"), index=False)

    # rerank bias-score deliverable
    def strip(sc):
        return {f"k={k}": sc[k] for k in ks}
    rerank = {
        "config": {
            "grouping": "Privileged = QS-2026 Top-20 institution (proxy_labels.csv)",
            "n_docs": n_docs, "n_privileged": n_priv_total, "n_queries": len(qids),
            "pool": args.pool, "k_values": ks, "mmr_lambdas": args.lambdas,
            "mmr_best_lambda": best_lam,
            "mmr_selection": "argmax 0.5*NDCG + 0.25*(1-|SPD|) + 0.25*(1-EO), "
                             "min-max normalized at primary k",
            "fairtopk_min_underrep_share": min_share,
            "fairtopk_rule": "DetGreedy floor-quota protecting Underrepresented",
            "bootstrap": args.bootstrap, "seed": args.seed,
        },
        "baseline": strip(base_sc), "mmr": strip(mmr_sc), "fair_topk": strip(fair_sc),
        "mmr_lambda_sweep": comp_tbl,
    }
    with open(os.path.join(data, "rerank_bias_scores.json"), "w") as f:
        json.dump(rerank, f, indent=2)

    # comparison table (step 17, now with NDCG@k + MRR)
    methods = [("Baseline (SPECTER)", base_sc),
               (f"MMR (lambda={best_lam})", mmr_sc), ("Fair-Top-K", fair_sc)]
    tbl = []
    for name, sc in methods:
        for k in ks:
            s = sc[k]
            tbl.append({"method": name, "k": k,
                        "P@k": round(s["mean_P"], 4),
                        "Recall@k": round(s["mean_recall"], 4),
                        "NDCG@k": round(s["mean_ndcg"], 4),
                        "MRR": round(s["mean_mrr"], 4),
                        "priv_share": round(s["mean_priv_share"], 4),
                        "SPD": s["spd_mean"], "SPD_ci_lo": s["spd_ci95"][0],
                        "SPD_ci_hi": s["spd_ci95"][1],
                        "EO": s["eo"], "EO_ci_lo": s["eo_ci95"][0],
                        "EO_ci_hi": s["eo_ci95"][1]})
    pd.DataFrame(tbl).to_csv(os.path.join(data, "comparison_table.csv"), index=False)

    ch_mmr = sum(mmr_orders[best_lam][q][:max(ks)] != base_order[q][:max(ks)] for q in qids)
    ch_fair = sum(fair_order[q][:max(ks)] != base_order[q][:max(ks)] for q in qids)
    print(f"  orderings changed vs baseline (top-{max(ks)}): "
          f"MMR={ch_mmr}/100, Fair-Top-K={ch_fair}/100")
    print("  wrote comparison_table.csv, rerank_bias_scores.json, mmr_topk.csv, "
          "fairtopk_topk.csv, mmr_lambda_sweep.csv")

    # ---- Markdown (paste into report Table 2 / Slide 7) -------------------- #
    print("\n=== MITIGATION COMPARISON (Top-20 proxy) — Slide 7 / report Table 2 ===\n")
    hdr = ["Method", "K", "NDCG@k", "MRR", "P@k", "Recall@k", "priv%",
           "SPD [95% CI]", "EO [95% CI]"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for name, sc in methods:
        for k in ks:
            s = sc[k]
            print(f"| {name} | {k} | {s['mean_ndcg']:.3f} | {s['mean_mrr']:.3f} "
                  f"| {s['mean_P']:.3f} | {s['mean_recall']:.3f} "
                  f"| {100*s['mean_priv_share']:.2f}% "
                  f"| {s['spd_mean']:+.2e} [{s['spd_ci95'][0]:+.1e},{s['spd_ci95'][1]:+.1e}] "
                  f"| {s['eo']:.2e} [{s['eo_ci95'][0]:.1e},{s['eo_ci95'][1]:.1e}] |")
    print("\nDone.")


if __name__ == "__main__":
    main()