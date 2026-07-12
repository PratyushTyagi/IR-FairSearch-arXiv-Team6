"""Step 12 - Fairness-utility tradeoff / Pareto frontier (FairSearch-arXiv).

The mitigation deliverable (11) reports Baseline vs MMR vs Fair-Top-K at fixed
settings. This step sweeps each re-ranker's control knob and traces the whole
**fairness-utility frontier**, the core deliverable of a fairness-ranking system:
how much retrieval quality (NDCG@k) you give up to reach a given level of group
representation (Privileged/elite share).

  * MMR:        sweep lambda in [0, 1] (0 = pure diversity, 1 = pure relevance
                = baseline order).
  * Fair-Top-K: sweep the min Underrepresented share target in [0, 1]
                (0 = no constraint = baseline; 1 = exclude all elite from the
                top ranks). DetGreedy floor quota, same as step 11.

Fairness axis = mean Privileged share @k (corpus base rate 3.30%); we also log
SPD@k. Utility axis = mean NDCG@k (also P@k, MRR). Grouping is the QS Top-20
proxy (proxy_labels.csv), reusing step 08's cached retrieval + embeddings
(CPU-only; no exposure metric or model reload required).

We evaluate at k=10 AND k=50 over a top-100 pool: at k=10 elite content is so
sparse the frontier is near-binary, but at k=50 there is enough elite mass for a
smooth curve that shows the real tradeoff (and whether one method Pareto-
dominates the other).

Outputs (data/):
  fairness_utility_pareto.csv   one row per (method, knob value, k)
  fairness_utility_pareto.png   NDCG@k vs Privileged share, both methods + baseline

Usage:
  python3 scripts/12_fairness_utility_tradeoff.py
  #   --pool 100 --k 10 50
  #   --mmr-lambdas 0 0.1 ... 1.0
  #   --fairtopk-shares 0 0.9 0.95 0.97 0.98 0.99 0.995 1.0
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_corpus_fields(path):
    ids, id2row, cats = [], {}, []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r["id"]); id2row[r["id"]] = i
            cats.append(set((r.get("categories") or "").strip().split()) - {""})
    return ids, id2row, cats


def mmr_order(rel, sim, lam, out_len):
    sel, rem = [], list(range(len(rel)))
    while rem and len(sel) < out_len:
        if not sel:
            best = max(rem, key=lambda i: rel[i])
        else:
            max_sim = sim[np.ix_(rem, sel)].max(axis=1)
            best = rem[int(np.argmax(lam * rel[rem] - (1 - lam) * max_sim))]
        sel.append(best); rem.remove(best)
    return sel


def fair_topk_order(priv, score, min_share, out_len):
    prot = [i for i in range(len(priv)) if not priv[i]]     # Underrepresented
    other = [i for i in range(len(priv)) if priv[i]]        # Privileged
    res, pi, oi, have = [], 0, 0, 0
    for pos in range(1, out_len + 1):
        if have < math.floor(min_share * pos) and pi < len(prot):
            res.append(prot[pi]); pi += 1; have += 1; continue
        hp = prot[pi] if pi < len(prot) else None
        ho = other[oi] if oi < len(other) else None
        if hp is None and ho is None:
            break
        if ho is None or (hp is not None and score[hp] >= score[ho]):
            res.append(hp); pi += 1; have += 1
        else:
            res.append(ho); oi += 1
    return res


def metrics_at_k(order, cand, q, k):
    pref = [cand[i] for i in order[:k]]
    ret_priv = sum(1 for d in pref if d["priv"])
    n_rel = sum(1 for d in pref if d["rel"])
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(pref, 1) if d["rel"])
    ideal = min(q["pool_rel"], k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    rr = 0.0
    for r, i in enumerate(order, 1):
        if cand[i]["rel"]:
            rr = 1.0 / r; break
    spd = (ret_priv / q["n_priv"] - (len(pref) - ret_priv) / q["n_und"]) \
        if q["n_priv"] and q["n_und"] else np.nan
    return {"ndcg": dcg / idcg if idcg else 0.0, "P": n_rel / k,
            "priv_share": ret_priv / len(pref) if pref else np.nan,
            "spd": spd, "rr": rr}


def agg(order_by_q, pools, qctx, qids, k):
    rows = [metrics_at_k(order_by_q[qid], pools[qid], qctx[qid], k) for qid in qids]
    return {m: float(np.nanmean([r[m] for r in rows]))
            for m in ("ndcg", "P", "priv_share", "spd", "rr")}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=root)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--mmr-lambdas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--fairtopk-shares", type=float, nargs="+",
                    default=[0.0, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
    args = ap.parse_args()
    data = os.path.join(args.root, "data")
    ks = sorted(args.k)

    print("Loading corpus + proxy labels + retrieval ...")
    ids, id2row, cats = load_corpus_fields(os.path.join(data, "final_enriched.jsonl"))
    n_docs = len(ids)
    lbl = pd.read_csv(os.path.join(data, "proxy_labels.csv"))
    priv_by_id = dict(zip(lbl["id"].astype(str), lbl["is_privileged"].astype(bool)))
    priv = np.array([bool(priv_by_id.get(str(i), False)) for i in ids])
    n_priv_total = int(priv.sum())
    base_rate = n_priv_total / n_docs

    cat_index = defaultdict(list)
    for i, cs in enumerate(cats):
        for c in cs:
            cat_index[c].append(i)

    df = pd.read_csv(os.path.join(data, "specter_retrieved_topk.csv")) \
        .sort_values(["query_arxiv_id", "rank"])
    qids = list(dict.fromkeys(df["query_arxiv_id"]))
    pools = {}
    for qid, g in df.groupby("query_arxiv_id"):
        pools[qid] = [{"doc_id": r.doc_arxiv_id, "score": float(r.score),
                       "rel": bool(r.relevant),
                       "priv": bool(priv[id2row[r.doc_arxiv_id]])}
                      for r in g.itertuples(index=False)][:args.pool]

    qctx = {}
    for qid in qids:
        qrow = id2row[qid]
        q_is_priv = bool(priv[qrow])
        n_priv = n_priv_total - (1 if q_is_priv else 0)
        qctx[qid] = {"n_priv": n_priv, "n_und": (n_docs - 1) - n_priv,
                     "pool_rel": sum(1 for c in pools[qid] if c["rel"])}

    doc_emb = np.load(os.path.join(data, "specter_doc_emb.npy"), mmap_mode="r")
    sims, mmr_rel, privs, scores = {}, {}, {}, {}
    for qid in qids:
        rows_idx = [id2row[c["doc_id"]] for c in pools[qid]]
        E = np.asarray(doc_emb[rows_idx], dtype="float32")
        sims[qid] = E @ E.T
        mmr_rel[qid] = np.array([c["score"] for c in pools[qid]])  # query-doc cosine
        privs[qid] = [c["priv"] for c in pools[qid]]
        scores[qid] = [c["score"] for c in pools[qid]]

    # ---- baseline (relevance order) ----
    base_order = {qid: list(range(len(pools[qid]))) for qid in qids}
    base = {k: agg(base_order, pools, qctx, qids, k) for k in ks}

    # ---- sweep MMR + Fair-Top-K ----
    print("Sweeping MMR lambda + Fair-Top-K target ...")
    rows = []
    for k in ks:
        b = base[k]
        rows.append({"method": "Baseline", "knob": np.nan, "k": k, **b})
    for lam in args.mmr_lambdas:
        orders = {qid: mmr_order(mmr_rel[qid], sims[qid], lam, args.pool) for qid in qids}
        for k in ks:
            rows.append({"method": "MMR", "knob": lam, "k": k,
                         **agg(orders, pools, qctx, qids, k)})
    for s in args.fairtopk_shares:
        orders = {qid: fair_topk_order(privs[qid], scores[qid], s, args.pool)
                  for qid in qids}
        for k in ks:
            rows.append({"method": "Fair-Top-K", "knob": s, "k": k,
                         **agg(orders, pools, qctx, qids, k)})

    out = pd.DataFrame(rows)
    csv_path = os.path.join(data, "fairness_utility_pareto.csv")
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(out)} operating points)")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(ks), figsize=(6.2 * len(ks), 5.0), squeeze=False)
    for ax, k in zip(axes[0], ks):
        sub = out[out.k == k]
        for method, color, marker in [("Fair-Top-K", "#2563eb", "o"),
                                      ("MMR", "#dc2626", "s")]:
            m = sub[sub.method == method].sort_values("priv_share")
            ax.plot(m.priv_share * 100, m.ndcg, marker=marker, color=color,
                    label=method, alpha=0.85, ms=6)
        b = sub[sub.method == "Baseline"].iloc[0]
        ax.scatter([b.priv_share * 100], [b.ndcg], marker="*", s=320,
                   color="#111", zorder=5, label="Baseline")
        ax.axvline(base_rate * 100, ls="--", lw=1, color="gray")
        ax.text(base_rate * 100, ax.get_ylim()[0], f" corpus {base_rate*100:.1f}%",
                fontsize=8, color="gray", va="bottom")
        ax.set_xlabel("Privileged (QS Top-20) share in top-k  (%)  ← fairer")
        ax.set_ylabel(f"NDCG@{k}  → higher utility")
        ax.set_title(f"Fairness–utility frontier @k={k}")
        ax.grid(alpha=0.25); ax.legend(loc="best", fontsize=9)
    fig.suptitle("FairSearch-arXiv: fairness–utility tradeoff (QS Top-20 proxy, "
                 f"pool={args.pool})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(data, "fairness_utility_pareto.png")
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")

    # ---- console summary ----
    print("\n=== Fairness-utility summary (Privileged share %  /  NDCG) ===")
    for k in ks:
        sub = out[out.k == k]
        b = sub[sub.method == "Baseline"].iloc[0]
        print(f"\n k={k}:  Baseline priv={100*b.priv_share:.2f}%  NDCG={b.ndcg:.3f}")
        ft = sub[sub.method == "Fair-Top-K"]
        strict = ft.sort_values("priv_share").iloc[0]
        print(f"   Fair-Top-K (strictest): priv={100*strict.priv_share:.2f}%  "
              f"NDCG={strict.ndcg:.3f}  (ΔNDCG {strict.ndcg-b.ndcg:+.3f} "
              f"for Δpriv {100*(strict.priv_share-b.priv_share):+.2f}pp)")
        mmr = sub[sub.method == "MMR"]
        best_mmr = mmr.sort_values("priv_share").iloc[0]
        print(f"   MMR (lowest priv):      priv={100*best_mmr.priv_share:.2f}%  "
              f"NDCG={best_mmr.ndcg:.3f}  (ΔNDCG {best_mmr.ndcg-b.ndcg:+.3f})")
    print("\nDone.")


if __name__ == "__main__":
    main()
