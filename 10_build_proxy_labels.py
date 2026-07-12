"""Step 10 - Demographic proxy labels for the FULL corpus (FairSearch-arXiv Team 6).

The fairness audit needs a per-paper PRIVILEGED vs UNDERREPRESENTED label.
07_demographic_enrich.py already derived an institution-prestige proxy from the
QS-2026 world university ranking (elite cutoff = global rank <= 20), but it only
persisted the enriched *institution table* and a 300-doc sample - not a
per-paper label over all 49,652 docs. This step materializes that label for the
whole corpus so 11_proxy_fairness_and_rerank.py can audit on it.

LABELING DECISION (justified for the report / Slide 3):
  * Privileged  := the paper's first-author institution is a QS-2026 Top-20
                   university (is_elite_institution == True in
                   institution_ranking_enriched.csv, elite cutoff rank<=20).
  * Underrepresented := everything else. Following 07 (underrep_includes=both)
                   this deliberately pools THREE cases: (a) ranked-but-not-Top-20
                   institutions, (b) institutions absent from the QS list, and
                   (c) papers with no resolved institution. Rationale: the RQ is
                   whether retrieval favours a small set of globally elite,
                   well-resourced institutions; the contrast group is "everyone
                   who is not in that elite set."
  * This is an INSTITUTIONAL-PRESTIGE proxy, NOT a measured demographic of any
    individual. It is used only for corpus-level structural analysis.
  * First-author institution is used (single canonical institution per paper,
    the key 05_rank_and_finalize.py computed) - consistent with 07.

Why Top-20 and not the citation-based `top_uni` flag from 05: the assignment
defines "Privileged" as Top-20 institutions from an external global ranking.
`top_uni` (top-50 by within-sample mean citations) is a DIFFERENT construct and
is kept only as a secondary robustness check. The two disagree materially - e.g.
a high-mean-citation observatory can be `top_uni=True` yet QS-unranked.

Inputs:
  data/final_enriched.jsonl                        (id, canonical_institution_key)
  ranking_outputs_example/institution_ranking_enriched.csv
                                                   (canonical_institution_key ->
                                                    is_elite_institution, tier, rank)
Output:
  data/proxy_labels.csv   columns: id, canonical_institution_key,
                          inst_global_rank, inst_prestige_tier,
                          is_privileged, proxy_group

Usage:
  python3 scripts/10_build_proxy_labels.py
  #   --elite-rank 20   (informational; the elite flag is read from the table)
"""
import argparse
import csv
import json
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=root)
    ap.add_argument("--ranking",
                    default="ranking_outputs_example/institution_ranking_enriched.csv")
    ap.add_argument("--elite-rank", type=int, default=20,
                    help="documented elite cutoff (the flag itself is read from "
                         "the ranking table's is_elite_institution column)")
    args = ap.parse_args()

    data = os.path.join(args.root, "data")
    corpus_path = os.path.join(data, "final_enriched.jsonl")
    ranking_path = os.path.join(args.root, args.ranking)
    out_path = os.path.join(data, "proxy_labels.csv")

    # ---- institution -> prestige attributes -------------------------------- #
    elite, tier, grank = {}, {}, {}
    with open(ranking_path) as f:
        for row in csv.DictReader(f):
            k = row["canonical_institution_key"]
            elite[k] = row["is_elite_institution"].strip().lower() == "true"
            tier[k] = row.get("inst_prestige_tier", "")
            grank[k] = row.get("inst_global_rank", "")
    n_elite_inst = sum(elite.values())
    print(f"Ranking table: {len(elite):,} institutions | "
          f"elite (Top-{args.elite_rank}): {n_elite_inst}")

    # ---- per-paper label --------------------------------------------------- #
    rows = []
    tier_dist, group_dist = Counter(), Counter()
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = r.get("canonical_institution_key") or ""
            is_priv = bool(k) and elite.get(k, False)
            group = "Privileged" if is_priv else "Underrepresented"
            t = tier.get(k, "unranked") if k else "no_institution"
            rows.append({
                "id": r["id"],
                "canonical_institution_key": k,
                "inst_global_rank": grank.get(k, "") if k else "",
                "inst_prestige_tier": t,
                "is_privileged": is_priv,
                "proxy_group": group,
            })
            tier_dist[t] += 1
            group_dist[group] += 1

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_priv = group_dist["Privileged"]
    print(f"\nWrote {out_path}  ({n:,} papers)")
    print(f"  proxy_group: Privileged={n_priv:,} "
          f"({100*n_priv/n:.2f}%) | Underrepresented={group_dist['Underrepresented']:,}")
    print(f"  prestige tiers: {dict(tier_dist)}")
    # validation against 07_demographic_enrich.py summary (Privileged == 1638)
    print(f"\n[validation] Privileged==1638 (07 summary): "
          f"{'PASS' if n_priv == 1638 else 'FAIL ('+str(n_priv)+')'}")


if __name__ == "__main__":
    main()