"""Build a small, self-contained display-data file for the Streamlit fairness
scorecard (app.py) so the demo runs from the repo WITHOUT the 109 MB corpus or
the 152 MB embedding matrix.

For every document that appears in any of the 100 audit queries' SPECTER
top-100 pools (specter_retrieved_topk.csv), we emit its title, first-author
institution, categories, and QS Top-20 proxy group. This union is only a few
thousand docs, so the output is < 1 MB and can live in the repo.

Inputs (data/):
  specter_retrieved_topk.csv   (query, rank, doc_arxiv_id, score, relevant)
  final_enriched.jsonl         (id -> title, categories, canonical_institution_name)
  proxy_labels.csv             (id -> is_privileged, proxy_group, inst_prestige_tier)
Output (data/):
  scorecard_docs.csv           (doc_arxiv_id, title, institution, categories,
                                proxy_group, is_privileged, inst_prestige_tier)

Usage:  python3 scripts/make_scorecard_data.py
"""
import csv
import json
import os

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA = os.path.join(ROOT, "data")


def main():
    # 1) union of doc ids appearing in any pool
    pool_ids = set()
    with open(os.path.join(DATA, "specter_retrieved_topk.csv")) as f:
        for row in csv.DictReader(f):
            pool_ids.add(row["doc_arxiv_id"])
    print(f"Union of pool docs across 100 queries: {len(pool_ids):,}")

    # 2) proxy labels for those ids
    lab = {}
    with open(os.path.join(DATA, "proxy_labels.csv")) as f:
        for row in csv.DictReader(f):
            if row["id"] in pool_ids:
                lab[row["id"]] = {
                    "proxy_group": row["proxy_group"],
                    "is_privileged": row["is_privileged"].strip().lower() == "true",
                    "inst_prestige_tier": row.get("inst_prestige_tier", ""),
                }

    # 3) title / institution / categories from the corpus (stream, filter)
    meta = {}
    with open(os.path.join(DATA, "final_enriched.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in pool_ids:
                meta[r["id"]] = {
                    "title": " ".join((r.get("title") or "").split()),
                    "institution": (r.get("canonical_institution_name")
                                    or r.get("first_author_institution_name") or ""),
                    "categories": (r.get("categories") or "").strip(),
                }

    out_path = os.path.join(DATA, "scorecard_docs.csv")
    n_written = n_priv = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_arxiv_id", "title", "institution", "categories",
                    "proxy_group", "is_privileged", "inst_prestige_tier"])
        for did in sorted(pool_ids):
            m = meta.get(did, {"title": "", "institution": "", "categories": ""})
            lb = lab.get(did, {"proxy_group": "Underrepresented",
                               "is_privileged": False, "inst_prestige_tier": ""})
            w.writerow([did, m["title"], m["institution"], m["categories"],
                        lb["proxy_group"], lb["is_privileged"], lb["inst_prestige_tier"]])
            n_written += 1
            n_priv += int(lb["is_privileged"])
    print(f"Wrote {out_path}  ({n_written:,} docs, {n_priv:,} Privileged)")


if __name__ == "__main__":
    main()
