"""Read locally cached OpenAlex DOI responses and build the per-record
enrichment parquet for all DOI papers. No API calls — pure cache replay.
"""
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from openalex_client import first_author_first_institution, norm_doi

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
OUT_PARQUET = os.path.join(ROOT, "data", "enrichment_results.parquet")
DOI_CACHE = os.path.join(ROOT, "cache", "doi")


def doi_cache_path(doi_norm: str) -> str:
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doi_norm)[:200]
    return os.path.join(DOI_CACHE, f"{safe}.json")


def main():
    print("Loading sample...")
    with open(SAMPLE) as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records):,} records")

    doi_records = [(r, norm_doi(r.get("doi"))) for r in records if norm_doi(r.get("doi"))]
    print(f"DOI-bearing records: {len(doi_records):,}")

    existing = {}
    if os.path.exists(OUT_PARQUET):
        df = pd.read_parquet(OUT_PARQUET)
        existing = {row["arxiv_id"]: row.to_dict() for _, row in df.iterrows()}
        print(f"Loaded {len(existing):,} existing rows from {OUT_PARQUET}")

    cache_hit = 0
    cache_miss = 0
    resolved_count = 0
    unresolved_count = 0

    for rec, d in tqdm(doi_records, desc="doi-cache"):
        if rec["id"] in existing:
            continue
        p = doi_cache_path(d)
        if not os.path.exists(p):
            cache_miss += 1
            continue
        try:
            w = json.load(open(p))
        except json.JSONDecodeError:
            w = None
        cache_hit += 1
        if w is None:
            row = {
                "arxiv_id": rec["id"],
                "openalex_id": None,
                "s2_paper_id": None,
                "matched_doi": None,
                "match_method": None,
                "cited_by_count": None,
                "citation_field": None,
                "first_author_institution_id": None,
                "first_author_institution_name": None,
                "title_similarity": None,
                "resolved": False,
                "source": "openalex",
                "status": "unresolved_no_hit",
            }
            unresolved_count += 1
        else:
            iid, iname = first_author_first_institution(w)
            row = {
                "arxiv_id": rec["id"],
                "openalex_id": w.get("id"),
                "s2_paper_id": None,
                "matched_doi": norm_doi(w.get("doi")) or d,
                "match_method": "doi",
                "cited_by_count": w.get("cited_by_count"),
                "citation_field": "OpenAlex.cited_by_count",
                "first_author_institution_id": iid,
                "first_author_institution_name": iname,
                "title_similarity": None,
                "resolved": True,
                "source": "openalex",
                "status": "resolved",
            }
            resolved_count += 1
        existing[rec["id"]] = row

    print(f"\nDOI cache hits: {cache_hit:,}  misses (no cached file): {cache_miss:,}")
    print(f"  resolved:   {resolved_count:,}")
    print(f"  unresolved: {unresolved_count:,}")

    df_out = pd.DataFrame(list(existing.values()))
    tmp = OUT_PARQUET + ".tmp"
    df_out.to_parquet(tmp, index=False)
    os.replace(tmp, OUT_PARQUET)
    print(f"\nWrote {OUT_PARQUET}  ({len(df_out):,} rows)")
    print(df_out["status"].value_counts())
    print(df_out["source"].value_counts())


if __name__ == "__main__":
    main()
