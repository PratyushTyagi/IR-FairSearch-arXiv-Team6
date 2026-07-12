"""Resolve title-only arXiv papers via Semantic Scholar batch endpoint.

Strategy: lookup by ARXIV:<id> in batches of 500 (no fuzzy title search needed,
since arXiv IDs are stable in S2). For each resolved paper, take the first
author's first affiliation as the institution.

Outputs into enrichment_results.parquet:
  arxiv_id, openalex_id, matched_doi, match_method ('arxiv_id'), cited_by_count,
  citation_field ('S2.citationCount'), first_author_institution_id (raw),
  first_author_institution_name (raw), title_similarity (None), resolved,
  source ('semantic_scholar'), status, s2_paper_id

Checkpoints every CHECKPOINT_BATCHES batches.
Resumable: reads existing parquet on start, skips ids already present.
Stale 'unresolved_retryable' rows are eligible for retry (we drop them before
processing so they get tried again).
"""
import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from s2_client import S2BatchClient, first_author_first_affiliation

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
PARQUET = os.path.join(ROOT, "data", "enrichment_results.parquet")
CACHE = os.path.join(ROOT, "cache", "s2")
SUMMARY = os.path.join(ROOT, "cache", "run_summary.txt")

BATCH = 500
RATE = 3.0           # 3 batches/sec was comfortable in testing
CHECKPOINT_BATCHES = 4  # write parquet every N batches (~2K records)
S2_API_KEY = os.environ.get("S2_API_KEY")  # optional


def load_sample():
    out = []
    with open(SAMPLE) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_parquet_dict():
    if not os.path.exists(PARQUET):
        return {}
    df = pd.read_parquet(PARQUET)
    return {row["arxiv_id"]: row.to_dict() for _, row in df.iterrows()}


def flush(existing: dict):
    df = pd.DataFrame(list(existing.values()))
    tmp = PARQUET + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, PARQUET)


def build_row_resolved(rec, paper):
    iid, iname = first_author_first_affiliation(paper)
    cite = paper.get("citationCount")
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI")
    return {
        "arxiv_id": rec["id"],
        "openalex_id": None,
        "s2_paper_id": paper.get("paperId"),
        "matched_doi": doi.lower() if isinstance(doi, str) else None,
        "match_method": "arxiv_id",
        "cited_by_count": int(cite) if isinstance(cite, int) else (int(cite) if cite is not None else None),
        "citation_field": "S2.citationCount",
        "first_author_institution_id": iid,
        "first_author_institution_name": iname,
        "title_similarity": None,
        "resolved": True,
        "source": "semantic_scholar",
        "status": "resolved",
    }


def build_row_unresolved(rec, status):
    return {
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
        "source": "semantic_scholar",
        "status": status,
    }


def main():
    print("Loading sample...")
    records = load_sample()
    print(f"  {len(records):,} records")

    existing = load_parquet_dict()
    print(f"Resumed from parquet: {len(existing):,} rows already present")

    # Drop stale 'unresolved_retryable' rows so they get re-attempted.
    stale = [k for k, v in existing.items() if v.get("status") == "unresolved_retryable"]
    if stale:
        print(f"  Dropping {len(stale):,} stale 'unresolved_retryable' rows for retry")
        for k in stale:
            del existing[k]

    title_only = [r for r in records if not r.get("doi")]
    todo = [r for r in title_only if r["id"] not in existing]
    print(f"Title-only total: {len(title_only):,}  to process: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    client = S2BatchClient(
        cache_dir=CACHE,
        api_key=S2_API_KEY,
        batch_size=BATCH,
        rate_per_sec=RATE,
    )

    t0 = time.monotonic()
    n_resolved = 0
    n_no_hit = 0
    n_retryable = 0
    n_with_affil = 0

    rec_by_id = {r["id"]: r for r in todo}
    arxiv_ids = list(rec_by_id.keys())

    batch_count = 0
    for i in tqdm(range(0, len(arxiv_ids), BATCH), desc="s2-batch"):
        batch_ids = arxiv_ids[i : i + BATCH]
        results, retryable = client.lookup_by_arxiv_ids(batch_ids)
        retry_set = set(retryable)
        for aid in batch_ids:
            rec = rec_by_id[aid]
            if aid in retry_set:
                existing[aid] = build_row_unresolved(rec, "unresolved_retryable")
                n_retryable += 1
                continue
            paper = results.get(aid)
            if paper is None:
                existing[aid] = build_row_unresolved(rec, "unresolved_no_hit")
                n_no_hit += 1
            else:
                row = build_row_resolved(rec, paper)
                existing[aid] = row
                n_resolved += 1
                if row["first_author_institution_id"]:
                    n_with_affil += 1
        batch_count += 1
        if batch_count % CHECKPOINT_BATCHES == 0:
            flush(existing)

    flush(existing)
    total = time.monotonic() - t0
    print(f"\nS2 title-path done in {total:.1f}s")
    print(f"  resolved:               {n_resolved:,}")
    print(f"    with first-aff:       {n_with_affil:,}")
    print(f"  unresolved (not in S2): {n_no_hit:,}")
    print(f"  unresolved_retryable:   {n_retryable:,}")

    # Final summary
    df = pd.read_parquet(PARQUET)
    total_n = len(df)
    resolved_n = int(df["resolved"].sum())
    by_source = df["source"].value_counts().to_dict()
    by_status = df["status"].value_counts().to_dict()
    print(f"\nOverall parquet now: {total_n:,} rows, {resolved_n:,} resolved")
    print(f"  by source: {by_source}")
    print(f"  by status: {by_status}")

    with open(SUMMARY, "a") as f:
        f.write(f"\n# After S2 title-path\n")
        f.write(f"s2_resolved\t{n_resolved}\n")
        f.write(f"s2_no_hit\t{n_no_hit}\n")
        f.write(f"s2_retryable_remaining\t{n_retryable}\n")
        f.write(f"s2_resolved_with_first_aff\t{n_with_affil}\n")
        f.write(f"NOTE: OpenAlex.cited_by_count and S2.citationCount are not computed identically.\n")
        f.write(f"      Institution ranking mixes the two counts (per user instruction).\n")


if __name__ == "__main__":
    main()
