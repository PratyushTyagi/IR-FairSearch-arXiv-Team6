"""Enrich the 50K arXiv sample with OpenAlex metadata.

Resumable: leans on the on-disk response cache (cache/doi, cache/title) for
network idempotency, and writes a single parquet checkpoint of per-record
results that we re-read on resume to skip already-processed arxiv_ids.

Output: data/enrichment_results.parquet
"""
import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from openalex_client import (
    OpenAlexClient,
    first_author_first_institution,
    norm_doi,
)

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
OUT_PARQUET = os.path.join(ROOT, "data", "enrichment_results.parquet")
CACHE = os.path.join(ROOT, "cache")
SUMMARY = os.path.join(ROOT, "cache", "run_summary.txt")

EMAIL = "jain.pav@northeastern.edu"
DOI_BATCH = 50
FLUSH_EVERY = 500


def load_sample():
    recs = []
    with open(SAMPLE) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def load_existing_results():
    if not os.path.exists(OUT_PARQUET):
        return {}
    df = pd.read_parquet(OUT_PARQUET)
    return {row["arxiv_id"]: row.to_dict() for _, row in df.iterrows()}


def flush(rows: list[dict], existing: dict):
    if not rows:
        return
    for r in rows:
        existing[r["arxiv_id"]] = r
    df = pd.DataFrame(list(existing.values()))
    tmp = OUT_PARQUET + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUT_PARQUET)


def main():
    print("Loading sample...")
    recs = load_sample()
    print(f"  {len(recs):,} records")

    existing = load_existing_results()
    print(f"Resumed with {len(existing):,} already-processed records")

    todo = [r for r in recs if r["id"] not in existing]
    print(f"  {len(todo):,} remaining")
    if not todo:
        print("Nothing to do.")
        return

    client = OpenAlexClient(email=EMAIL, cache_dir=CACHE, doi_batch_size=DOI_BATCH)

    # split into DOI vs title cohorts
    doi_records = []
    title_records = []
    for r in todo:
        d = norm_doi(r.get("doi"))
        if d:
            doi_records.append((r, d))
        else:
            title_records.append(r)
    print(f"  DOI path:   {len(doi_records):,}")
    print(f"  TITLE path: {len(title_records):,}")

    new_rows: list[dict] = []
    no_inst_count = 0

    # ---- DOI path: batched ----
    print("\n=== DOI batched lookups ===")
    pbar = tqdm(total=len(doi_records), desc="doi", unit="rec")
    for i in range(0, len(doi_records), DOI_BATCH):
        batch = doi_records[i : i + DOI_BATCH]
        dois = [d for _, d in batch]
        works = client.lookup_by_dois(dois)
        for rec, d in batch:
            w = works.get(d)
            if w is None:
                row = {
                    "arxiv_id": rec["id"],
                    "openalex_id": None,
                    "matched_doi": None,
                    "match_method": None,
                    "cited_by_count": None,
                    "first_author_institution_id": None,
                    "first_author_institution_name": None,
                    "title_similarity": None,
                    "resolved": False,
                }
            else:
                iid, iname = first_author_first_institution(w)
                if iid is None:
                    no_inst_count += 1
                row = {
                    "arxiv_id": rec["id"],
                    "openalex_id": w.get("id"),
                    "matched_doi": norm_doi(w.get("doi")) or d,
                    "match_method": "doi",
                    "cited_by_count": w.get("cited_by_count"),
                    "first_author_institution_id": iid,
                    "first_author_institution_name": iname,
                    "title_similarity": None,
                    "resolved": True,
                }
            new_rows.append(row)
        pbar.update(len(batch))
        if len(new_rows) >= FLUSH_EVERY:
            flush(new_rows, existing)
            new_rows = []
    pbar.close()
    flush(new_rows, existing)
    new_rows = []

    # ---- TITLE path: one-by-one ----
    print("\n=== Title-search lookups ===")
    pbar = tqdm(total=len(title_records), desc="title", unit="rec")
    for rec in title_records:
        try:
            w, sim = client.lookup_by_title(rec["title"])
        except Exception as e:
            w, sim = None, 0.0
        if w is None:
            row = {
                "arxiv_id": rec["id"],
                "openalex_id": None,
                "matched_doi": None,
                "match_method": None,
                "cited_by_count": None,
                "first_author_institution_id": None,
                "first_author_institution_name": None,
                "title_similarity": float(sim),
                "resolved": False,
            }
        else:
            iid, iname = first_author_first_institution(w)
            if iid is None:
                no_inst_count += 1
            row = {
                "arxiv_id": rec["id"],
                "openalex_id": w.get("id"),
                "matched_doi": norm_doi(w.get("doi")),
                "match_method": "title",
                "cited_by_count": w.get("cited_by_count"),
                "first_author_institution_id": iid,
                "first_author_institution_name": iname,
                "title_similarity": float(sim),
                "resolved": True,
            }
        new_rows.append(row)
        pbar.update(1)
        if len(new_rows) >= FLUSH_EVERY:
            flush(new_rows, existing)
            new_rows = []
    pbar.close()
    flush(new_rows, existing)

    # ---- Summary ----
    df = pd.read_parquet(OUT_PARQUET)
    total = len(df)
    resolved = int(df["resolved"].sum())
    dropped_no_resolve = total - resolved
    dropped_no_inst = int(
        (df["resolved"] & df["first_author_institution_id"].isna()).sum()
    )
    by_method = df.loc[df["resolved"]].groupby("match_method").size().to_dict()

    summary_lines = [
        f"sample_size\t{total}",
        f"resolved\t{resolved}",
        f"dropped_no_resolve\t{dropped_no_resolve}",
        f"dropped_no_institution\t{dropped_no_inst}",
        f"by_method\t{json.dumps(by_method)}",
    ]
    print("\n=== Run summary ===")
    for line in summary_lines:
        print(" ", line)
    with open(SUMMARY, "w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
