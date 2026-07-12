"""Dedupe the arXiv dump (by arxiv id, normalized DOI, normalized title) and
take a seeded random sample of 50,000 records.

Two-pass over the 5GB file:
  Pass 1: scan to determine which line indices are "kept" (first occurrence
          per fingerprint). Reports dup counts.
  Pass 2: read only the sampled line indices and write them to JSONL+Parquet.

Seed=42 for the sample.
"""
import json
import os
import random
import re
import sys
import unicodedata

import pandas as pd

ROOT = "/Users/pavanijain/Desktop/Kaggle_Preprocessing_Project"
SRC = os.path.join(ROOT, "arxiv-metadata-oai-snapshot.json")
OUT_DIR = os.path.join(ROOT, "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "sample_50k.jsonl")
OUT_PARQUET = os.path.join(OUT_DIR, "sample_50k.parquet")
DEDUP_STATS = os.path.join(OUT_DIR, "dedup_stats.txt")

SAMPLE_SIZE = 50_000
SEED = 42


def norm_doi(doi):
    if not doi:
        return None
    s = str(doi).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = s.strip()
    return s or None


_punct_re = re.compile(r"[^\w\s]+", re.UNICODE)
_ws_re = re.compile(r"\s+")


def norm_title(t):
    if not t:
        return None
    s = unicodedata.normalize("NFKC", str(t)).lower()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s or None


def pass1():
    """Return list of kept line indices and a stats dict."""
    seen_id = {}
    seen_doi = {}
    seen_title = {}
    dups_id = 0
    dups_doi = 0
    dups_title = 0
    kept = []
    total = 0

    with open(SRC, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = obj.get("id")
            doi = norm_doi(obj.get("doi"))
            ttl = norm_title(obj.get("title"))

            is_dup = False
            if aid:
                if aid in seen_id:
                    dups_id += 1
                    is_dup = True
                else:
                    seen_id[aid] = idx
            if doi:
                if doi in seen_doi:
                    dups_doi += 1
                    is_dup = True
                else:
                    seen_doi[doi] = idx
            if ttl:
                if ttl in seen_title:
                    dups_title += 1
                    is_dup = True
                else:
                    seen_title[ttl] = idx

            if not is_dup:
                kept.append(idx)

            if total % 500_000 == 0:
                print(f"  pass1: {total:,} lines scanned, {len(kept):,} kept")

    stats = {
        "total": total,
        "dups_by_id": dups_id,
        "dups_by_doi": dups_doi,
        "dups_by_title": dups_title,
        "unique_after_dedup": len(kept),
    }
    return kept, stats


def pass2_extract(line_indices, out_path):
    """Pass 2: read only the given line indices (sorted) and write JSONL."""
    target = set(line_indices)
    records = []
    with open(SRC, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for idx, line in enumerate(f):
            if idx in target:
                out.write(line if line.endswith("\n") else line + "\n")
                records.append(json.loads(line))
                if len(records) % 10_000 == 0:
                    print(f"  pass2: extracted {len(records):,}/{len(target):,}")
                if len(records) == len(target):
                    break
    return records


def main():
    print("=== PASS 1: dedup scan ===")
    kept, stats = pass1()
    print(f"\nDedup stats:")
    for k, v in stats.items():
        print(f"  {k:25s} {v:>12,}")

    if len(kept) < SAMPLE_SIZE:
        print(f"\nDeduped set has {len(kept):,} < {SAMPLE_SIZE:,}; using all.")
        sample_idx = kept
    else:
        rng = random.Random(SEED)
        sample_idx = rng.sample(kept, SAMPLE_SIZE)
    sample_idx.sort()

    print(f"\nSampled {len(sample_idx):,} records (seed={SEED}).")
    print("=== PASS 2: extract sample ===")
    records = pass2_extract(sample_idx, OUT_JSONL)
    print(f"Wrote {OUT_JSONL} ({len(records):,} rows)")

    # also save as parquet for fast intermediate use
    df = pd.DataFrame(records)
    # versions / authors_parsed are nested — keep as JSON strings for parquet stability
    for col in ("versions", "authors_parsed"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(df):,} rows, {len(df.columns)} cols)")

    with open(DEDUP_STATS, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}\t{v}\n")
        f.write(f"sample_size\t{len(sample_idx)}\n")
        f.write(f"seed\t{SEED}\n")

    print("\n=== First 2 sampled records ===")
    for r in records[:2]:
        print(json.dumps({k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + "...") for k, v in r.items()}, indent=2)[:1500])
        print("---")


if __name__ == "__main__":
    main()
