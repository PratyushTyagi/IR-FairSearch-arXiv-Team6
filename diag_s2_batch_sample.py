"""Send 5 batches of 500 arXiv IDs each (= 2500 records) to S2 batch endpoint.
Measure: throughput, resolution rate, affiliation availability, citation availability.
"""
import json
import os
import sys
import time

import requests

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

sess = requests.Session()
sess.headers.update({
    "User-Agent": "kaggle-preproc/1.0 (jain.pav@northeastern.edu)",
    "Content-Type": "application/json",
})


def load_title_only_ids(n):
    out = []
    with open(SAMPLE) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("doi"):
                out.append(r["id"])
                if len(out) >= n:
                    break
    return out


def main():
    BATCH = 500
    TOTAL = 2500
    arxiv_ids = load_title_only_ids(TOTAL)
    print(f"Loaded {len(arxiv_ids)} title-only arxiv ids")

    params = {"fields": "title,authors.affiliations,authors.name,citationCount,externalIds"}
    n_resolved = 0
    n_404 = 0
    n_with_affil = 0
    n_with_cite = 0
    n_with_first_author_affil = 0
    n_first_author_inst_strings_sample = []
    t0 = time.monotonic()
    for i in range(0, len(arxiv_ids), BATCH):
        batch = arxiv_ids[i : i + BATCH]
        body = {"ids": [f"ARXIV:{a}" for a in batch]}
        t_call = time.monotonic()
        r = sess.post(URL, params=params, json=body, timeout=60)
        elapsed = time.monotonic() - t_call
        if r.status_code != 200:
            print(f"  batch {i//BATCH}: HTTP {r.status_code}: {r.text[:200]}")
            continue
        data = r.json()
        for d in data:
            if d is None:
                n_404 += 1
                continue
            n_resolved += 1
            auths = d.get("authors") or []
            if d.get("citationCount") is not None:
                n_with_cite += 1
            if auths:
                if any((a.get("affiliations") or []) for a in auths):
                    n_with_affil += 1
                first = auths[0]
                affs = first.get("affiliations") or []
                if affs:
                    n_with_first_author_affil += 1
                    if len(n_first_author_inst_strings_sample) < 8:
                        n_first_author_inst_strings_sample.append((d.get("title", "")[:50], affs[0]))
        print(f"  batch {i//BATCH+1}/{(len(arxiv_ids)+BATCH-1)//BATCH}: {len(batch)} ids -> {elapsed:.2f}s, resolved so far {n_resolved}, w/firstauth_affil {n_with_first_author_affil}")
    total = time.monotonic() - t0
    print(f"\nTotal: {len(arxiv_ids)} ids in {total:.1f}s ({len(arxiv_ids)/total:.0f} rec/s)")
    print(f"  resolved: {n_resolved}/{len(arxiv_ids)}  ({100*n_resolved/len(arxiv_ids):.1f}%)")
    print(f"  not in S2 (null in response): {n_404}")
    print(f"  with non-empty citationCount: {n_with_cite}")
    print(f"  with ANY author having affiliations: {n_with_affil}/{n_resolved}  ({100*n_with_affil/max(1,n_resolved):.1f}%)")
    print(f"  with FIRST author having affiliations: {n_with_first_author_affil}/{n_resolved}  ({100*n_with_first_author_affil/max(1,n_resolved):.1f}%)")
    print(f"\nSample first-author affiliations:")
    for t, a in n_first_author_inst_strings_sample:
        print(f"  title={t!r}\n    aff={a!r}")
    # Extrapolate
    print(f"\nFull run estimate for 28,533 title-only papers @ {len(arxiv_ids)/total:.0f} rec/s:")
    print(f"  {28533 / (len(arxiv_ids)/total):.1f}s = {28533 / (len(arxiv_ids)/total) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
