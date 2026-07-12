"""Smoke-test the S2 client on 30 title-only records: rate, 429 count, hit rate."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from s2_client import SemanticScholarClient, first_author_first_affiliation, norm_text

ROOT = "/Users/pavanijain/Desktop/Kaggle_Preprocessing_Project"
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
CACHE = os.path.join(ROOT, "cache", "s2")
N = 30


def main():
    client = SemanticScholarClient(cache_dir=CACHE, rate_per_sec=1.0)
    title_only = []
    with open(SAMPLE) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("doi"):
                title_only.append(r)
                if len(title_only) >= N:
                    break

    print(f"Testing {len(title_only)} title-only records at 1 req/s...")
    t0 = time.monotonic()
    stats = {"resolved": 0, "unresolved_no_hit": 0, "unresolved_retryable": 0, "no_inst": 0}
    for r in title_only:
        paper, sim, status = client.search_title(r["title"])
        stats[status] += 1
        if status == "resolved":
            aid, aname = first_author_first_affiliation(paper)
            inst = aname or "<none>"
            if not aid:
                stats["no_inst"] += 1
            print(f"  ✓ {r['id']:>10}  sim={sim:>5.1f}  cite={paper.get('citationCount')}  inst={inst[:50]!r}")
        else:
            print(f"  ✗ {r['id']:>10}  status={status}  sim={sim:.1f}")
    total = time.monotonic() - t0
    print(f"\nDone in {total:.1f}s  ({len(title_only)/total:.2f} rec/s effective)")
    print(f"Stats: {stats}")
    if len(title_only) > 0:
        n_resolved = stats["resolved"]
        print(f"Hit rate: {100*n_resolved/len(title_only):.1f}%")
        print(f"Of resolved, {stats['no_inst']}/{n_resolved} had no first-author affiliation")
    # extrapolate
    rem = 28461 - 72
    eta_s = rem / max(1e-3, len(title_only)/total)
    print(f"\nETA for {rem} title-only records at this rate: {eta_s/3600:.1f} hours")


if __name__ == "__main__":
    main()
