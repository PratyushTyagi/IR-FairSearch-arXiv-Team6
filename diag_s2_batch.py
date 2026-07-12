"""Test S2 batch endpoint with arXiv ids and see if it has separate rate limits."""
import json
import time

import requests

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
sess = requests.Session()
sess.headers.update({
    "User-Agent": "kaggle-preproc/1.0 (jain.pav@northeastern.edu)",
    "Content-Type": "application/json",
})

ids = ["ARXIV:0704.0040", "ARXIV:0704.0001", "ARXIV:0704.0252", "ARXIV:0704.1358"]
params = {"fields": "title,authors.affiliations,authors.name,citationCount,externalIds"}

for delay in [0, 5, 30]:
    if delay:
        time.sleep(delay)
    t0 = time.monotonic()
    r = sess.post(BATCH_URL, params=params, json={"ids": ids}, timeout=30)
    print(f"\n[batch after {delay}s wait] status={r.status_code} elapsed={time.monotonic()-t0:.2f}s")
    print(f"  url={r.url}")
    print(f"  body[:500]={r.text[:500]!r}")
    if r.status_code == 200:
        try:
            data = r.json()
            for d in (data or [])[:2]:
                if d:
                    print(f"  - title={d.get('title')!r}")
                    print(f"    citationCount={d.get('citationCount')}  externalIds={d.get('externalIds')}")
                    auths = d.get('authors') or []
                    if auths:
                        print(f"    authors[0]={auths[0]}")
        except Exception as e:
            print(f"  parse: {e}")
