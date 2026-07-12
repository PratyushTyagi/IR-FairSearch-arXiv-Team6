"""Dig into actual S2 responses to understand affiliation availability + rate limits."""
import json
import time

import requests

URL = "https://api.semanticscholar.org/graph/v1/paper/search"
sess = requests.Session()
sess.headers.update({"User-Agent": "kaggle-preproc/1.0 (jain.pav@northeastern.edu)"})


def probe(label, **params):
    t0 = time.monotonic()
    r = sess.get(URL, params=params, timeout=20)
    print(f"\n[{label}] status={r.status_code} elapsed={time.monotonic()-t0:.2f}s")
    print(f"  url={r.url}")
    print(f"  retry-after={r.headers.get('Retry-After','')}")
    print(f"  body[:400]={r.text[:400]!r}")
    try:
        j = r.json()
        if r.status_code == 200 and j.get("data"):
            p = j["data"][0]
            print(f"  ---")
            print(f"  paperId={p.get('paperId')}")
            print(f"  title={p.get('title')!r}")
            print(f"  citationCount={p.get('citationCount')}")
            print(f"  externalIds={p.get('externalIds')}")
            print(f"  authors[0]={(p.get('authors') or [{}])[0]}")
    except Exception as e:
        print(f"  json parse: {e}")


# space the calls
probe("search-no-affil",
      query="Multilinear function series in conditionally free probability",
      limit=3, fields="title,authors,citationCount,externalIds")
time.sleep(2)
probe("search-with-affil",
      query="Multilinear function series in conditionally free probability",
      limit=3, fields="title,authors.affiliations,authors.name,citationCount,externalIds")
time.sleep(2)
# Try arxiv-id lookup directly (S2 supports arxiv: prefix)
r = sess.get("https://api.semanticscholar.org/graph/v1/paper/arXiv:0704.0001",
             params={"fields": "title,authors.affiliations,authors.name,citationCount,externalIds"},
             timeout=20)
print(f"\n[GET /paper/arXiv:0704.0001] status={r.status_code}")
print(f"  body[:500]={r.text[:500]!r}")
time.sleep(2)
# Try another arxiv id
r = sess.get("https://api.semanticscholar.org/graph/v1/paper/arXiv:0704.0040",
             params={"fields": "title,authors.affiliations,authors.name,citationCount,externalIds"},
             timeout=20)
print(f"\n[GET /paper/arXiv:0704.0040] status={r.status_code}")
print(f"  body[:500]={r.text[:500]!r}")
