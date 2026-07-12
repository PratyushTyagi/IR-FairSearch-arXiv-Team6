"""Probe whether ANY OpenAlex endpoint is still working under our current budget.

Tries a few endpoint shapes and logs status + response body.
"""
import time

import requests

EMAIL = "jain.pav@northeastern.edu"
URL = "https://api.openalex.org/works"

sess = requests.Session()
sess.headers.update({"User-Agent": f"kaggle-preproc/1.0 (mailto:{EMAIL})"})


def probe(label, params):
    params = dict(params)
    params["mailto"] = EMAIL
    t0 = time.monotonic()
    r = sess.get(URL, params=params, timeout=20)
    elapsed = time.monotonic() - t0
    print(f"[{label}]  status={r.status_code}  elapsed={elapsed:.2f}s")
    print(f"  url={r.url}")
    print(f"  headers: x-rl-rem={r.headers.get('x-ratelimit-remaining','')}  retry-after={r.headers.get('Retry-After','')}")
    print(f"  body[:300]={r.text[:300]!r}\n")


# 1. trivial /works (no filter) – cheapest browse
probe("plain works (per_page=1)", {"per-page": 1})

# 2. DOI filter (batched OR) – the supposedly cheap path
probe("DOI filter (1 DOI)", {"filter": "doi:10.1103/PhysRevB.75.174437", "per-page": 5})

# 3. DOI filter (batched 10 OR)
sample_dois = [
    "10.1103/PhysRevB.75.174437",
    "10.1103/PhysRevD.76.013009",
    "10.1209/0295-5075/79/27002",
    "10.1086/587860",
    "10.1109/tit.2008.2011437",
]
probe("DOI filter (batch 5)", {"filter": "doi:" + "|".join(sample_dois), "per-page": 50})

# 4. title.search
probe("title.search", {"filter": "title.search:Spin chain magnetism", "per-page": 5})

# 5. fetch a single work by openalex id
probe("get one work by id", {})
r = sess.get("https://api.openalex.org/works/W2002185777", params={"mailto": EMAIL}, timeout=20)
print(f"[direct /works/W...]  status={r.status_code}  body[:300]={r.text[:300]!r}")
