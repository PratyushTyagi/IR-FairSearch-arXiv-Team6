"""Probe whether OpenAlex supports a cheap arXiv-id filter (instead of title.search).
We have ~$0.0006 budget left today, so this script tries at most 3 cheap probes.
"""
import json
import time
import requests

EMAIL = "jain.pav@northeastern.edu"
URL = "https://api.openalex.org/works"
sess = requests.Session()
sess.headers.update({"User-Agent": f"kaggle-preproc/1.0 (mailto:{EMAIL})"})


def probe(label, params=None, path_suffix=""):
    p = dict(params or {})
    p["mailto"] = EMAIL
    full = URL + path_suffix
    t0 = time.monotonic()
    r = sess.get(full, params=p, timeout=20)
    print(f"\n[{label}] status={r.status_code} elapsed={time.monotonic()-t0:.2f}s")
    print(f"  url={r.url}")
    try:
        j = r.json()
    except Exception:
        print(f"  body[:300]={r.text[:300]!r}")
        return
    if r.status_code != 200:
        print(f"  body={json.dumps(j)[:400]}")
        return
    meta = j.get("meta", {})
    count = meta.get("count")
    cost = meta.get("cost_usd")
    print(f"  meta.count={count}  cost_usd={cost}")
    res = j.get("results") or []
    if res:
        first = res[0]
        print(f"  result0: id={first.get('id')} doi={first.get('doi')} title={first.get('title')!r}")
        # try to find arxiv id info
        ids = first.get("ids") or {}
        print(f"  result0.ids={ids}")
    # we know there is a paper with arxiv id 0704.0040 (Mihai Popa). Inspect its ids field via direct fetch if matched.


# 1. The /works/{external-id} pattern with arxiv prefix — this is often FREE
#    (single record lookup by external id).
r = sess.get("https://api.openalex.org/works/arxiv:0704.0040",
             params={"mailto": EMAIL, "select": "id,doi,title,ids,cited_by_count,authorships"}, timeout=20)
print(f"\n[GET /works/arxiv:0704.0040] status={r.status_code} elapsed=- ")
print(f"  url={r.url}")
if r.status_code == 200:
    j = r.json()
    print(f"  id={j.get('id')}  doi={j.get('doi')}  title={(j.get('title') or '')[:80]!r}")
    print(f"  ids={j.get('ids')}")
    print(f"  authorships[0]={ (j.get('authorships') or [{}])[0] }")
else:
    print(f"  body[:300]={r.text[:300]!r}")
