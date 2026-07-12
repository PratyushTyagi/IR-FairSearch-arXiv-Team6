"""Diagnose what OpenAlex is actually returning on title.search requests.

Sends N sequential title.search calls (with mailto), logs the exact URL,
status code, and any rate-limit headers. Helps confirm whether:
- mailto is actually being included (it should be)
- responses are 429 or something else
- there's a Retry-After header value driving the backoff
"""
import json
import os
import sys
import time

import requests

EMAIL = "jain.pav@northeastern.edu"
ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
URL = "https://api.openalex.org/works"
N = 80   # try around the point where the prior run blew up
DELAY = 0.12  # ~8 req/s

def load_title_only(n):
    out = []
    with open(SAMPLE) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("doi"):
                out.append(r)
                if len(out) >= n:
                    break
    return out


def main():
    sess = requests.Session()
    sess.headers.update({"User-Agent": f"kaggle-preproc/1.0 (mailto:{EMAIL})"})

    recs = load_title_only(N)
    print(f"Loaded {len(recs)} title-only records.\n")
    print(f"{'#':>3}  {'status':>6}  {'elapsed':>8}  {'Retry-After':>11}  {'X-RateLimit-Remaining':>22}  url_tail")
    counts = {"200": 0, "429": 0, "other": 0, "exc": 0}
    for i, r in enumerate(recs):
        title = r["title"]
        params = {
            "filter": f"title.search:{title}",
            "per-page": 5,
            "select": "id,doi,title,cited_by_count",
            "mailto": EMAIL,
        }
        t0 = time.monotonic()
        try:
            resp = sess.get(URL, params=params, timeout=20)
            elapsed = time.monotonic() - t0
            ra = resp.headers.get("Retry-After", "")
            rl_remaining = resp.headers.get("X-RateLimit-Remaining",
                                            resp.headers.get("x-ratelimit-remaining", ""))
            sc = resp.status_code
            if sc == 200:
                counts["200"] += 1
            elif sc == 429:
                counts["429"] += 1
            else:
                counts["other"] += 1
            url_tail = resp.url[-80:]
            print(f"{i:>3}  {sc:>6}  {elapsed:>8.2f}  {ra:>11}  {rl_remaining:>22}  {url_tail}")
            if sc != 200:
                # peek at first 200 chars of body
                print(f"      body[:200]: {resp.text[:200]!r}")
        except Exception as e:
            counts["exc"] += 1
            print(f"{i:>3}  EXC     {time.monotonic()-t0:>8.2f}  {'':>11}  {'':>22}  {type(e).__name__}: {e}")
        time.sleep(DELAY)
    print()
    print(f"Summary: {counts}")
    print(f"Effective rate over {N} reqs ~ {N / (sum([0.6]*N) + DELAY*N):.1f} req/s (rough)")


if __name__ == "__main__":
    main()
