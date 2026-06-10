"""
enrich_affiliations_openalex.py   (NEXT PHASE — not needed for baseline P/R)

WHY THIS EXISTS
The Cornell arXiv Kaggle snapshot has NO affiliation field; `authors_parsed`
contains author names only. Your fairness metrics (EED, demographic parity,
citation-share) all key off institution, so you must enrich affiliations from
an external source. OpenAlex is free, needs no API key, returns institutions
already mapped to ROR ids, and lets you look papers up by arXiv id or DOI.

Polite-pool etiquette: put your email in MAILTO (OpenAlex prioritizes
identified traffic and it stays free).

This script is network-dependent and rate-limited; run it once over the 50K
sample and cache the result. After enrichment, map ROR -> institution and join
a CWUR top-50 list to produce the protected attribute (tier) and region.

Run:  python src/enrich_affiliations_openalex.py
"""
import json
import time

import requests   # pip install requests

import config as C

MAILTO = "your-email@northeastern.edu"   # <-- set this
OPENALEX = "https://api.openalex.org/works/arxiv:{arxiv_id}"


def load_sample():
    with open(C.SAMPLE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fetch_affiliations(arxiv_id):
    """Return list of {author, institution, ror, country} for one paper."""
    url = OPENALEX.format(arxiv_id=arxiv_id) + f"?mailto={MAILTO}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None  # not indexed in OpenAlex
    data = r.json()
    out = []
    for a in data.get("authorships", []):
        for inst in (a.get("institutions") or [{}]):
            out.append({
                "author": (a.get("author") or {}).get("display_name"),
                "institution": inst.get("display_name"),
                "ror": inst.get("ror"),
                "country": inst.get("country_code"),
            })
    return out


def main():
    sample = load_sample()
    out_path = C.DATA_DIR / "affiliations.jsonl"
    resolved = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(sample):
            affs = fetch_affiliations(rec["id"])
            if affs:
                resolved += 1
            f.write(json.dumps({"id": rec["id"], "affiliations": affs}) + "\n")
            time.sleep(0.1)  # be polite; ~10 req/s
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(sample):,}  resolved={resolved:,}")
    print(f"Done. Resolved {resolved:,}/{len(sample):,} "
          f"({resolved/len(sample):.1%}) -> {out_path}")
    print("This % is the number for the Dataset section / Slide 7 [verify] tag.")


if __name__ == "__main__":
    main()
