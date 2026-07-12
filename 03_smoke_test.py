"""Tiny smoke test of the OpenAlex client over the first 50 records of the
50K sample. Verifies: DOI batching works, title-search verification works,
authorships parsing works.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from openalex_client import OpenAlexClient, norm_doi, first_author_first_institution

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
CACHE = os.path.join(ROOT, "cache")

EMAIL = "jain.pav@northeastern.edu"
N = 50


def main():
    client = OpenAlexClient(email=EMAIL, cache_dir=CACHE)
    recs = []
    with open(SAMPLE) as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) >= N:
                break

    doi_recs = [r for r in recs if norm_doi(r.get("doi"))]
    title_recs = [r for r in recs if not norm_doi(r.get("doi"))]
    print(f"Smoke: {len(recs)} records ({len(doi_recs)} DOI, {len(title_recs)} title-only)")

    # batch DOI lookup
    dois = [norm_doi(r["doi"]) for r in doi_recs]
    print(f"Batching {len(dois)} DOIs in one OR-filter request...")
    works = client.lookup_by_dois(dois)
    print(f"  resolved {len(works)}/{len(dois)} DOIs")

    # title lookup for first 5 title-only
    sample_titles = title_recs[:5]
    print(f"Title-search verification on {len(sample_titles)} records:")
    for r in sample_titles:
        w, sim = client.lookup_by_title(r["title"])
        print(f"  arxiv {r['id']}: sim={sim:.1f}, matched={'yes' if w else 'no'}")
        if w:
            iid, iname = first_author_first_institution(w)
            print(f"    -> openalex {w.get('id')}, cited={w.get('cited_by_count')}, inst={iname!r}")

    # spot check an institution from a DOI hit
    for d, w in list(works.items())[:3]:
        iid, iname = first_author_first_institution(w)
        print(f"DOI {d}: openalex {w.get('id')}, cited={w.get('cited_by_count')}, inst={iname!r} ({iid})")


if __name__ == "__main__":
    main()
