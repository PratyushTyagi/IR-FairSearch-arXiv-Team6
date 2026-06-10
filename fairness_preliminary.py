"""
fairness_preliminary.py
Early fairness observations for Slide 7 / report Section: "Early Fairness
Observations." Computes proxy signals of retrieval bias from the baseline
results, without needing the full CWUR tier or Global N/S join yet.

What it computes per retriever (BM25, SPECTER2, MiniLM):

  1. Category skew    -- KL(retrieved-cat-dist || corpus-cat-dist) + top
                         over-represented categories. Signals whether dense
                         retrieval over-concentrates on cs.LG/cs.CL.
  2. Year skew        -- same idea over publication years. Recency bias is a
                         known dense-retriever failure mode.
  3. Retrieval concentration (Gini) -- of all top-K retrievals across the
                         query set, how many distinct papers ever appear,
                         and how concentrated is retrieval on a small
                         "popular" set? Higher Gini = more concentration.
  4. Country / Global-N share (only if topk_affiliations.jsonl exists)
                      -- distribution of retrieved papers' institution
                         countries, and the Global-N vs Global-S split as
                         a first cut at institutional bias.

Run AFTER index_chroma.py. enrich_topk_openalex.py is optional but unlocks
the country analysis -- without it you still get categories / years / Gini,
which are enough material for Slide 7.

Run:  python src/fairness_preliminary.py
"""
import json
from collections import Counter

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

import config as C
from encoders import get_encoder


K = 10

# Rough OECD-ish proxy for Global-N. Document this choice in the report --
# it's an admitted simplification; the final pipeline will use a curated
# World Bank classification instead.
GLOBAL_N = {
    "US", "CA", "GB", "DE", "FR", "JP", "AU", "NL", "CH", "SE", "DK",
    "NO", "FI", "BE", "AT", "IE", "IT", "ES", "NZ", "IL", "KR", "SG",
    "LU", "IS", "PT", "GR", "CZ", "PL", "EE", "SK", "SI", "HU",
}


# ---------- math helpers ----------
def kl_div(p, q, eps=1e-9):
    """KL(P||Q) over the union of keys, with epsilon smoothing."""
    keys = set(p) | set(q)
    return sum((p.get(k, 0) + eps) * np.log((p.get(k, 0) + eps) / (q.get(k, 0) + eps))
               for k in keys)


def gini(counts):
    """Gini coefficient over a list of counts; 0 = uniform, 1 = total concentration."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


def normalize(counter):
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def top_movers(retrieved, corpus, n=5):
    """Keys with the largest positive (retrieved - corpus) share difference."""
    keys = set(retrieved) | set(corpus)
    diffs = [(k, retrieved.get(k, 0) - corpus.get(k, 0)) for k in keys]
    diffs.sort(key=lambda x: -x[1])
    return diffs[:n]


# ---------- retrievers ----------
def topk_bm25(sample, queries):
    corpus = [f'{r["title"]}. {r["abstract"]}'.lower().split() for r in sample]
    ids = [r["id"] for r in sample]
    bm25 = BM25Okapi(corpus)
    out = {}
    for q in queries:
        scores = bm25.get_scores(q["query"].lower().split())
        out[q["qid"]] = [ids[i] for i in np.argsort(scores)[::-1][:K]]
    return out


def topk_dense(queries, model_name):
    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    col = client.get_collection(model_name)
    enc = get_encoder(model_name)
    out = {}
    for q in queries:
        vec = enc.encode([""], [q["query"]])[0].tolist()
        res = col.query(query_embeddings=[vec], n_results=K, include=[])
        out[q["qid"]] = res["ids"][0]
    return out


# ---------- analyses ----------
def analyze_cat_year(topk, sample_by_id, corpus_cat, corpus_year):
    cat_c, yr_c = Counter(), Counter()
    for ids in topk.values():
        for pid in ids:
            r = sample_by_id.get(pid)
            if not r:
                continue
            cat_c[r["primary_category"]] += 1
            yr_c[r["year"]] += 1
    cat_d, yr_d = normalize(cat_c), normalize(yr_c)
    return {
        "cat_kl": kl_div(cat_d, corpus_cat),
        "year_kl": kl_div(yr_d, corpus_year),
        "cat_top": top_movers(cat_d, corpus_cat, 5),
        "year_top": top_movers(yr_d, corpus_year, 5),
    }


def analyze_concentration(topk):
    hit_c = Counter()
    for ids in topk.values():
        for pid in ids:
            hit_c[pid] += 1
    total = sum(hit_c.values())
    return {
        "unique_papers": len(hit_c),
        "total_slots": total,
        "gini": gini(list(hit_c.values())),
        "top10_share": (sum(c for _, c in hit_c.most_common(10)) / total) if total else 0,
    }


def analyze_country(topk, affiliations):
    country_c = Counter()
    n_total, n_resolved = 0, 0
    for ids in topk.values():
        for pid in ids:
            n_total += 1
            affs = affiliations.get(pid)
            if not affs:
                continue
            countries = {a.get("country") for a in affs if a.get("country")}
            if not countries:
                continue
            n_resolved += 1
            for c in countries:
                country_c[c] += 1
    dist = normalize(country_c)
    gn = sum(v for c, v in dist.items() if c in GLOBAL_N)
    return {
        "resolution_rate": n_resolved / n_total if n_total else 0,
        "top10_countries": country_c.most_common(10),
        "global_n_share": gn,
        "global_s_share": 1 - gn,
    }


# ---------- driver ----------
def main():
    sample = [json.loads(l) for l in open(C.SAMPLE_FILE, "r", encoding="utf-8")]
    queries = [json.loads(l) for l in open(C.QUERIES_FILE, "r", encoding="utf-8")]
    sample_by_id = {r["id"]: r for r in sample}

    corpus_cat = normalize(Counter(r["primary_category"] for r in sample))
    corpus_year = normalize(Counter(r["year"] for r in sample))

    retrievers = {
        "BM25": topk_bm25(sample, queries),
        "SPECTER2": topk_dense(queries, "specter2"),
        "MiniLM": topk_dense(queries, "minilm"),
    }

    aff_path = C.DATA_DIR / "topk_affiliations.jsonl"
    affiliations = {}
    if aff_path.exists():
        with open(aff_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                affiliations[row["id"]] = row.get("affiliations")
        print(f"Loaded affiliations for {len(affiliations):,} papers")
    else:
        print(f"(no {aff_path.name} -- skipping country/region analysis)")

    rows = []
    for name, topk in retrievers.items():
        print(f"\n=== {name} ===")
        skew = analyze_cat_year(topk, sample_by_id, corpus_cat, corpus_year)
        conc = analyze_concentration(topk)
        print(f"  Category KL(retrieved||corpus): {skew['cat_kl']:.3f}")
        print(f"  Year KL(retrieved||corpus):     {skew['year_kl']:.3f}")
        print(f"  Top over-represented categories: {skew['cat_top']}")
        print(f"  Top over-represented years:      {skew['year_top']}")
        print(f"  Unique papers ever retrieved: {conc['unique_papers']:,}  "
              f"Gini={conc['gini']:.3f}  top-10 share={conc['top10_share']:.1%}")
        row = {
            "retriever": name,
            "cat_kl": round(skew["cat_kl"], 4),
            "year_kl": round(skew["year_kl"], 4),
            "gini": round(conc["gini"], 4),
            "unique_papers": conc["unique_papers"],
        }
        if affiliations:
            cs = analyze_country(topk, affiliations)
            print(f"  OpenAlex resolution on retrieved slots: {cs['resolution_rate']:.1%}")
            print(f"  Top-10 countries: {cs['top10_countries']}")
            print(f"  Global-N share: {cs['global_n_share']:.1%}   "
                  f"Global-S share: {cs['global_s_share']:.1%}")
            row.update({
                "global_n_share": round(cs["global_n_share"], 4),
                "global_s_share": round(cs["global_s_share"], 4),
                "aff_resolution_rate": round(cs["resolution_rate"], 4),
            })
        rows.append(row)

    out_csv = C.RESULTS_DIR / "fairness_preliminary.csv"
    cols = ["retriever", "cat_kl", "year_kl", "gini", "unique_papers"]
    if affiliations:
        cols += ["global_n_share", "global_s_share", "aff_resolution_rate"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nWrote {out_csv}")

    print("\n--- Slide 7 / report Markdown ---")
    header = ["Retriever", "Cat KL", "Year KL", "Gini"]
    if affiliations:
        header.append("Global-N share")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        cells = [r["retriever"], f"{r['cat_kl']:.3f}",
                 f"{r['year_kl']:.3f}", f"{r['gini']:.3f}"]
        if affiliations:
            cells.append(f"{r['global_n_share']:.1%}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
