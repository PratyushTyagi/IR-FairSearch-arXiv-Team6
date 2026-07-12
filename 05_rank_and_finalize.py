"""Rank institutions and emit the FINAL enriched dataset.

Inputs:
  data/sample_50k.jsonl              (original arXiv records)
  data/enrichment_results.parquet    (per-record OpenAlex + S2 results)

Outputs:
  data/institution_ranking.csv       (top-50 + everyone else)
  data/final_enriched.jsonl          (drop-in for original JSON; resolved papers only)
  data/final_enriched.parquet
  cache/run_summary.txt              (extended)

The institution-grouping step normalizes names ACROSS sources:
  - OpenAlex: stable id + display_name
  - Semantic Scholar: free-text affiliation string (only ~3% populated)
A canonical key per row is built by normalizing the name (stripping
"department/school/lab of …, " prefixes and ", city/country" suffixes,
lowercasing, dropping punctuation). Within each canonical bucket the
OpenAlex display_name wins as the label; if no OpenAlex member, the most
common S2 string wins.

The mean-citation ranking mixes OpenAlex `cited_by_count` and S2
`citationCount` — these are NOT computed identically. We flag this in the
summary.
"""
import json
import os
import re
import unicodedata
from collections import Counter

import pandas as pd

ROOT = "/Users/pavanijain/Desktop/Kaggle_Preprocessing_Project"
SAMPLE = os.path.join(ROOT, "data", "sample_50k.jsonl")
ENRICH = os.path.join(ROOT, "data", "enrichment_results.parquet")
RANK_CSV = os.path.join(ROOT, "data", "institution_ranking.csv")
FINAL_JSONL = os.path.join(ROOT, "data", "final_enriched.jsonl")
FINAL_PARQUET = os.path.join(ROOT, "data", "final_enriched.parquet")
SUMMARY = os.path.join(ROOT, "cache", "run_summary.txt")

MIN_PAPERS_PER_INST = 5
TOP_N_INSTS = 50

# Prefix patterns: leading "Department of X, " "School of X, " etc.
_PREFIX_PATTERNS = [
    re.compile(r"^(department|dept\.?|school|faculty|laboratory|lab|college|institute|division|center|centre|group)\s+(of|for)\s+[^,]+,\s*", re.I),
    re.compile(r"^(department|dept\.?|school|faculty|laboratory|lab|division)\s*,\s*", re.I),
]
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")

# Common country/region suffixes that should be stripped (tail tokens)
_COUNTRY_HINTS = {
    "usa", "us", "united states", "united states of america",
    "uk", "united kingdom", "england", "scotland",
    "canada", "germany", "france", "italy", "spain", "japan", "china",
    "india", "australia", "russia", "switzerland", "netherlands",
    "belgium", "sweden", "denmark", "finland", "norway", "austria",
    "ireland", "south korea", "korea", "brazil", "mexico", "israel",
    "poland", "czech republic", "greece", "portugal", "turkey",
    "argentina", "chile", "south africa", "new zealand", "singapore",
    "taiwan", "hong kong",
}


def normalize_inst_name(name: str) -> str:
    """Return canonical key for grouping. Empty string for None/empty."""
    if not name:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # strip leading "Department of ..., " etc.
    prev = None
    while prev != s:
        prev = s
        for pat in _PREFIX_PATTERNS:
            s = pat.sub("", s)
    # split by commas; drop trailing comma-tail tokens that look like
    # city/state/postcode/country (heuristic)
    parts = [p.strip() for p in s.split(",")]
    # If we have multiple comma-separated parts, the first non-empty one
    # is most likely the institution; drop tail tokens that match country list
    # or postal codes / short codes.
    if len(parts) > 1:
        kept = []
        for i, p in enumerate(parts):
            pl = p.lower().strip()
            if not pl:
                continue
            # Drop if it's a known country hint or looks like a postal/postcode
            if pl in _COUNTRY_HINTS:
                continue
            if re.fullmatch(r"[A-Z0-9 ]{2,12}", p) and len(p) <= 6 and i > 0:
                # short code like "MA 02139", "CA", etc.
                continue
            if re.fullmatch(r"\d[\w\s\-]*", p):
                # numeric postcode
                continue
            kept.append(p)
        if kept:
            # The first remaining part is usually the institution
            s = kept[0]
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    # Collapse common abbrev/synonyms
    syn_map = [
        (r"\buniv\.?\b", "university"),
        (r"\binst\.?\b", "institute"),
        (r"\btechnol\.?\b", "technology"),
        (r"\blab\.?\b", "laboratory"),
        (r"\bnatl\.?\b", "national"),
        (r"\bintl\.?\b", "international"),
        (r"\beur\.?\b", "european"),
    ]
    for pat, rep in syn_map:
        s = re.sub(pat, rep, s)
    s = _WS.sub(" ", s).strip()
    return s


def load_originals():
    out = {}
    with open(SAMPLE) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main():
    print("Loading enrichment results...")
    df = pd.read_parquet(ENRICH)
    print(f"  {len(df):,} rows; resolved={int(df['resolved'].sum()):,}")

    resolved = df[df["resolved"]].copy()
    print(f"Resolved papers: {len(resolved):,}")
    print(f"  by source: {resolved['source'].value_counts().to_dict()}")
    print(f"  by match_method: {resolved['match_method'].value_counts().to_dict()}")

    # ---- Institution canonicalization ----
    resolved["canonical_institution_key"] = (
        resolved["first_author_institution_name"].fillna("").map(normalize_inst_name)
    )
    has_inst = resolved[resolved["canonical_institution_key"] != ""].copy()
    print(f"  papers with non-empty canonical key: {len(has_inst):,}")
    print(f"    from openalex: {int((has_inst['source']=='openalex').sum()):,}")
    print(f"    from s2:       {int((has_inst['source']=='semantic_scholar').sum()):,}")

    # Display label: prefer OpenAlex display_name within each key
    label_per_key = {}
    for key, group in has_inst.groupby("canonical_institution_key"):
        oa = group[group["source"] == "openalex"]
        if len(oa) > 0:
            label = Counter(oa["first_author_institution_name"].dropna().tolist()).most_common(1)
            if label:
                label_per_key[key] = label[0][0]
                continue
        s2 = group[group["source"] == "semantic_scholar"]
        label = Counter(s2["first_author_institution_name"].dropna().tolist()).most_common(1)
        label_per_key[key] = label[0][0] if label else key
    has_inst["canonical_institution_name"] = has_inst["canonical_institution_key"].map(label_per_key)
    # back-propagate to the resolved df
    resolved["canonical_institution_name"] = resolved["canonical_institution_key"].map(label_per_key).fillna("")

    # ---- Rank ----
    inst_grp = (
        has_inst.groupby("canonical_institution_key")
        .agg(
            canonical_institution_name=("canonical_institution_name", "first"),
            n_papers=("arxiv_id", "count"),
            mean_citations=("cited_by_count", "mean"),
            n_papers_from_openalex=("source", lambda s: int((s == "openalex").sum())),
            n_papers_from_s2=("source", lambda s: int((s == "semantic_scholar").sum())),
        )
        .reset_index()
    )
    print(f"  unique canonical institutions: {len(inst_grp):,}")

    eligible = inst_grp[inst_grp["n_papers"] >= MIN_PAPERS_PER_INST].copy()
    print(f"  with >= {MIN_PAPERS_PER_INST} papers: {len(eligible):,}")
    eligible = eligible.sort_values("mean_citations", ascending=False).reset_index(drop=True)
    eligible["rank"] = eligible.index + 1
    top_keys = set(eligible.head(TOP_N_INSTS)["canonical_institution_key"].tolist())
    eligible["is_top_uni"] = eligible["canonical_institution_key"].isin(top_keys)

    ineligible = inst_grp[inst_grp["n_papers"] < MIN_PAPERS_PER_INST].copy()
    ineligible["rank"] = pd.NA
    ineligible["is_top_uni"] = False

    full_rank = pd.concat([eligible, ineligible], ignore_index=True)
    full_rank = full_rank[
        [
            "canonical_institution_key", "canonical_institution_name",
            "n_papers", "mean_citations", "rank", "is_top_uni",
            "n_papers_from_openalex", "n_papers_from_s2",
        ]
    ]
    full_rank.to_csv(RANK_CSV, index=False)
    print(f"\nWrote {RANK_CSV}")

    # ---- Build final dataset ----
    print("\nBuilding final dataset...")
    originals = load_originals()
    final_records = []
    for _, erow in resolved.iterrows():
        aid = erow["arxiv_id"]
        orig = originals.get(aid)
        if orig is None:
            continue
        out = dict(orig)
        out["openalex_id"] = erow.get("openalex_id") or None
        out["s2_paper_id"] = erow.get("s2_paper_id") or None
        out["matched_doi"] = erow.get("matched_doi") or None
        out["match_method"] = erow.get("match_method") or None
        out["first_author_institution_id"] = erow.get("first_author_institution_id") or None
        out["first_author_institution_name"] = erow.get("first_author_institution_name") or None
        out["canonical_institution_key"] = erow.get("canonical_institution_key") or None
        out["canonical_institution_name"] = erow.get("canonical_institution_name") or None
        cite = erow.get("cited_by_count")
        out["cited_by_count"] = int(cite) if pd.notna(cite) else None
        out["citation_field"] = erow.get("citation_field") or None
        out["source"] = erow.get("source")
        out["top_uni"] = bool(out["canonical_institution_key"] in top_keys) if out["canonical_institution_key"] else False
        final_records.append(out)

    n_top = sum(1 for r in final_records if r["top_uni"])
    print(f"  {len(final_records):,} resolved papers in final dataset")
    print(f"  top_uni=True: {n_top:,}")

    # JSONL (drop-in for original)
    with open(FINAL_JSONL, "w") as f:
        for r in final_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {FINAL_JSONL}")

    # Parquet
    df_final = pd.DataFrame(final_records)
    for col in ("versions", "authors_parsed"):
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(
                lambda x: json.dumps(x) if x is not None else None
            )
    df_final.to_parquet(FINAL_PARQUET, index=False)
    print(f"Wrote {FINAL_PARQUET}")

    # ---- Extend run summary ----
    by_source = df_final["source"].value_counts().to_dict()
    by_method = df_final["match_method"].value_counts().to_dict()
    with open(SUMMARY, "a") as f:
        f.write("\n# After ranking + final dataset\n")
        f.write(f"final_dataset_rows\t{len(final_records)}\n")
        f.write(f"top_uni_true_count\t{n_top}\n")
        f.write(f"unique_canonical_institutions\t{len(inst_grp)}\n")
        f.write(f"institutions_ge_{MIN_PAPERS_PER_INST}_papers\t{len(eligible)}\n")
        f.write(f"by_source\t{json.dumps(by_source)}\n")
        f.write(f"by_match_method\t{json.dumps(by_method)}\n")
        f.write("\n# IMPORTANT: dual-source citation mix\n")
        f.write("OpenAlex.cited_by_count and S2.citationCount are NOT computed identically.\n")
        f.write("DOI papers were resolved via OpenAlex (cited_by_count).\n")
        f.write("Title-only papers were resolved via Semantic Scholar batch by ARXIV: id (citationCount).\n")
        f.write("The institution-mean-citations ranking aggregates both counts as if comparable.\n")
        f.write("\n# Top 50 institutions\n")
        for _, row in eligible.head(TOP_N_INSTS).iterrows():
            f.write(
                f"  {int(row['rank']):>2}. {row['canonical_institution_name']}"
                f"\tn={int(row['n_papers'])}"
                f"\tmean_cites={row['mean_citations']:.1f}"
                f"\toa={int(row['n_papers_from_openalex'])}"
                f"\ts2={int(row['n_papers_from_s2'])}\n"
            )

    # ---- Stdout previews ----
    print(f"\n=== TOP {TOP_N_INSTS} INSTITUTIONS ===")
    for _, row in eligible.head(TOP_N_INSTS).iterrows():
        print(
            f"  {int(row['rank']):>2}. {row['canonical_institution_name']}  "
            f"(n={int(row['n_papers'])}, mean_cites={row['mean_citations']:.1f}, "
            f"oa={int(row['n_papers_from_openalex'])}, s2={int(row['n_papers_from_s2'])})"
        )

    print(f"\n=== Final dataset preview (5 rows, key fields) ===")
    preview = df_final.head(5)[
        [
            "id", "title", "doi", "openalex_id", "s2_paper_id",
            "match_method", "source", "canonical_institution_name",
            "cited_by_count", "citation_field", "top_uni",
        ]
    ]
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(preview.to_string())

    print(f"\nFINAL FILES:")
    print(f"  JSONL:    {FINAL_JSONL}  ({len(final_records):,} rows)")
    print(f"  Parquet:  {FINAL_PARQUET}")
    print(f"  Ranking:  {RANK_CSV}")
    print(f"  top_uni=True: {n_top:,}")


if __name__ == "__main__":
    main()
