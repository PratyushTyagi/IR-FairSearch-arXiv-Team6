"""Step 07 — Demographic mapping: enrich the corpus with an institution-prestige
proxy label ("Privileged" vs "Underrepresented") derived from ONE external
global university ranking (QS / THE / ARWU).

This step is an *additive extension* of the existing pipeline. It does not
overwrite any field produced by 05_rank_and_finalize.py (in particular the
within-sample `top_uni` mean-citation flag is left untouched). It joins an
external ranking onto the corpus using the SAME institution-name canonical key
that 05 already computed, so a paper inherits the global rank of its
first-author institution.

What it computes (Steps 1-5 of the brief):
  Step 1/2  Read ONE ranking table (rank, university[, country]) and record N.
  Step 3    percentile = (1 - R/N) * 100   (percentile WITHIN the ranked list)
  Step 4    apply an "elite" cutoff (default: global rank <= 20) and flag schools.
  Step 5    keep "ranked-but-not-elite" and "unranked" DISTINCT, and (optionally)
            also report the very different "percentile among ALL universities"
            so the two claims are never conflated.

New per-record fields written to the corpus:
  inst_global_rank            int   | None   (R; banded ranks resolved per policy)
  inst_global_rank_banded     bool                (True if R came from a band)
  inst_global_percentile      float | None        ((1-R/N)*100, within ranked list)
  inst_rank_source            str                 (e.g. "QS-2026")
  inst_rank_N                 int                 (denominator used)
  inst_prestige_tier          str                 top20|top50|top100|top200|top500|ranked|unranked
  is_elite_institution        bool                (per chosen cutoff)
  proxy_group                 str                 e.g. "Privileged" | "Underrepresented"
  proxy_group_basis           str                 self-documenting provenance of the label
  inst_percentile_all         float | None        OPTIONAL (only if --n-global given)

IMPORTANT — read demographic_enrichment_README.md. `proxy_group` is a coarse
*institutional-prestige* proxy, NOT a measured demographic attribute of any
author. Top-20 affiliation does not establish that an individual is privileged,
and absence from the ranked list does not establish that anyone is
"underrepresented". The label exists for corpus-level analysis of structural
concentration in publishing; do not use it to make decisions about individuals.

Usage (defaults assume the repo layout used by the other scripts):
  python scripts/07_demographic_enrich.py \
      --ranking data/qs_world_university_rankings_2026.csv \
      --source QS-2026 --n 1501 --elite-rank 20

Outputs (into --out-dir, default = data/):
  university_global_ranking_normalized.csv
  institution_ranking_enriched.csv
  final_enriched_demographics.jsonl
  final_enriched_demographics.parquet        (skipped if no parquet engine)
  demographic_enrichment_summary.txt
  demographic_enrichment_README.md           (copied next to outputs if present)
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# Institution name canonicalization.
#
# We MUST use the exact same normalizer that 05_rank_and_finalize.py used to
# build `canonical_institution_key`, otherwise the join silently misses. By
# default we import it from 05 at runtime; if that file is not reachable we
# fall back to the vendored copy below, which is byte-identical to 05's.
# --------------------------------------------------------------------------- #
_PREFIX_PATTERNS = [
    re.compile(r"^(department|dept\.?|school|faculty|laboratory|lab|college|institute|division|center|centre|group)\s+(of|for)\s+[^,]+,\s*", re.I),
    re.compile(r"^(department|dept\.?|school|faculty|laboratory|lab|division)\s*,\s*", re.I),
]
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")
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


def _vendored_normalize_inst_name(name: str) -> str:
    """Byte-identical copy of 05_rank_and_finalize.normalize_inst_name."""
    if not name:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    prev = None
    while prev != s:
        prev = s
        for pat in _PREFIX_PATTERNS:
            s = pat.sub("", s)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) > 1:
        kept = []
        for i, p in enumerate(parts):
            pl = p.lower().strip()
            if not pl:
                continue
            if pl in _COUNTRY_HINTS:
                continue
            if re.fullmatch(r"[A-Z0-9 ]{2,12}", p) and len(p) <= 6 and i > 0:
                continue
            if re.fullmatch(r"\d[\w\s\-]*", p):
                continue
            kept.append(p)
        if kept:
            s = kept[0]
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
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


def load_normalizer(rank_script_path):
    """Return (normalize_fn, where_from). Prefer the live 05 function."""
    if rank_script_path and os.path.exists(rank_script_path):
        try:
            spec = importlib.util.spec_from_file_location("_rank05", rank_script_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, "normalize_inst_name", None)
            if callable(fn):
                # sanity-probe: ensure it agrees with the vendored copy
                probes = ["University College London (UCL)", "ETH Zurich",
                          "Massachusetts Institute of Technology"]
                if all(fn(p) == _vendored_normalize_inst_name(p) for p in probes):
                    return fn, f"imported from {rank_script_path}"
                return fn, f"imported from {rank_script_path} (NOTE: differs from vendored copy)"
        except Exception as e:  # pragma: no cover
            print(f"  [warn] could not import normalizer from {rank_script_path}: {e}")
    return _vendored_normalize_inst_name, "vendored copy (identical to 05)"


# --------------------------------------------------------------------------- #
# Ranking-table parsing
# --------------------------------------------------------------------------- #
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def strip_trailing_parens(name: str) -> str:
    """Remove trailing '(ACRONYM)' that ranking publishers append but OpenAlex
    display names omit, e.g. 'Stanford University (SU)' -> 'Stanford University'."""
    prev = None
    s = str(name)
    while prev != s:
        prev = s
        s = _TRAILING_PAREN.sub("", s).strip()
    return s


def parse_rank_value(raw, band_policy):
    """Resolve a possibly-banded rank string to a single integer R.

    Handles: '30', '601-610', '601–610', '1001+', '1001-1200', '=42'.
    Returns (R:int|None, banded:bool).
    """
    if raw is None:
        return None, False
    s = str(raw).strip().lstrip("=").strip()
    if not s:
        return None, False
    s = s.replace("–", "-").replace("—", "-")
    # open-ended band like "1001+"
    m = re.fullmatch(r"(\d+)\s*\+", s)
    if m:
        return int(m.group(1)), True
    # closed band like "601-610"
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if band_policy == "midpoint":
            return int(round((lo + hi) / 2)), True
        return lo, True  # "lower" (default)
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1)), False
    return None, False


def tier_from_rank(r):
    if r is None:
        return "unranked"
    for cut, name in [(20, "top20"), (50, "top50"), (100, "top100"),
                      (200, "top200"), (500, "top500")]:
        if r <= cut:
            return name
    return "ranked"


# --------------------------------------------------------------------------- #
# Corpus IO (JSONL is the source of truth; parquet is optional)
# --------------------------------------------------------------------------- #
def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def try_to_parquet(records, path):
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        for col in ("versions", "authors_parsed"):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
        df.to_parquet(path, index=False)
        return True, None
    except Exception as e:
        return False, str(e)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("PROJECT_ROOT", "."),
                    help="Project root containing data/ and scripts/ (default: cwd)")
    ap.add_argument("--ranking", default=None,
                    help="External ranking CSV with columns rank,university[,country]. "
                         "Default: <root>/data/qs_world_university_rankings_2026.csv")
    ap.add_argument("--final", default=None,
                    help="Corpus JSONL. Default: <root>/data/final_enriched.jsonl")
    ap.add_argument("--inst-ranking", default=None,
                    help="Existing institution_ranking.csv. Default: <root>/data/institution_ranking.csv")
    ap.add_argument("--rank-script", default=None,
                    help="Path to 05_rank_and_finalize.py (to import its exact normalizer). "
                         "Default: sibling of this script.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write outputs. Default: <root>/data")
    ap.add_argument("--aliases", default=None,
                    help="Optional JSON {ranking_name: openalex_display_name} for manual fixes.")
    # methodology knobs
    ap.add_argument("--source", default="QS-2026", help="Label recorded in inst_rank_source.")
    ap.add_argument("--n", type=int, default=1501,
                    help="N = size of the ranked list (denominator). QS-2026 published ~1,501.")
    ap.add_argument("--band-policy", choices=["lower", "midpoint"], default="lower",
                    help="How to resolve banded ranks like 601-610 (default: lower bound).")
    ap.add_argument("--elite-mode", choices=["rank", "percentile"], default="rank",
                    help="Define 'elite' by absolute rank (default) or by percentile.")
    ap.add_argument("--elite-rank", type=int, default=20,
                    help="If --elite-mode rank: global rank <= this is elite (default 20).")
    ap.add_argument("--elite-percentile", type=float, default=95.0,
                    help="If --elite-mode percentile: percentile >= this is elite.")
    ap.add_argument("--underrep-includes", choices=["both", "ranked_below_cutoff", "unranked"],
                    default="both",
                    help="Which non-elite rows get the 'underrepresented' proxy label "
                         "(default both: everything not elite).")
    ap.add_argument("--privileged-label", default="Privileged")
    ap.add_argument("--underrep-label", default="Underrepresented")
    ap.add_argument("--n-global", type=int, default=None,
                    help="OPTIONAL: estimated number of ALL universities worldwide "
                         "(~25000-30000). If set, also writes inst_percentile_all so the "
                         "'top X%% of ranked' vs 'top X%% of all' claims stay separate (Step 5).")
    args = ap.parse_args()

    root = args.root
    ranking_path = args.ranking or os.path.join(root, "data", "qs_world_university_rankings_2026.csv")
    final_path = args.final or os.path.join(root, "data", "final_enriched.jsonl")
    inst_rank_path = args.inst_ranking or os.path.join(root, "data", "institution_ranking.csv")
    out_dir = args.out_dir or os.path.join(root, "data")
    rank_script = args.rank_script or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "05_rank_and_finalize.py")
    os.makedirs(out_dir, exist_ok=True)

    normalize, norm_src = load_normalizer(rank_script)
    print(f"Normalizer: {norm_src}")

    aliases = {}
    if args.aliases and os.path.exists(args.aliases):
        with open(args.aliases) as f:
            aliases = json.load(f)
        print(f"Loaded {len(aliases)} manual aliases from {args.aliases}")

    # ---- Steps 1+2: read the ONE ranking, record N, resolve banded ranks ----
    import pandas as pd
    rdf = pd.read_csv(ranking_path)
    cols = {c.lower().strip(): c for c in rdf.columns}
    if "rank" not in cols or "university" not in cols:
        sys.exit(f"Ranking CSV must have 'rank' and 'university' columns; got {list(rdf.columns)}")
    rank_col, uni_col = cols["rank"], cols["university"]

    N = args.n
    max_rank_present = 0
    # canonical_key -> (best_rank, banded, original_name)  (keep best/lowest rank per key)
    key_to_rank = {}
    n_band = 0
    for _, row in rdf.iterrows():
        raw_name = str(row[uni_col])
        R, banded = parse_rank_value(row[rank_col], args.band_policy)
        if R is None:
            continue
        n_band += int(banded)
        max_rank_present = max(max_rank_present, R)
        display = aliases.get(raw_name, strip_trailing_parens(raw_name))
        key = normalize(display)
        if not key:
            continue
        if key not in key_to_rank or R < key_to_rank[key][0]:
            key_to_rank[key] = (R, banded, raw_name)

    if max_rank_present > N:
        print(f"  [warn] a rank ({max_rank_present}) exceeds N={N}. "
              f"Percentiles would go negative; bump --n to your list size.")
    print(f"Ranking '{args.source}': {len(key_to_rank):,} distinct institutions "
          f"loaded (N={N}, banded rows={n_band}, max rank seen={max_rank_present}).")

    # ---- Step 3: percentile within the ranked list ----
    def pct_ranked(R):
        return round((1.0 - R / N) * 100.0, 2)

    def pct_all(R):
        if not args.n_global:
            return None
        return round((1.0 - R / args.n_global) * 100.0, 2)

    # ---- Step 4: elite decision ----
    def is_elite(R, pr):
        if R is None:
            return False
        if args.elite_mode == "rank":
            return R <= args.elite_rank
        return pr >= args.elite_percentile

    def proxy_label(elite, R):
        if elite:
            return args.privileged_label
        # non-elite: decide per --underrep-includes
        if args.underrep_includes == "both":
            return args.underrep_label
        if args.underrep_includes == "unranked":
            return args.underrep_label if R is None else ""
        # ranked_below_cutoff
        return args.underrep_label if R is not None else ""

    cutoff_desc = (f"rank<={args.elite_rank}" if args.elite_mode == "rank"
                   else f"percentile>={args.elite_percentile}")
    basis = f"{args.source} {cutoff_desc} (institutional-prestige proxy)"

    # ---- Build the normalized lookup artifact ----
    lookup_rows = []
    for key, (R, banded, raw) in sorted(key_to_rank.items(), key=lambda x: x[1][0]):
        pr = pct_ranked(R)
        lookup_rows.append({
            "canonical_institution_key": key,
            "ranking_source_name": raw,
            "inst_global_rank": R,
            "inst_global_rank_banded": banded,
            "inst_global_percentile": pr,
            "inst_prestige_tier": tier_from_rank(R),
            "is_elite_institution": is_elite(R, pr),
            "inst_percentile_all": pct_all(R),
        })
    pd.DataFrame(lookup_rows).to_csv(
        os.path.join(out_dir, "university_global_ranking_normalized.csv"), index=False)
    print(f"Wrote university_global_ranking_normalized.csv ({len(lookup_rows)} rows)")

    # ---- Enrich the existing institution-level ranking csv ----
    if os.path.exists(inst_rank_path):
        idf = pd.read_csv(inst_rank_path)

        def lk(key, field, default=None):
            v = key_to_rank.get(key)
            if v is None:
                return default
            R, banded, raw = v
            pr = pct_ranked(R)
            return {
                "inst_global_rank": R,
                "inst_global_rank_banded": banded,
                "inst_global_percentile": pr,
                "inst_prestige_tier": tier_from_rank(R),
                "is_elite_institution": is_elite(R, pr),
                "inst_percentile_all": pct_all(R),
            }[field]

        for field, default in [("inst_global_rank", pd.NA),
                               ("inst_global_rank_banded", False),
                               ("inst_global_percentile", pd.NA),
                               ("inst_prestige_tier", "unranked"),
                               ("is_elite_institution", False),
                               ("inst_percentile_all", pd.NA)]:
            idf[field] = idf["canonical_institution_key"].map(
                lambda k: lk(k, field, default))
        idf["inst_rank_source"] = args.source
        idf["inst_rank_N"] = N
        idf.to_csv(os.path.join(out_dir, "institution_ranking_enriched.csv"), index=False)
        print(f"Wrote institution_ranking_enriched.csv ({len(idf)} rows)")
    else:
        print(f"  [skip] {inst_rank_path} not found; institution_ranking_enriched.csv not written")

    # ---- Enrich every corpus record (Steps 3-5 applied per paper) ----
    out_jsonl = os.path.join(out_dir, "final_enriched_demographics.jsonl")
    tier_counts = Counter()
    proxy_counts = Counter()
    elite_insts = Counter()
    n_records = 0
    n_with_inst = 0
    enriched_records = []
    with open(out_jsonl, "w") as out:
        for rec in iter_jsonl(final_path):
            n_records += 1
            key = rec.get("canonical_institution_key") or ""
            entry = key_to_rank.get(key) if key else None
            if key:
                n_with_inst += 1
            if entry is not None:
                R, banded, raw = entry
                pr = pct_ranked(R)
                tier = tier_from_rank(R)
                elite = is_elite(R, pr)
                rec["inst_global_rank"] = R
                rec["inst_global_rank_banded"] = bool(banded)
                rec["inst_global_percentile"] = pr
                rec["inst_prestige_tier"] = tier
                rec["is_elite_institution"] = bool(elite)
                rec["inst_percentile_all"] = pct_all(R)
                if elite:
                    elite_insts[rec.get("canonical_institution_name") or key] += 1
            else:
                R = None
                pr = None
                tier = "unranked"
                elite = False
                rec["inst_global_rank"] = None
                rec["inst_global_rank_banded"] = False
                rec["inst_global_percentile"] = None
                rec["inst_prestige_tier"] = tier
                rec["is_elite_institution"] = False
                rec["inst_percentile_all"] = None
            rec["inst_rank_source"] = args.source
            rec["inst_rank_N"] = N
            label = proxy_label(elite, R)
            rec["proxy_group"] = label
            rec["proxy_group_basis"] = basis
            tier_counts[tier] += 1
            proxy_counts[label or "(unlabeled)"] += 1
            out.write(json.dumps(rec) + "\n")
            enriched_records.append(rec)
    print(f"Wrote {out_jsonl} ({n_records:,} records)")

    # optional parquet (their env has pyarrow; here it may not)
    ok, err = try_to_parquet(enriched_records,
                             os.path.join(out_dir, "final_enriched_demographics.parquet"))
    if ok:
        print("Wrote final_enriched_demographics.parquet")
    else:
        print(f"  [skip] parquet not written (no engine?): {err}")

    # ---- Summary (Step 5 distinctions made explicit) ----
    summary_path = os.path.join(out_dir, "demographic_enrichment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("# Demographic mapping — institution-prestige proxy enrichment\n")
        f.write(f"ranking_source\t{args.source}\n")
        f.write(f"N_ranked_list\t{N}\n")
        f.write(f"normalizer\t{norm_src}\n")
        f.write(f"band_policy\t{args.band_policy}\n")
        f.write(f"elite_definition\t{cutoff_desc}\n")
        f.write(f"underrep_includes\t{args.underrep_includes}\n")
        f.write(f"corpus_records\t{n_records}\n")
        f.write(f"records_with_institution\t{n_with_inst}\n")
        f.write(f"distinct_ranked_institutions_loaded\t{len(key_to_rank)}\n")
        f.write("\n# prestige tier distribution (records)\n")
        for t in ["top20", "top50", "top100", "top200", "top500", "ranked", "unranked"]:
            f.write(f"  {t}\t{tier_counts.get(t, 0)}\n")
        f.write("\n# proxy_group distribution (records)\n")
        for lbl, c in proxy_counts.most_common():
            f.write(f"  {lbl}\t{c}\n")
        f.write("\n# elite institutions present in corpus (by record count)\n")
        for nm, c in elite_insts.most_common():
            f.write(f"  {nm}\t{c}\n")
        f.write("\n# Step-5 denominator caveat (DO NOT CONFLATE)\n")
        f.write("'inst_global_percentile' is percentile WITHIN the ranked list only.\n")
        f.write(f"A rank-30 school is ~{round((1-30/N)*100,1)}th percentile of the ~{N} RANKED schools,\n")
        f.write("but merely appearing on the list already places a school in roughly the\n")
        f.write("global top ~5-6%% of the ~25,000-30,000 universities worldwide.\n")
        if args.n_global:
            f.write(f"'inst_percentile_all' uses N_global={args.n_global} for the all-universities view.\n")
        else:
            f.write("Pass --n-global to also emit 'inst_percentile_all' for the all-universities view.\n")
        f.write("\n# What this label is / is NOT\n")
        f.write("proxy_group is an INSTITUTIONAL-PRESTIGE proxy, not a measured demographic\n")
        f.write("attribute. It must not be used to make decisions about individuals.\n")
    print(f"Wrote {summary_path}")

    # append to the pipeline's run_summary.txt if it exists (mirrors 05)
    run_summary = os.path.join(root, "cache", "run_summary.txt")
    if os.path.exists(os.path.dirname(run_summary)):
        with open(run_summary, "a") as f:
            f.write("\n# After 07_demographic_enrich\n")
            f.write(f"demographic_ranking_source\t{args.source}\n")
            f.write(f"demographic_N\t{N}\n")
            f.write(f"proxy_group_counts\t{json.dumps(dict(proxy_counts))}\n")
            f.write(f"prestige_tier_counts\t{json.dumps(dict(tier_counts))}\n")

    # ---- stdout preview ----
    print("\n=== prestige tier distribution (records) ===")
    for t in ["top20", "top50", "top100", "top200", "top500", "ranked", "unranked"]:
        print(f"  {t:9s} {tier_counts.get(t, 0):>7,}")
    print("\n=== proxy_group distribution (records) ===")
    for lbl, c in proxy_counts.most_common():
        print(f"  {lbl:18s} {c:>7,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
