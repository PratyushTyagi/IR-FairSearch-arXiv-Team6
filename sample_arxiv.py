"""
sample_arxiv.py
Draw a stratified random sample (default 50K) from the Cornell arXiv
Kaggle snapshot, stratified by (primary category x year).

The snapshot is ~4 GB of newline-delimited JSON, so we never load it all
into memory. Two streaming passes:
  Pass 1 - count how many papers fall in each (primary_category, year) stratum.
  Pass 2 - reservoir-sample each stratum down to its proportional quota.

Run:  python src/sample_arxiv.py
"""
import json
import random
from collections import defaultdict

import config as C


def parse_year(rec):
    """Best-effort publication year: first version's created date, else update_date."""
    versions = rec.get("versions") or []
    if versions:
        # created looks like 'Mon, 2 Apr 2020 00:00:00 GMT'
        created = versions[0].get("created", "")
        for tok in created.replace(",", " ").split():
            if tok.isdigit() and len(tok) == 4:
                return int(tok)
    ud = rec.get("update_date", "")
    return int(ud[:4]) if ud[:4].isdigit() else None


def primary_category(rec):
    cats = (rec.get("categories") or "").split()
    return cats[0] if cats else None


def keep(rec):
    """Filter predicate: category prefix + year window."""
    cat = primary_category(rec)
    if cat is None:
        return False
    if C.CATEGORY_PREFIX and not cat.startswith(C.CATEGORY_PREFIX):
        return False
    yr = parse_year(rec)
    if yr is None or not (C.YEAR_MIN <= yr <= C.YEAR_MAX):
        return False
    return True


def stream(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    random.seed(C.RANDOM_SEED)

    # ---- Pass 1: stratum counts ----
    print("Pass 1: counting strata ...")
    counts = defaultdict(int)
    total = 0
    for rec in stream(C.RAW_SNAPSHOT):
        if keep(rec):
            counts[(primary_category(rec), parse_year(rec))] += 1
            total += 1
    print(f"  {total:,} eligible papers across {len(counts):,} strata")

    # ---- proportional quotas summing to SAMPLE_SIZE ----
    quotas = {s: max(1, round(C.SAMPLE_SIZE * n / total)) for s, n in counts.items()}

    # ---- Pass 2: reservoir-sample each stratum to its quota ----
    print("Pass 2: sampling ...")
    reservoirs = defaultdict(list)   # stratum -> list of kept records
    seen = defaultdict(int)
    for rec in stream(C.RAW_SNAPSHOT):
        if not keep(rec):
            continue
        s = (primary_category(rec), parse_year(rec))
        q = quotas[s]
        seen[s] += 1
        slim = {
            "id": rec["id"],
            "title": " ".join((rec.get("title") or "").split()),
            "abstract": " ".join((rec.get("abstract") or "").split()),
            "categories": rec.get("categories", ""),
            "primary_category": s[0],
            "year": s[1],
            "authors_parsed": rec.get("authors_parsed", []),
            "doi": rec.get("doi"),
        }
        res = reservoirs[s]
        if len(res) < q:
            res.append(slim)
        else:                                 # reservoir replacement
            j = random.randint(0, seen[s] - 1)
            if j < q:
                res[j] = slim

    # ---- flatten, dedup by id, cap to SAMPLE_SIZE ----
    sample, ids = [], set()
    for res in reservoirs.values():
        for r in res:
            if r["id"] not in ids:
                ids.add(r["id"])
                sample.append(r)
    random.shuffle(sample)
    sample = sample[: C.SAMPLE_SIZE]

    with open(C.SAMPLE_FILE, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(sample):,} papers -> {C.SAMPLE_FILE}")


if __name__ == "__main__":
    main()
