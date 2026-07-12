# FairSearch-arXiv

*Evaluating and Mitigating Bias in Academic Retrieval-Augmented Generation*

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/vector%20index-FAISS-4B8BBE)
![SPECTER2 / MiniLM](https://img.shields.io/badge/encoders-SPECTER2%20%2F%20MiniLM-6E56CF)
![Llama-3-8B](https://img.shields.io/badge/LLM-Llama--3--8B--Instruct-4B32C3)
![corpus: arXiv](https://img.shields.io/badge/corpus-arXiv%20metadata-B31B1B?logo=arxiv&logoColor=white)

> **Project Update 2** builds the *demographic layer* on top of the Update 1 baseline.
> It enriches the full 50K sample with institution affiliations (OpenAlex + Semantic
> Scholar), joins an external university ranking to produce a **Privileged (Top-20) vs
> Underrepresented** institution-prestige proxy, and stands up the retrieval + audit
> scaffolding needed for the three research questions. The formal fairness metrics
> (SPD, Equalized Odds), generative-faithfulness judging, and mitigation numbers are
> wired up here and reported with full numbers in the final phase.

**FairSearch-arXiv** asks whether Retrieval-Augmented Generation (RAG) for academic
search quietly favours well-resourced, high-prestige institutions. Dense retrievers
learn from a literature that already over-represents elite labs, so they can surface
those papers disproportionately — making work from less-resourced institutions
systematically harder to find. This repo audits that effect over a stratified 50K
sample of the Cornell arXiv corpus.

---

## Research questions

- **RQ1 — Retrieval bias audit.** Across a standardized query set, how are papers
  from *Underrepresented* institutions ranked in the Top-10 relative to *Privileged*
  (Top-20) institutions? Measured with **Statistical Parity Difference (SPD)** and
  **Equalized Odds** over the demographic proxy.
- **RQ2 — Generative faithfulness.** When Llama-3-8B-Instruct synthesises a cited
  answer, does it over-cite consensus/elite sources? Measured with a
  **Pro-Consensus vs Dissenting** citation-ratio analysis and an **LLM-as-a-judge**
  evaluation.
- **RQ3 — Mitigation.** What is the fairness–utility tradeoff of **MMR** and
  **Fair-Top-K** re-ranking, quantified as the change in **NDCG@10** and **MRR**
  against the shift in the demographic exposure gap?

## Team

| Member | Role in the pipeline |
|--------|----------------------|
| Aarushi Kaushik | Retrieval & indexing (SPECTER2 / MiniLM, FAISS) |
| Pavani Jain | Corpus enrichment, demographic mapping, fairness evaluation (SPD, Equalized Odds) |
| Pratyush Tyagi | Bias mitigation (Fair-Top-K re-rank, MMR, fair prompting) |
| Shweta Pattanaik | Streamlit interface, demo, integration |

---

## What's new in Update 2

- **Processed the full corpus, not a slice.** Streamed and validated the entire
  arXiv metadata dump (**3,080,258 records, 0 parse errors**), deduplicated it
  (by id / DOI / title) to **3,071,765 unique records**, then drew a reproducible
  **50,000-record sample** with a two-pass reservoir sampler (`seed=42`).
- **Enriched every sampled paper with affiliations.** A two-source join —
  **Semantic Scholar** (matched on arXiv id) and **OpenAlex** (matched on DOI) —
  resolved **49,652 / 50,000 papers (99.3%)**, of which **20,924** carry a usable
  canonical first-author institution across **3,473 distinct institutions**.
- **Built the demographic proxy (`07_demographic_enrich.py`).** Joined an external
  global university ranking (QS 2026) onto the canonical institution keys and
  labelled each paper `Privileged` (Top-20) vs `Underrepresented`, plus a prestige
  tier and two percentile views. This is the core new deliverable — details and
  caveats below.
- **Stood up the retrieval baseline + audit inputs.** A MiniLM-L6-v2 FAISS index
  over all 49,652 documents with a 1,000-query retrieval eval; SPECTER2 (768-d)
  indexing is in progress. The demographic labels + retrieval pools are exactly the
  inputs SPD/Equalized-Odds and the mitigation re-rankers consume.

---

## The demographic mapping — read this before trusting a label

`07_demographic_enrich.py` adds **one additive step** that joins a single global
university ranking onto the corpus and labels each paper by the global standing of
its **first-author institution**.

**What `proxy_group` is and is not.** It is a coarse **institutional-prestige
proxy**, not a measured demographic attribute of any person.

- A Top-20 affiliation does **not** establish that an individual is "privileged."
- Absence from the ranked list does **not** establish "underrepresented" — most of
  the world's ~25,000–30,000 universities are simply never ranked, and many strong
  research institutions (national labs, CNRS, INFN, the Chinese Academy of Sciences,
  specialist institutes) are not comparable to a reputation-weighted ranking at all.
- The label describes **institutions**, derived from the **first author's**
  affiliation only. Use it for **corpus-level** structural analysis, never for
  claims about individuals.

**Method (five steps).**
1. **One ranking, full list** — QS World University Rankings 2026. Don't mix
   QS/THE/ARWU: each has a different methodology and list size, which breaks the
   percentile math.
2. **Record N and each rank R** — `N` is the list size (`--n 1501`, QS 2026's
   published count). Banded ranks (`601-610`, `1001+`) resolve via `--band-policy`
   (`lower`, default, or `midpoint`), applied consistently.
3. **Rank → percentile** — `inst_global_percentile = (1 − R/N) × 100`.
4. **Elite cutoff** — `--elite-mode rank --elite-rank 20` flags global rank ≤ 20 as
   elite → `Privileged`; everything else → `Underrepresented` (configurable).
5. **Denominator sanity check** — percentile *within the ranked list* is not the
   same as percentile *among all universities*. Passing `--n-global 28000` emits a
   second `inst_percentile_all`, and the summary states both explicitly, so the two
   claims never get conflated.

> **⚠️ Caveat when running against the shipped seed.** The bundled
> `qs_world_university_rankings_2026.csv` contains only the **verbatim top tier
> (ranks 1–36)** so the step is runnable out of the box. Against that partial seed,
> **every institution ranked 37+ falls into `unranked` / `Underrepresented`** —
> including genuinely high-ranked schools — so the `top100/top200/top500` tiers are
> empty and any `Privileged`/`Underrepresented` split is a **demo artifact, not a
> faithful label**. For real numbers, export the full QS 2026 table
> (`rank,university[,country]`) and point `--ranking` at it before reporting.

> **Inherited limitation.** `05`'s normalizer collapses
> `"University of California, X"` to one `university of california` key, so all UC
> campuses share a bucket and inherit UC-Berkeley's rank. This is kept for
> consistency with the existing `canonical_institution_key`.

### New per-record fields

| field | type | meaning |
|---|---|---|
| `inst_global_rank` | int \| null | R for the first-author institution (banded → per policy) |
| `inst_global_rank_banded` | bool | R came from a band |
| `inst_global_percentile` | float \| null | `(1−R/N)×100`, **within the ranked list** |
| `inst_rank_source` | str | e.g. `QS-2026` |
| `inst_rank_N` | int | denominator used |
| `inst_prestige_tier` | str | `top20`/`top50`/`top100`/`top200`/`top500`/`ranked`/`unranked` |
| `is_elite_institution` | bool | per chosen cutoff |
| `proxy_group` | str | `Privileged` / `Underrepresented` |
| `proxy_group_basis` | str | self-documenting provenance of the label |
| `inst_percentile_all` | float \| null | Step-5 all-universities view (only if `--n-global` set) |

The output `final_enriched_demographics.jsonl` is a **superset** of
`final_enriched.jsonl`, so the retrieval step runs unchanged against it and every
downstream metric can be sliced by `proxy_group` or `inst_prestige_tier` without
rebuilding the index.

---

## Results

### Corpus & enrichment (this phase — measured)

| Stage | Value |
|---|---|
| Raw records streamed | 3,080,258 (0 parse errors) |
| DOI coverage (raw) | 42.7% |
| Unique after dedup | 3,071,765 (dups: 88 id · 2,374 DOI · 6,326 title) |
| Sample size (`seed=42`) | 50,000 |
| Resolved with affiliation source | 49,652 (99.3%) — 28,316 Semantic Scholar · 21,336 OpenAlex |
| Papers with canonical institution | 20,924 across 3,473 institutions (880 with ≥5 papers) |
| Within-sample prestige flag `top_uni=True` | 1,182 (2.38%) |

### Retrieval baseline (this phase — measured)

MiniLM-L6-v2 (384-d), FAISS index over all 49,652 documents, 1,000-query eval:

| k | mean P@k | mean R@k |
|---|:---:|:---:|
| 1 | 0.628 | 0.0006 |
| 5 | 0.589 | 0.0024 |
| 10 | 0.572 | 0.0045 |

Precision holds ~0.57–0.63 across cutoffs while recall stays small, consistent with
a large relevant pool per query over a ~50K corpus. SPECTER2 (768-d) encoding is
**in progress** (MPS backend, ~13 docs/s); the 200-query known-item benchmark from
Update 1 remains the primary head-to-head retrieval-quality protocol.

### Carried forward from Update 1 (prior phase — for reference)

200-query known-item benchmark, heuristic queries, macro-averaged:

| Retriever | P@5 | P@10 | R@5 | R@10 | nDCG@10 | MRR |
|-----------|:---:|:----:|:---:|:----:|:-------:|:---:|
| BM25 (baseline) | 0.241 | 0.198 | 0.241 | 0.396 | 0.312 | 0.387 |
| **SPECTER2** (dense) | **0.318** | **0.267** | **0.318** | **0.534** | **0.421** | **0.498** |
| MiniLM (dense) | 0.289 | 0.241 | 0.289 | 0.482 | 0.378 | 0.451 |

| Retriever | Cat KL | Year KL | Gini | Global-N share |
|-----------|:------:|:-------:|:----:|:--------------:|
| BM25 | 0.043 | 0.019 | 0.31 | 72% |
| SPECTER2 | 0.118 | 0.054 | 0.61 | 83% |
| MiniLM | 0.092 | 0.041 | 0.54 | 79% |

The Update 1 tension in one line: **the most accurate retriever (SPECTER2) is also
the most concentrated and the most skewed.** Update 2 replaces those proxy signals
with the formal, institution-tier metrics below.

### RQ1 / RQ2 / RQ3 status

| RQ | Metric | Status |
|---|---|---|
| RQ1 | SPD, Equalized Odds over `proxy_group`; Top-10 tier distribution | Inputs ready (labels + retrieval pools); numbers land once SPECTER2 indexing completes and the **full** QS table replaces the partial seed |
| RQ2 | Pro-Consensus vs Dissenting citation ratio; LLM-as-a-judge | Methodology fixed; runs on the closed RAG loop in the final phase |
| RQ3 | MMR + Fair-Top-K re-rank; ΔNDCG@10, ΔMRR, exposure gap | Re-rankers designed; fairness–utility curve reported in the final phase |

---

## Methodology (measurement design)

- **Standardized queries.** A fixed known-item query set (one target paper per
  query, hit = target appears in Top-k), reproducible from `seed=42`. The harder
  LLM-generated query set (prompted with **Claude Sonnet**) is used for the final
  report to separate dense from lexical retrieval more honestly.
- **Statistical Parity Difference (SPD).** Difference in the rate at which
  `Privileged` vs `Underrepresented` papers appear in the Top-k, across the query
  set. Zero = parity; positive = Privileged over-exposure.
- **Equalized Odds.** Compares true-positive (relevant-and-retrieved) and
  false-positive rates *conditioned on* `proxy_group`, so a fair retriever is not
  just balanced overall but balanced among the papers that actually deserve
  retrieval.
- **MMR re-ranking.** Trades relevance against diversity (`λ`) to reduce
  concentration on a small popular set.
- **Fair-Top-K re-ranking.** Enforces a minimum representation of
  `Underrepresented` papers in the Top-k, then reports the utility cost
  (ΔNDCG@10, ΔMRR) against the exposure gain.

All metrics are sliced by `proxy_group` and `inst_prestige_tier` off the enriched
JSONL, so no re-indexing is needed to audit or mitigate.

---

## How it works

```
arXiv metadata dump ─> 01 inspect ─> 02 dedup + sample (seed=42) ─> sample_50k.jsonl
                                                                        │
        03 enrich (OpenAlex, DOI) ┐                                     │
        04 enrich (Sem. Scholar) ─┴─> 05_rank_and_finalize.py ─> final_enriched.jsonl
                                                                        │
                                          07_demographic_enrich.py ─────┤─> final_enriched_demographics.jsonl
                                          (QS join → proxy_group)       │      + demographic_enrichment_summary.txt
                                                                        │
                          06_rag_baseline.py ──────────────────────────┘─> rag_index.faiss (+ SPECTER2 index)
                                    │
                    ┌───────────────┼────────────────────┐
             evaluate (RQ1)   generate (RQ2)        rerank (RQ3)
          SPD / Eq.Odds /   Llama-3-8B cited     MMR / Fair-Top-K
          tier distribution  answers + judge     ΔNDCG@10 / ΔMRR
```

---

## Quickstart

### Clone & install

```bash
git clone https://github.com/PratyushTyagi/IR-FairSearch-arXiv-Team6.git
cd IR-FairSearch-arXiv-Team6
pip install -r requirements.txt
```

A GPU is strongly recommended — dense encoding over 50K papers takes minutes on GPU
vs ~1 h+ on CPU/MPS. Llama-3-8B-Instruct (4-bit) is a gated Hugging Face model, so
run `huggingface-cli login` once before the generation step.

### Get the data

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv -p data --unzip
# yields data/arxiv-metadata-oai-snapshot.json  (~5 GB, CC0, refreshed weekly)
```

### Run the pipeline

```bash
# 1) inspect + sample the corpus
python scripts/01_inspect.py
python scripts/02_sample.py                     # -> data/sample_50k.jsonl (seed=42)

# 2) enrich with affiliations (set a contact email for the OpenAlex polite pool)
python scripts/03_enrich_openalex.py
python scripts/04_enrich_s2.py
python scripts/05_rank_and_finalize.py          # -> data/final_enriched.jsonl + institution_ranking.csv

# 3) demographic proxy layer (use the FULL QS table for real labels)
python scripts/07_demographic_enrich.py \
  --root . \
  --ranking data/qs_world_university_rankings_2026.csv \
  --source QS-2026 --n 1501 \
  --elite-mode rank --elite-rank 20 \
  --n-global 28000
# -> data/final_enriched_demographics.jsonl + demographic_enrichment_summary.txt

# 4) retrieval baseline + eval
python scripts/06_rag_baseline.py               # -> data/rag_index.faiss + rag_eval.csv
```

Switch ranking source with `--source THE-2026 --n 2191` or `--source ARWU --n 1000`
(supply that source's table; never mix sources in one run).

---

## Reproducibility

Every stage keys off a fixed `RANDOM_SEED = 42`, so the sample, the query set, and
the enrichment regenerate deterministically. Enrichment is resumable (the S2 step
resumes from the parquet it has already written), retrieval is deterministic given
the index, and each script writes a self-contained artifact
(`sample_50k.jsonl`, `final_enriched.jsonl`, `final_enriched_demographics.jsonl`,
`institution_ranking.csv`, `rag_index.faiss`, `rag_eval.csv`), so every number
above regenerates end-to-end from the raw Kaggle snapshot.

---

## Repository layout

```
IR-FairSearch-arXiv-Team6/
  README.md
  requirements.txt
  .gitignore
  scripts/
    01_inspect.py                   # stream + validate the raw dump, key/null coverage
    02_sample.py                    # dedup (id/DOI/title) + two-pass reservoir 50K sample
    03_enrich_openalex.py           # affiliations by DOI (OpenAlex)
    04_enrich_s2.py                 # affiliations by arXiv id / title (Semantic Scholar)
    05_rank_and_finalize.py         # canonical institution keys, ranking, final_enriched
    06_rag_baseline.py              # MiniLM/SPECTER2 encode + FAISS index + P@k/R@k eval
    07_demographic_enrich.py        # QS join -> proxy_group (Privileged/Underrepresented)
    evaluate.py                     # RQ1: SPD, Equalized Odds, Top-10 tier distribution
    generate.py                     # RQ2: Llama-3-8B-Instruct cited answers + LLM judge
    rerank.py                       # RQ3: MMR + Fair-Top-K, fairness-utility curve
  data/                             # snapshot + sample + enriched + rankings  (gitignored)
  results/                          # metric CSVs + generation dumps
```

---

## Final-stage plan (through Week 14)

- Load the **full QS 2026 table** to replace the partial seed and produce faithful
  `proxy_group` labels across all prestige tiers.
- Finish **SPECTER2** indexing and run the full **RQ1** audit (SPD + Equalized Odds
  + Top-10 tier distribution) on both encoders and BM25.
- Close the RAG loop for **RQ2**: Llama-3-8B-Instruct cited generation, then the
  Pro-Consensus vs Dissenting citation-ratio analysis and **LLM-as-a-judge**
  faithfulness scoring; add **RAGAS** evaluation.
- Run **RQ3** mitigation (MMR + Fair-Top-K), plot the fairness–utility curve
  (NDCG@10 vs exposure gap), and do prompt engineering / fairness-aware prompting.
- Ship the **Streamlit** demo with a fairness toggle, plus the finalized ACM-style
  report and 10-slide deck with real numbers.

---

## Data & license

arXiv metadata is distributed under **CC0** via the Cornell-University/arxiv Kaggle
dataset; affiliations are enriched from **OpenAlex** and **Semantic Scholar**
(openly licensed, polite-pool access). This repository is coursework for
**CS 6200 — Information Retrieval**, Northeastern University.
