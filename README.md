# FairSearch-arXiv

*Evaluating and mitigating bias in academic Retrieval-Augmented Generation (RAG)*

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/vector%20index-FAISS-4B8BBE)
![SPECTER2 / MiniLM](https://img.shields.io/badge/encoders-SPECTER2%20%2F%20MiniLM-6E56CF)
![Llama-3-8B](https://img.shields.io/badge/LLM-Llama--3--8B--Instruct-4B32C3)
![corpus: arXiv](https://img.shields.io/badge/corpus-arXiv%20metadata-B31B1B?logo=arxiv&logoColor=white)

## What this project is about

When you search academic papers with a modern RAG system — a dense retriever that
finds papers, plus an LLM that reads them and writes a cited answer — the retriever
learns from a body of literature that already over-represents large, well-resourced,
high-prestige institutions. So it could quietly surface those papers more often,
making work from smaller or less-resourced institutions harder to find.

**FairSearch-arXiv** measures whether that actually happens, and tries to fix it. We
build a RAG pipeline over a 50,000-paper sample of the arXiv corpus, label each paper
by the prestige of its first-author institution, and then ask three questions:

1. **Does retrieval favour high-prestige institutions?** (retrieval bias audit)
2. **Does the LLM's cited answer over-rely on those same sources?** (generative faithfulness)
3. **Can we rebalance results without wrecking search quality?** (mitigation)

This repo is coursework for **CS 6200 — Information Retrieval**, Northeastern
University.

## Team

| Member | Focus |
|--------|-------|
| Aarushi Kaushik | Retrieval & indexing (SPECTER2 / MiniLM, FAISS) |
| Pavani Jain | Corpus enrichment, demographic mapping, fairness evaluation |
| Pratyush Tyagi | Bias mitigation (Fair-Top-K re-rank, MMR, fair prompting) |
| Shweta Pattanaik | Streamlit interface, demo, integration |

---

## Where the project is right now

Built bottom-up in four stages. The first three are **done with results**; the
generation-faithfulness study and the demo are the remaining work.

### 1. Data — cleaned, sampled, enriched *(done)*

- **Read the whole corpus.** Streamed the full arXiv metadata dump —
  **3,080,258 records, 0 parse errors** — and checked field coverage (DOIs exist on
  only ~43% of papers, so we don't rely on DOI alone).
- **Deduplicated it** by id, DOI, and title → **3,071,765 unique records**.
- **Sampled 50,000 papers** with a fixed seed (`seed=42`) so anyone regenerates the
  same set.
- **Added institution affiliations** from two open sources — **Semantic Scholar**
  (by arXiv id) and **OpenAlex** (by DOI) — resolving **49,652 / 50,000 papers
  (99.3%)**. About **20,924** carry a clean first-author institution across
  **3,473 distinct institutions**.

### 2. Prestige labels *(done)*

Two ways to mark a paper as "elite," so the fairness analysis has a group to compare:

- **`top_uni`** *(used in the audit below)* — the paper's first-author institution
  is among the **top 50 by mean citations** in our sample. This flags **1,182 papers
  (2.38%)** and is fully data-driven from our own corpus.
- **`proxy_group`** *(QS Top-20)* — a parallel label from an external ranking (QS
  2026): `Privileged` if the institution is global Top-20, else `Underrepresented`.
  On the current run it tags **1,638 papers Privileged** and **48,014
  Underrepresented**; the Top-20 schools present include Berkeley, Cambridge,
  Caltech, ETH Zürich, Oxford, MIT, Harvard, Peking, Stanford, and Tsinghua.

> **What these labels mean.** They describe the *institution*, from the *first
> author's* affiliation only — never an individual person's background. A top-tier
> affiliation doesn't make someone "privileged," and being off a ranked list doesn't
> make someone "underrepresented" (most of the world's ~25,000+ universities are
> never ranked). We use these for **corpus-level** analysis of what shows up in
> search, not for judgements about people.

> **Note — the QS split is still a demo.** The bundled ranking file holds only the
> top ~37 schools so the code runs out of the box, so *every* institution ranked 38+
> currently lands in `unranked` / `Underrepresented` (tier counts: top20 = 1,638,
> top50 = 633, everything else unranked). The 1,638 / 48,014 split above is therefore
> an illustrative artifact, not a faithful result — load the full QS export to
> populate the lower tiers. One inherited quirk: the normalizer collapses all
> University of California campuses to one key, so "Berkeley" (339 papers) really
> means system-wide UC. Because of this, the audit results below use the complete,
> data-driven `top_uni` group rather than `proxy_group`.

### 3. Retrieval + bias audit + mitigation *(done)*

**Retrieval quality.** A MiniLM-L6-v2 FAISS index over all 49,652 papers retrieves
with precision **0.628 / 0.589 / 0.572** at k = 1 / 5 / 10 (1,000-query test,
same-category relevance). SPECTER2 is the stronger encoder used for the fairness
study; its document embeddings are materialized over the full corpus (49,652 × 768).

**The bias audit (RQ1).** Over 100 known-item queries we measured how often
elite-institution (`top_uni`) papers reach the Top-k, using **Statistical Parity
Difference (SPD)** and **Equalized Odds (EO)** with 1,000-sample bootstrap CIs.

> **Headline finding: the retriever tracks corpus prevalence — no meaningful
> parity gap.** `top_uni` papers are 2.38% of the corpus and fill ~2.6–2.9% of the
> Top-10 pool. SPD is statistically indistinguishable from zero (its 95% CI spans
> zero at every k), and the Equalized-Odds gap is tiny (~0.001–0.002). This is a
> more careful, per-query measurement than the concentration proxies (Gini,
> Global-North share) we reported in the first phase, and it does **not** reproduce
> a large elite-institution skew on this group.

**Mitigation (RQ3).** We re-ranked with **MMR** (diversity, best λ = 0.7) and
**Fair-Top-K** (a floor quota that guarantees representation of non-elite papers),
then measured the utility cost.

| Method (SPECTER) | k | P@k | Top-elite share | SPD | Equalized Odds |
|---|:--:|:--:|:--:|:--:|:--:|
| Baseline | 5 | 0.554 | 2.6% | ≈0 (CI spans 0) | 0.0021 |
| Baseline | 10 | 0.536 | 2.9% | ≈0 (CI spans 0) | 0.0014 |
| MMR (λ=0.7) | 5 | 0.504 | 2.2% | ≈0 | 0.0019 |
| MMR (λ=0.7) | 10 | 0.515 | 3.0% | ≈0 | 0.0012 |
| **Fair-Top-K** | 5 | **0.554** | 2.4% | ≈0 | 0.0021 |
| **Fair-Top-K** | 10 | **0.536** | 2.6% | ≈0 | 0.0014 |

> **The fairness–utility trade-off, in one line:** because the baseline is already
> near parity, there's little to fix — and it shows in the trade-off. **Fair-Top-K
> is effectively free** (precision unchanged at 0.554 / 0.536, elite share nudged
> down), while **MMR pays ~2–5 precision points for no measurable fairness gain**.

### 4. Generative faithfulness + demo *(next)*

RQ2 — whether Llama-3-8B-Instruct's cited answers over-rely on consensus/elite
sources, scored with an **LLM-as-a-judge** setup and RAGAS — plus the Streamlit demo,
are the remaining pieces.

---

## How it fits together

```
arXiv metadata dump ─> inspect ─> dedup + sample (seed=42) ─> sample_50k.jsonl
                                                                   │
        enrich (OpenAlex by DOI) ┐                                 │
        enrich (Semantic Scholar) ┴─> rank + finalize ─> final_enriched.jsonl
                                                                   │
                          demographic enrich (QS join) ────────────┤─> final_enriched_demographics.jsonl
                          proxy_group: Privileged / Underrepresented│
                                                                   │
                          build RAG index (MiniLM / SPECTER2) ──────┘─> FAISS index
                                    │
                   ┌────────────────┼─────────────────────┐
            retrieval audit   cited generation        re-ranking
            (SPD, Eq. Odds)   (Llama-3 + LLM judge)   (MMR, Fair-Top-K)
              [done]              [next]                 [done]
```

---

## Try it yourself

### Install

```bash
git clone https://github.com/PratyushTyagi/IR-FairSearch-arXiv-Team6.git
cd IR-FairSearch-arXiv-Team6
pip install -r requirements.txt
```

A GPU helps a lot — encoding 50K papers takes minutes on GPU vs an hour-plus on
CPU/MPS. The generation step uses Llama-3-8B-Instruct, a gated Hugging Face model, so
run `huggingface-cli login` once first.

### Get the data

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv -p data --unzip
# yields data/arxiv-metadata-oai-snapshot.json  (~5 GB, CC0, refreshed weekly)
```

### Run the pipeline

```bash
python scripts/01_inspect.py                    # scan + validate the raw dump
python scripts/02_sample.py                     # dedup + 50K sample -> sample_50k.jsonl
python scripts/03_enrich_openalex.py            # affiliations by DOI
python scripts/04_enrich_s2.py                  # affiliations by arXiv id / title
python scripts/05_rank_and_finalize.py          # institution ranking -> final_enriched.jsonl
python scripts/07_demographic_enrich.py \       # prestige proxy -> final_enriched_demographics.jsonl
  --ranking data/qs_world_university_rankings_2026.csv --source QS-2026 --n 1501 \
  --elite-mode rank --elite-rank 20 --n-global 28000
python scripts/06_rag_baseline.py               # embed + FAISS index + retrieval eval
python scripts/08_specter_fairness_baseline.py  # SPECTER audit: SPD, Equalized Odds
python scripts/09_rerank.py                     # MMR + Fair-Top-K, fairness-utility trade-off
```

Everything keys off a fixed `seed=42`, and each step writes a self-contained file, so
the whole thing regenerates end-to-end from the raw snapshot. You can swap the ranking
source (`--source THE-2026 --n 2191`, `--source ARWU --n 1000`) — just don't mix
sources in one run.

---

## Repository layout

```
IR-FairSearch-arXiv-Team6/
  README.md
  requirements.txt
  scripts/
    01_inspect.py                  # stream + validate the raw dump, field coverage
    02_sample.py                   # dedup (id/DOI/title) + reproducible 50K sample
    03_enrich_openalex.py          # affiliations by DOI (OpenAlex)
    04_enrich_s2.py                # affiliations by arXiv id / title (Semantic Scholar)
    05_rank_and_finalize.py        # canonical institution keys + ranking + final_enriched
    06_rag_baseline.py             # MiniLM encode + FAISS index + retrieval eval
    07_demographic_enrich.py       # QS join -> Privileged / Underrepresented proxy
    08_specter_fairness_baseline.py# SPECTER audit: SPD + Equalized Odds (bootstrap CIs)
    09_rerank.py                   # MMR + Fair-Top-K, fairness-utility trade-off
    generate.py                    # (next) cited answers (Llama-3) + LLM-as-a-judge
  data/                            # snapshot + sample + enriched + indexes  (gitignored)
  results/                         # metric CSVs + bias-score JSON + generation dumps
```

Key result files: `rag_eval.csv` (retrieval quality), `baseline_bias_scores.json`
(RQ1 audit), `comparison_table.csv` / `rerank_bias_scores.json` (RQ3 mitigation),
`mmr_lambda_sweep.csv` (λ selection). The demographic layer emits
`institution_ranking_enriched.csv`, `university_global_ranking_normalized.csv`,
`final_enriched_demographics.jsonl`, and `demographic_enrichment_summary.txt`.

---

## What's left

- Load the **full** QS ranking so the `proxy_group` labels are complete, not a demo,
  and re-run the audit sliced by QS Top-20 alongside `top_uni`.
- Run the **generation** step (RQ2): Llama-3-8B cited answers, Pro-Consensus vs
  Dissenting citation analysis, LLM-as-a-judge scoring, and RAGAS.
- Ship a **Streamlit** demo with a fairness toggle, plus the final report and slides.

## Data & license

arXiv metadata is distributed under **CC0** via the Cornell-University/arxiv Kaggle
dataset. Affiliations come from **OpenAlex** and **Semantic Scholar** (openly
licensed). Coursework for CS 6200 — Information Retrieval, Northeastern University.
