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
high-prestige institutions. So it can quietly surface those papers more often,
making work from smaller or less-resourced institutions harder to find.

**FairSearch-arXiv** measures whether that happens, and tries to fix it. We build a
RAG pipeline over a 50,000-paper sample of the arXiv corpus, label each paper by the
global standing of its first-author institution, and then ask three questions:

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

Think of it as three layers, built bottom-up. The **data layer** and the
**demographic layer** are done; the **audit + mitigation layer** is wired up and
being run.

### 1. Data layer — cleaned, sampled, enriched *(done)*

- **Read the whole corpus.** Streamed the full arXiv metadata dump —
  **3,080,258 records, 0 parse errors** — and checked field coverage (DOIs are
  present on only ~43% of papers, which is why we don't rely on DOI alone).
- **Deduplicated it.** Removed duplicates by id, DOI, and title, leaving
  **3,071,765 unique records**.
- **Took a reproducible sample.** Drew a **50,000-paper sample** with a fixed seed
  (`seed=42`) so anyone can regenerate the exact same set.
- **Added institution affiliations.** Looked each paper up in two open sources —
  **Semantic Scholar** (matched by arXiv id) and **OpenAlex** (matched by DOI) —
  and resolved **49,652 of 50,000 papers (99.3%)**. About **20,924** of them carry a
  clean first-author institution, spanning **3,473 distinct institutions**.

### 2. Demographic layer — the prestige proxy *(done)*

This is the label the fairness analysis depends on. We join a global university
ranking (QS 2026) onto each paper's first-author institution and tag it:

- **`Privileged`** — first-author institution ranks in the global **Top 20**.
- **`Underrepresented`** — everything else.

Each paper also gets a prestige tier (`top20` / `top50` / … / `unranked`) and a
percentile, so we can slice results finely later.

> **Important — what the label means.** `proxy_group` describes the *institution*,
> based only on the *first author's* affiliation. It is **not** a statement about any
> individual person's background. A Top-20 affiliation doesn't make someone
> "privileged," and being off the ranked list doesn't make someone
> "underrepresented" — most of the world's ~25,000+ universities are simply never
> ranked, and many strong research bodies (national labs, CNRS, the Chinese Academy
> of Sciences, specialist institutes) don't fit a reputation ranking at all. We use
> this only for **corpus-level** analysis of who shows up in search results, never
> for judgements about people.

> **Note on the sample data.** The ranking file bundled in the repo only contains
> the top ~36 universities so the code runs out of the box. Run against that seed and
> everything below rank 36 is treated as unranked — so the split is a **demo, not a
> real result**. Real labels need the full QS export, which we load for the actual
> analysis.

### 3. Audit + mitigation layer — being run now

- **Retrieval works.** A MiniLM-L6-v2 dense index over all 49,652 papers (via FAISS)
  retrieves with precision around **0.57–0.63** across the top 1, 5, and 10 results
  on a 1,000-query test. A stronger SPECTER2 encoder is being indexed alongside it.
- **The fairness audit** (RQ1) compares how often `Privileged` vs `Underrepresented`
  papers reach the Top-10, using **Statistical Parity Difference** and
  **Equalized Odds**. The labels and retrieval results it needs are in place.
- **Generative faithfulness** (RQ2) will check whether the LLM's cited answers lean
  toward consensus/elite sources, scored with an **LLM-as-a-judge** setup.
- **Mitigation** (RQ3) re-ranks results with **MMR** and **Fair-Top-K** and measures
  the trade-off — how much search quality (NDCG@10, MRR) we give up to close the gap.

An earlier baseline comparison found the pattern we expected: the most *accurate*
retriever (SPECTER2) was also the most *concentrated* and most skewed toward
high-resource institutions. Confirming and then reducing that gap is the whole point
of the remaining work.

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
    01_inspect.py            # stream + validate the raw dump, field coverage
    02_sample.py             # dedup (id/DOI/title) + reproducible 50K sample
    03_enrich_openalex.py    # affiliations by DOI (OpenAlex)
    04_enrich_s2.py          # affiliations by arXiv id / title (Semantic Scholar)
    05_rank_and_finalize.py  # canonical institution keys + ranking + final_enriched
    06_rag_baseline.py       # encode + FAISS index + retrieval eval
    07_demographic_enrich.py # QS join -> Privileged / Underrepresented proxy
    evaluate.py              # retrieval fairness audit (SPD, Equalized Odds)
    generate.py              # cited answers (Llama-3-8B) + LLM-as-a-judge
    rerank.py                # MMR + Fair-Top-K, fairness-utility trade-off
  data/                      # snapshot + sample + enriched files  (gitignored)
  results/                   # metric CSVs + generation dumps
```

---

## What's left

- Load the **full** university ranking so the prestige labels are complete, not a demo.
- Finish **SPECTER2** indexing and run the full retrieval-bias audit on both encoders
  plus a BM25 baseline.
- Run the **generation** step and score answer faithfulness with an LLM judge and
  RAGAS.
- Apply **MMR** and **Fair-Top-K** re-ranking and plot the fairness-vs-quality curve.
- Ship a **Streamlit** demo with a fairness toggle, plus the final report and slides.

## Data & license

arXiv metadata is distributed under **CC0** via the Cornell-University/arxiv Kaggle
dataset. Affiliations come from **OpenAlex** and **Semantic Scholar** (openly
licensed). Coursework for CS 6200 — Information Retrieval, Northeastern University.
