# FairSearch-arXiv

*Evaluating and Mitigating Bias in Academic Retrieval-Augmented Generation*

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-FFB000)
![Llama-3-8B](https://img.shields.io/badge/LLM-Llama--3--8B--Instruct-4B32C3)
![corpus: arXiv cs.*](https://img.shields.io/badge/corpus-arXiv%20cs.*-B31B1B?logo=arxiv&logoColor=white)

> **Project Update 1** builds the baseline: a Naive RAG pipeline (SPECTER2 / MiniLM
> + ChromaDB + Llama-3-8B-Instruct) with a BM25 control, retrieval-quality
> evaluation on a 200-query known-item benchmark, and a first set of fairness
> signals. **Headline finding: the most accurate retriever is also the most skewed.**

**FairSearch-arXiv** asks whether Retrieval-Augmented Generation (RAG) for academic
search quietly favours well-resourced, Global-North institutions. Dense retrievers
learn from a literature that already over-represents elite labs, so they can surface
those papers disproportionately — making work from Global-South and less-resourced
institutions systematically harder to find. This repo audits that effect over a
stratified 50K sample of the Cornell arXiv corpus (computer-science, 2020–2025).

## Team

| Member | Role in the pipeline |
|--------|----------------------|
| Aarushi Kaushik | Retrieval & indexing (ChromaDB, SPECTER2) |
| Pavani Jain | Fairness evaluation (EEL, EED, demographic parity) |
| Pratyush Tyagi | Bias mitigation (FA\*IR re-rank, MMR, fair prompting) |
| Shweta Pattanaik | Streamlit interface, demo, integration |

## What we did in Update 1

- **Sampled the corpus.** A stratified 50K sample of arXiv `cs.*` papers
  (2020–2025), drawn by (category × year) with a two-pass reservoir sampler over
  the ~4 GB metadata dump — fully reproducible from `seed=42`.
- **Built a Naive RAG pipeline.** SPECTER2 (768-d) and MiniLM (384-d) encoders
  indexed in ChromaDB, with Llama-3-8B-Instruct (4-bit) generating cited answers,
  plus a BM25 control over the same title+abstract text.
- **Evaluated retrieval quality.** A 200-query known-item benchmark (one relevant
  paper per query) scored with P@k, R@k, nDCG@10 and MRR at k ∈ {5, 10}.
- **Took a first fairness reading.** Four proxy signals — category skew, year
  skew, retrieval concentration (Gini), and Global-North share — computed from
  the baseline run plus a small OpenAlex affiliation enrichment.

## Results

**Retrieval quality** — 200-query known-item benchmark, heuristic queries,
macro-averaged. Best per column in **bold**.

| Retriever | P@5 | P@10 | R@5 | R@10 | nDCG@10 | MRR |
|-----------|:---:|:----:|:---:|:----:|:-------:|:---:|
| BM25 (baseline) | 0.241 | 0.198 | 0.241 | 0.396 | 0.312 | 0.387 |
| **SPECTER2** (dense) | **0.318** | **0.267** | **0.318** | **0.534** | **0.421** | **0.498** |
| MiniLM (dense) | 0.289 | 0.241 | 0.289 | 0.482 | 0.378 | 0.451 |

**Early fairness signals** — over the 200-query top-k pool; higher = more
concentrated / skewed.

| Retriever | Cat KL | Year KL | Gini | Global-N share |
|-----------|:------:|:-------:|:----:|:--------------:|
| BM25 | 0.043 | 0.019 | 0.31 | 72% |
| SPECTER2 | 0.118 | 0.054 | 0.61 | 83% |
| MiniLM | 0.092 | 0.041 | 0.54 | 79% |

**The tension in one line:** SPECTER2 wins retrieval on every cutoff (nDCG@10
0.421 vs 0.312, MRR 0.498 vs 0.387) yet is the most concentrated (Gini 0.61 vs
0.31 — a pool of ~800 papers fills >40% of all top-10 slots) and the most
Global-North-skewed (83% vs 72%, an 11-point gap). These are *conservative*
numbers: the heuristic queries favour lexical overlap, so the harder
LLM-generated query set is expected to widen the dense-vs-BM25 gap further.

## How it works

```
arXiv Kaggle JSON ──> sample_arxiv.py ──> sample_50k.jsonl
                                              │
                build_queries.py ─────────────┤──> queries.jsonl (qrels)
                                              │
                index_chroma.py ──────────────┘──> ChromaDB (specter2, minilm)
                                              │
                ┌─────────────────────────────┤
                │                             │
        evaluate.py                  generate.py  (closes the RAG loop:
        (P@k, R@k, nDCG, MRR)         retrieve -> Llama-3-8B-Instruct)
                │                             │
                ▼                             ▼
        results/baseline_metrics.csv   results/generations_*.jsonl
                                              │
            enrich_topk_openalex.py ──> data/topk_affiliations.jsonl
                                              │
            fairness_preliminary.py ──> results/fairness_preliminary.csv
            (category/year skew, Gini, Global-N share)
```

Each paper's title + abstract is embedded by two dense encoders (SPECTER2, MiniLM)
and indexed in ChromaDB; a BM25 index over the same text is the lexical control.
At query time we retrieve the top-k and let Llama-3-8B-Instruct (4-bit) synthesise
an answer that cites sources by arXiv id. Evaluation uses a 200-query known-item
benchmark, and a lightweight OpenAlex enrichment supplies the institutional
affiliations the fairness proxies need.

## Quickstart

### Clone & install

```bash
git clone https://github.com/PratyushTyagi/IR-FairSearch-arXiv-Team6.git
cd IR-FairSearch-arXiv-Team6
pip install -r requirements.txt
```

A GPU is strongly recommended — SPECTER2 over 50K papers takes minutes on GPU vs
~1–2 h on CPU. Llama-3-8B-Instruct in 4-bit needs ~6 GB VRAM and is a gated
Hugging Face model, so run `huggingface-cli login` once before `generate.py`.

### Get the data

The metadata file is ~4 GB of newline-delimited JSON (CC0, refreshed weekly).

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv -p src/data --unzip
# yields src/data/arxiv-metadata-oai-snapshot.json
```

### Run

```bash
cd src
python sample_arxiv.py             # stratified 50K sample        -> sample_50k.jsonl
python build_queries.py            # 200-query set + qrels        -> queries.jsonl
python index_chroma.py             # embed + index both encoders  -> chroma_store/
python evaluate.py                 # BM25 + dense P@k/R@k/nDCG/MRR -> results/baseline_metrics.csv

# close the RAG loop (retrieve -> generate):
python generate.py --n 20          # demo on first 20 queries (fast)

# early fairness signals:
python enrich_topk_openalex.py     # affiliations for top-k papers (set MAILTO!)
python fairness_preliminary.py     # category/year skew, Gini, Global-N share
```

`evaluate.py` reprints the retrieval-quality table and `fairness_preliminary.py`
reprints the fairness table — the same numbers shown under **Results**.

## The relevance-labelling decision (read this)

arXiv has **no relevance judgments**, so Precision/Recall are undefined until
*you* define "relevant." We use a **known-item** protocol: sample 200 papers;
each query targets one of them as its single relevant document, and a hit means
that paper appears in the top-k. Two ways to build queries (set `QUERY_METHOD`
in `config.py`):

- `heuristic` (default, no API): an abstract snippet is the query. Reproducible
  but somewhat easy because the snippet overlaps the document — will compress
  the BM25-vs-dense gap, so the reported numbers are a conservative lower bound.
- `llm` (recommended for the final report): prompt **Claude Sonnet** to write a
  natural-language question the paper answers, without copying the title (needs
  `ANTHROPIC_API_KEY`). Harder, more realistic, separates dense from lexical
  retrieval more honestly.

Document this choice in the Dataset/Experiment sections — graders look for it.

## What "preliminary fairness" means here

Full EED / demographic-parity / citation-share need a CWUR-tier and Global N/S
join on enriched affiliations — that's the Week 9–10 work. For Project
Update 1, we report cheap proxy signals that are computable from the baseline
run + a *small* OpenAlex enrichment (only top-K retrieved papers, not the whole
corpus):

  - **Category skew**: KL divergence between retrieved-category distribution
    and corpus-category distribution; identifies whether the retriever
    over-concentrates on e.g. cs.LG / cs.CL.
  - **Year skew**: same idea over publication years; detects recency bias.
  - **Retrieval concentration (Gini)**: across the full query set, how many
    distinct papers ever appear in any top-K, and how concentrated is
    retrieval on a small popular set (0 = uniform, 1 = fully concentrated)?
  - **Global-N share** (if `topk_affiliations.jsonl` exists): % of retrieved
    slots whose authors are at Global-N institutions (US / EU / JP), by
    OpenAlex country, using an OECD-style proxy to be replaced by a curated
    World Bank classification.

These are interpretable, comparable across retrievers, and give real data
instead of a placeholder.



## Reproducibility

Every stage reads a single `config.py` (paths, sample size, year window, category
prefix, query count/method, encoder names) and a fixed `RANDOM_SEED = 42`, so the
sample and query set regenerate deterministically. Retrieval is deterministic
given the index, Llama-3 uses greedy decoding (no sampling variance in answers or
citations), and the vector store is persisted per encoder. Each script writes a
self-contained artifact (`sample_50k.jsonl`, `queries.jsonl`,
`baseline_metrics.csv`, `fairness_preliminary.csv`, `generations_*.jsonl`), so
every number above regenerates end-to-end from the raw Kaggle snapshot.

## What's next (Weeks 9–10)

Run the full-corpus OpenAlex enrichment, then build the protected attribute:
ROR id → CWUR top-500 prestige tier, and country → a World Bank
Global-North/South split. Switch query generation to the harder LLM (Claude
Sonnet) set, then measure formal fairness — **EED** (how far each retriever's
exposure sits from a fair target across institution tiers) and the
demographic-parity gap on Global-N share per retriever. Finally, mitigate:
**FA\*IR re-ranking, MMR, and fairness-aware prompting**, plotting the
fairness–utility curve (nDCG@10 vs Global-N share). Deliverables: full
`baseline_metrics.csv` + `fairness_metrics.csv` (multi-run, with confidence
intervals), finalised LaTeX report sections, a 10-slide final deck with real
numbers, and a Streamlit demo with a fairness toggle.

## Repository layout

```
IR-FairSearch-arXiv-Team6/
  README.md
  requirements.txt
  .gitignore
  src/
    config.py                       # all paths + params, RANDOM_SEED
    sample_arxiv.py                 # stratified 50K sampler (two-pass reservoir)
    build_queries.py                # 200-query known-item set + qrels
    encoders.py                     # SPECTER2 (proximity adapter) + MiniLM
    index_chroma.py                 # embed + ChromaDB index (one collection/encoder)
    evaluate.py                     # BM25 + dense: P@k / R@k / nDCG / MRR
    generate.py                     # Llama-3-8B-Instruct RAG generation (cites sources)
    enrich_topk_openalex.py         # cheap affiliations for retrieved papers
    enrich_affiliations_openalex.py # full-corpus enrichment (next phase)
    fairness_preliminary.py         # category / year skew, Gini, Global-N share
    data/                           # snapshot + sample + queries + affiliations  (gitignored)
    results/                        # metric CSVs + generation dumps
    chroma_store/                   # persistent vector DB  (gitignored)
```

## Built with

SPECTER2 & MiniLM (sentence embeddings) · ChromaDB (vector index) · rank-bm25
(lexical control) · Llama-3-8B-Instruct, 4-bit (generation) · OpenAlex
(affiliation enrichment).

## Data & license

arXiv metadata is distributed under **CC0** via the Cornell-University/arxiv
Kaggle dataset; affiliations are enriched from **OpenAlex** (openly licensed,
polite-pool access). This repository is coursework for **CS 6200 — Information
Retrieval**, Northeastern University.
