# FairSearch-arXiv — Baseline RAG Pipeline (Project Update 1)

Audit pipeline for institutional bias in academic Retrieval-Augmented
Generation over the Cornell arXiv corpus. This repo covers **Project Update 1**:
stratified sampling, a naive RAG baseline (SPECTER2 / MiniLM + ChromaDB +
Llama-3-8B-Instruct), a BM25 control, Precision/Recall evaluation, and
**preliminary fairness observations**.

## Pipeline at a glance

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
            (Slide 7: cat/year skew, Gini, Global-N share)
```

## 1. Get the data

The metadata file is ~4 GB of newline-delimited JSON.

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv -p src/data --unzip
# yields src/data/arxiv-metadata-oai-snapshot.json
```

## 2. Install

```bash
pip install -r requirements.txt
```
A GPU is strongly recommended. SPECTER2 encoding of 50K papers takes minutes
on GPU vs ~1–2h on CPU. Llama-3-8B-Instruct in 4-bit needs ~6 GB VRAM and is
a gated Hugging Face model — run `huggingface-cli login` once before
`generate.py`.

## 3. Run

```bash
cd src
python sample_arxiv.py             # stratified 50K sample
python build_queries.py            # query set + qrels
python index_chroma.py             # embed + index both encoders
python evaluate.py                 # BM25 + dense P@k / R@k -> Table 1 / Slide 6

# RAG generation step (closes the loop on the "RAG" label):
python generate.py --n 20          # demo on first 20 queries (fast)

# Preliminary fairness observations for Slide 7:
python enrich_topk_openalex.py     # affiliations for top-K retrieved papers (set MAILTO!)
python fairness_preliminary.py     # category/year skew, Gini, Global-N share
```

`evaluate.py` prints a Markdown table — paste into **Table 1** of `main.tex`
and **Slide 6** of the deck. `fairness_preliminary.py` prints a second table
for **Slide 7**.

## The relevance-labelling decision (read this)

arXiv has **no relevance judgments**, so Precision/Recall are undefined until
*you* define "relevant." We use a **known-item** protocol: each query targets
one source paper; a hit means that paper appears in the top-k. Two ways to
build queries (set `QUERY_METHOD` in `config.py`):

- `heuristic` (default, no API): an abstract snippet is the query. Reproducible
  but somewhat easy because the snippet overlaps the document — will compress
  the BM25-vs-dense gap.
- `llm` (recommended for the final report): generate a natural-language
  question the paper answers (needs `ANTHROPIC_API_KEY`). Harder, more
  realistic, separates dense from lexical retrieval more honestly.

Document this choice in the Dataset/Experiment sections — graders look for it.

## What "preliminary fairness" means here

Full EED / demographic-parity / citation-share need a CWUR-tier and Global N/S
join on enriched affiliations — that's the Week 9–10 work. For Project
Update 1 (Slide 7), we report cheap proxy signals that are computable from
the baseline run + a *small* OpenAlex enrichment (only top-K retrieved
papers, not the whole corpus):

  - **Category skew**: KL divergence between retrieved-category distribution
    and corpus-category distribution; identifies whether the retriever
    over-concentrates on e.g. cs.LG / cs.CL.
  - **Year skew**: same idea over publication years; detects recency bias.
  - **Retrieval concentration (Gini)**: across the full query set, how many
    distinct papers ever appear in any top-K, and how concentrated is
    retrieval on a small popular set?
  - **Global-N share** (if `topk_affiliations.jsonl` exists): % of retrieved
    slots whose authors are at Global-N institutions, by OpenAlex country.

These are interpretable, comparable across retrievers, and give Slide 7 real
data instead of a placeholder.

## ⚠️ Affiliations are NOT in the arXiv snapshot

The Kaggle snapshot provides `authors_parsed` (names only) — **no affiliation
field.** We enrich from OpenAlex (free, no API key, polite-pool MAILTO).
`enrich_topk_openalex.py` runs the *cheap* version (top-K only, ~15 min);
`enrich_affiliations_openalex.py` runs the full corpus enrichment for the
final report (overnight, ~50K papers).

## Layout

```
fairsearch-arxiv/
  requirements.txt
  README.md
  src/
    config.py                       # all paths + params
    sample_arxiv.py                 # stratified 50K sampler
    build_queries.py                # qrels / query set
    encoders.py                     # SPECTER2 (adapters) + MiniLM
    index_chroma.py                 # embed + ChromaDB index
    evaluate.py                     # BM25 + dense, P@k / R@k / nDCG / MRR
    generate.py                     # Llama-3-8B-Instruct RAG generation
    enrich_topk_openalex.py         # cheap affiliations for retrieved papers
    enrich_affiliations_openalex.py # FULL corpus enrichment (next phase)
    fairness_preliminary.py         # Slide 7: category/year/Gini/Global-N
  data/                             # snapshot + sample + queries + affiliations
  results/                          # metrics + generations + fairness CSV
  chroma_store/                     # persistent vector DB
```
