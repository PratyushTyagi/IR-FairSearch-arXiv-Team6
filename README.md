# IR-FairSearch-arXiv-Team6

**FairSearch-arXiv: Evaluating and Mitigating Bias in Academic Retrieval-Augmented Generation**
CS 6200: Information Retrieval · Northeastern University · Summer 2026 · **Team 6**

**Team:** Aarushi Kaushik · Pavani Jain · Pratyush Tyagi · Shweta Pattanaik

---

## Overview

We audit a Retrieval-Augmented Generation (RAG) pipeline for academic search
over the [Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(2M+ preprints), where dense retrieval and LLM synthesis may *multiplicatively*
amplify institutional homophily. We decompose the bias into retrieval-stage and
synthesis-stage effects, then characterise the fairness–utility trade-off of
FA*IR re-ranking and fairness-aware prompting.

## Research Questions

- **RQ1 — Retrieval Parity.** Does dense retrieval over-represent top-tier institutions versus BM25?
- **RQ2 — Synthesis Neutrality.** Does LLM synthesis add citation skew given a balanced top-*k*?
- **RQ3 — Fairness–Utility.** What EED reduction is achievable at ≤ 5% nDCG@10 cost?

## Stack

| Layer | Choice |
|---|---|
| Dataset | Cornell arXiv on Kaggle — 2M+ papers, JSONL, weekly update |
| Sample | Stratified random 50K by primary category × year (2020–25) |
| Embeddings | SPECTER2 (primary), all-MiniLM-L6-v2 (baseline) |
| Vector store | ChromaDB (Qdrant fallback past 250K vectors) |
| Generator | Llama-3-8B (Instruct) |
| Retrieval eval | Expected Exposure Loss / Disparity (EEL, EED); demographic parity@*k* |
| Synthesis eval | Citation-share-in-synth across institution tiers |
| Mitigation | FA*IR re-ranker, MMR diversification, fairness-aware prompting |
| Significance | 10K bootstrap over the query set |
| Protected attributes | Institution tier (top-50 CWUR vs. rest), region (Global N / S) |
| UI | Streamlit demo with fairness toggle |

## Current Deliverables

| File | What it is |
|---|---|
| [`FairSearch-arXiv-Proposal.pdf`](FairSearch-arXiv-Proposal.pdf) | 1-page ACM-format project proposal |
| [`FairSearch-arXiv-Proposal.tex`](FairSearch-arXiv-Proposal.tex) | LaTeX source (Overleaf-ready, `acmart` sigconf) |
| [`FairSearch-arXiv-Proposal-Slides.pptx`](FairSearch-arXiv-Proposal-Slides.pptx) | 5-slide presentation deck |

## Planned Repository Layout

```
IR-FairSearch-arXiv-Team6/
├── report/        ACM-format proposal: PDF + LaTeX source
├── slides/        Presentation deck (.pptx)
├── data/          arXiv snapshot + sampled subsets (gitignored)
├── notebooks/     EDA, exploratory analyses
├── src/
│   ├── data/         sampling, affiliation extraction, ROR/CWUR mapping
│   ├── retrieval/    SPECTER2 embed + ChromaDB index
│   ├── synthesis/    Llama-3-8B RAG synthesis
│   ├── eval/         EEL, EED, parity@k, citation-share
│   └── mitigation/   FA*IR, MMR, fairness-aware prompts
├── app/           Streamlit demo
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

## Timeline

| Weeks | Phase | Exit gate |
|---|---|---|
| W3–W5   | Data + EDA           | ≥ 90% affiliations resolved |
| W6–W8   | Baseline RAG         | End-to-end query ≤ 3 s |
| W9–W12  | RQ1/RQ2 + Mitigation | Significant (p<0.05) result |
| W13–W14 | UI + Deliverables    | Report, demo, slides |

## Quickstart (once code lands)

```bash
git clone https://github.com/PratyushTyagi/IR-FairSearch-arXiv-Team6.git
cd IR-FairSearch-arXiv-Team6
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# After placing the Kaggle arXiv snapshot in data/raw/:
python -m src.data.sample --n 50000 --out data/sample.parquet
python -m src.retrieval.build_index --in data/sample.parquet
python -m app.streamlit_app
```

## Background References

The proposal builds directly on six prior works:

1. Singh & Joachims (KDD '18) — Fairness of Exposure in Rankings
2. Biega, Gummadi, Weikum (SIGIR '18) — Equity of Attention
3. Zehlike et al. (CIKM '17) — FA*IR
4. Diaz et al. (CIKM '20) — Expected Exposure (EEL/EED)
5. Rekabsaz & Schedl (SIGIR '20) — Neural Ranking Models & Gender Bias
6. Ekstrand et al. (FAccT '18) — Demographic Biases in Recommenders

Full citations in the proposal PDF.

## License

[MIT](LICENSE).
