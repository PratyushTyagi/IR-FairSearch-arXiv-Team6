"""
build_queries.py
Create the query set and relevance judgments (qrels).

arXiv ships NO relevance labels, so we must define what "relevant" means.
This is the single most important design decision for Precision/Recall, and
it is the thing the Slide-6 "labelling method" placeholder is asking for.

We use a *known-item* protocol: each query targets exactly one paper, and a
retrieval is correct if that source paper appears in the top-k. With one
relevant doc per query, Precision@k and Recall@k are both well defined and
make the dense-vs-BM25 comparison interpretable.

Two query-construction methods:
  - "heuristic" (default, zero dependencies): use a snippet of the abstract
    as the query. Reproducible; differentiates lexical (BM25) vs semantic
    (dense) retrieval. Somewhat easy because the snippet overlaps the doc.
  - "llm": generate a natural-language *question* the paper answers. Harder
    and more realistic; needs an API key (ANTHROPIC_API_KEY). Recommended for
    the final report.

Run:  python src/build_queries.py
"""
import json
import os
import random

import config as C


def load_sample():
    with open(C.SAMPLE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def heuristic_query(rec, n_tokens=30):
    """First ~n_tokens of the abstract as a known-item query."""
    return " ".join(rec["abstract"].split()[:n_tokens])


def llm_query(rec, client):
    """Ask an LLM for a question this paper would answer (no title leakage)."""
    prompt = (
        "Write ONE concise natural-language search question (max 20 words) that "
        "this paper would answer. Do not copy the title. Abstract:\n\n"
        f"{rec['abstract'][:1500]}"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def main():
    random.seed(C.RANDOM_SEED)
    sample = load_sample()
    query_papers = random.sample(sample, min(C.N_QUERIES, len(sample)))

    client = None
    if C.QUERY_METHOD == "llm":
        import anthropic  # pip install anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    queries = []
    for i, rec in enumerate(query_papers):
        q = llm_query(rec, client) if C.QUERY_METHOD == "llm" else heuristic_query(rec)
        queries.append({
            "qid": f"q{i:04d}",
            "query": q,
            "relevant_ids": [rec["id"]],   # known-item: exactly one relevant doc
            "source_category": rec["primary_category"],
        })

    with open(C.QUERIES_FILE, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(queries)} queries ({C.QUERY_METHOD}) -> {C.QUERIES_FILE}")


if __name__ == "__main__":
    main()
