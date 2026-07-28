"""FairSearch-arXiv — Streamlit Fairness Scorecard.

An interactive diagnostic for institutional bias in SPECTER retrieval over the
arXiv corpus. Pick one of the 100 standardized audit queries, choose a ranking
method (Baseline / Fair-Top-K / MMR), and read a per-query *fairness scorecard*:
the Privileged (QS Top-20) share in the top-k, Statistical Parity Difference,
NDCG@k and Precision@k, plus the ranked results tagged by institution group.

Fair-Top-K exposes its representation target as a slider, so you can watch the
elite share fall (and utility barely move) as you tighten the fairness knob.

Runs from small cached files only (no GPU / no LLM / no embeddings):
  scorecard_docs.csv, specter_retrieved_topk.csv, queries_100.jsonl
  (mmr_topk.csv optional, enables the MMR view).

Run:  streamlit run app.py           (from the repo root)
      streamlit run scripts/app.py    (from the project root)
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard as sc

st.set_page_config(page_title="FairSearch-arXiv Fairness Scorecard",
                   page_icon="⚖️", layout="wide")


@st.cache_data(show_spinner=False)
def _load():
    return sc.load()


queries, pools, docs, mmr = _load()
qtext = {q["qid"]: q["text"] for q in queries}

st.title("⚖️ FairSearch-arXiv — Fairness Scorecard")
st.caption("Institutional-bias diagnostic for academic RAG retrieval. "
           "Group = QS-2026 Top-20 institution (**Privileged**) vs. everyone else "
           f"(**Underrepresented**). Corpus base rate: {100*sc.CORPUS_PRIV_RATE:.2f}% Privileged.")

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Query")
    qid = st.selectbox("Standardized audit query (n=100)", [q["qid"] for q in queries],
                       format_func=lambda i: f"{i} — {qtext[i][:60]}…")
    st.header("Ranking method")
    methods = ["Baseline (SPECTER)", "Fair-Top-K"] + (["MMR (λ=0.7)"] if mmr else [])
    method_label = st.radio("Method", methods, index=0)
    k = st.slider("Top-k", min_value=3, max_value=10, value=10)
    min_share = sc.CORPUS_UNDER_RATE
    if method_label == "Fair-Top-K":
        min_share = st.slider(
            "Min. Underrepresented share (fairness target)",
            min_value=0.90, max_value=1.00, value=round(sc.CORPUS_UNDER_RATE, 3),
            step=0.005,
            help="Every prefix must hold at least this share of non-elite papers. "
                 "1.00 excludes all elite papers from the top ranks.")
    st.divider()
    st.caption("Baseline & Fair-Top-K are computed live from the cached SPECTER "
               "pool; MMR is a precomputed reference.")

method = {"Baseline (SPECTER)": "baseline", "Fair-Top-K": "fairtopk",
          "MMR (λ=0.7)": "mmr"}[method_label]

pool = pools[qid]
order = sc.rerank(pool, method, min_share=min_share, mmr_order=mmr.get(qid))
base_order = sc.rerank(pool, "baseline")
card = sc.scorecard(pool, order, k)
base_card = sc.scorecard(pool, base_order, k)

# ---------------- header ----------------
st.subheader(f"Query `{qid}`")
st.markdown(f"> {qtext[qid]}")

# ---------------- scorecard tiles ----------------
c1, c2, c3, c4 = st.columns(4)
priv_delta = card["priv_share"] - sc.CORPUS_PRIV_RATE
c1.metric("Privileged share in top-k", f"{100*card['priv_share']:.1f}%",
          delta=f"{100*priv_delta:+.1f} pp vs corpus", delta_color="inverse")
c2.metric("Statistical Parity Diff.", f"{card['spd']:+.2e}",
          help="P(retrieved|Privileged) − P(retrieved|Underrepresented). 0 = parity.")
c3.metric("NDCG@k (utility)", f"{card['ndcg']:.3f}",
          delta=f"{card['ndcg']-base_card['ndcg']:+.3f} vs baseline")
c4.metric("Precision@k", f"{card['precision']:.3f}",
          delta=f"{card['precision']-base_card['precision']:+.3f} vs baseline")

# fairness verdict
if card["priv_share"] <= sc.CORPUS_PRIV_RATE + 1e-9:
    st.success(f"✅ At or below corpus parity — Privileged papers hold "
               f"{card['priv_in_topk']}/{k} slots ({100*card['priv_share']:.1f}%).")
else:
    st.warning(f"⚠️ Elite over-representation — Privileged papers hold "
               f"{card['priv_in_topk']}/{k} slots ({100*card['priv_share']:.1f}%), "
               f"above the {100*sc.CORPUS_PRIV_RATE:.1f}% corpus base rate.")

# ---------------- composition + comparison ----------------
left, right = st.columns([1, 1])
with left:
    st.markdown("**Group composition of top-k**")
    sel = order[:k]
    comp = pd.DataFrame({
        "group": ["Privileged", "Underrepresented"],
        "count": [sum(d["is_priv"] for d in sel), sum(not d["is_priv"] for d in sel)],
    }).set_index("group")
    st.bar_chart(comp, horizontal=True, color="#2563eb")
with right:
    st.markdown("**This method vs. Baseline**")
    st.dataframe(pd.DataFrame({
        "Baseline": [f"{100*base_card['priv_share']:.1f}%", f"{base_card['ndcg']:.3f}",
                     f"{base_card['precision']:.3f}"],
        method_label: [f"{100*card['priv_share']:.1f}%", f"{card['ndcg']:.3f}",
                       f"{card['precision']:.3f}"],
    }, index=["Privileged share", "NDCG@k", "Precision@k"]), use_container_width=True)

# ---------------- ranked results ----------------
st.markdown(f"**Top-{k} results** — 🔴 Privileged (QS Top-20) · 🔵 Underrepresented · ✅ relevant")
rows = []
for rank, d in enumerate(order[:k], 1):
    rows.append({
        "rank": rank,
        "group": "🔴 Privileged" if d["is_priv"] else "🔵 Underrep",
        "relevant": "✅" if d["relevant"] else "",
        "title": (d["title"][:90] + "…") if len(d["title"]) > 90 else d["title"],
        "institution": d["institution"] or "—",
        "arxiv_id": d["doc_id"],
        "score": round(d["score"], 3),
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.caption("SPD uses corpus group sizes (Privileged=1,638 / 49,652). Retrieval is "
           "the cached SPECTER top-100; Fair-Top-K re-ranks it live. "
           "Data: FairSearch-arXiv, CS 6200 Team 6.")
