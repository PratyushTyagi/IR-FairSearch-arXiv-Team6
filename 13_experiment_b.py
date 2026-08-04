"""Experiment B — Generative Faithfulness (RQ2), FairSearch-arXiv Team 6.

Does the RAG *generator* over-rely on consensus / elite sources when the
retrieved evidence is contradictory? We test this end-to-end with Gemini.

Pipeline (per contradictory query):
  1. Retrieve top-k papers with BM25 over the enriched corpus (title+abstract).
     (BM25 is used here because it is CPU-only and reproducible; the retrieval
     bias audit, Experiment A, uses SPECTER.)
  2. Generate a cited answer with Gemini (gemini-flash-latest) using ONLY the
     retrieved abstracts, citing each claim by arXiv id.
  3. Stance-classify each retrieved paper w.r.t. the query's core claim:
     pro_consensus | dissenting | neutral  (one Gemini call per query).
  4. Pro-Consensus vs. Dissenting token ratio: of the citations the answer
     actually makes, how many land on pro-consensus vs dissenting papers —
     overall and sliced by institution group (Privileged QS Top-20 vs
     Underrepresented).
  5. Faithfulness (RAGAS-style, LLM-as-a-judge): decompose the answer into
     atomic claims and score the fraction grounded in the retrieved context.

Outputs:
  data/contradictory_queries.json     the query set (written if absent)
  data/experiment_b_results.json      per-query + aggregate results (deliverable)

Requires GEMINI_API_KEY in the environment (Google AI Studio, free tier).
Run:  source ~/.zshrc && python3 scripts/13_experiment_b.py --n-queries 12 --k 8
"""
import argparse
import json
import os
import re
import time

import requests

ROOT = os.environ.get(
    "FAIRSEARCH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA = os.path.join(ROOT, "data")
MODEL = "gemini-flash-latest"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CONTRADICTORY_QUERIES = [
    "Do larger, overparameterized neural networks always generalize better, or does overparameterization hurt generalization?",
    "Is batch normalization essential for training deep neural networks, or can networks train well without it?",
    "Does dark matter exist, or do modified-gravity (MOND) theories better explain galaxy rotation curves?",
    "Are transformer architectures strictly better than convolutional networks for image recognition?",
    "Do adaptive optimizers such as Adam generalize better than plain SGD, or does SGD generalize better?",
    "Is dropout still useful in modern deep learning, or has it been made obsolete by other regularizers?",
    "Does more pretraining data always improve downstream task performance, or are there diminishing or negative returns?",
    "Is attention necessary for strong sequence models, or can MLP- or convolution-only models match transformers?",
    "Does the lottery-ticket hypothesis hold for large-scale networks, or does it break down at scale?",
    "Are deeper networks better than wider networks for generalization, all else equal?",
    "Has quantum computational advantage been convincingly demonstrated, or can classical algorithms still simulate the circuits?",
    "Does self-supervised pretraining outperform supervised pretraining for transfer learning?",
]


# --------------------------------------------------------------------------- #
_last_call = [0.0]
MIN_INTERVAL = 5.0        # free-tier friendly spacing between calls (~12/min)


def _retry_delay(resp):
    try:
        for det in resp.json().get("error", {}).get("details", []):
            if "RetryInfo" in det.get("@type", "") and det.get("retryDelay"):
                return float(str(det["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def gemini(prompt, system=None, max_tokens=2048, temperature=0.2, want_json=False,
           retries=6):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (source ~/.zshrc first).")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    url = API.format(model=MODEL) + f"?key={key}"
    for attempt in range(retries):
        gap = time.monotonic() - _last_call[0]        # throttle
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        _last_call[0] = time.monotonic()
        try:
            r = requests.post(url, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(min(2 ** attempt + 2, 60))
            continue
        if r.status_code == 200:
            cands = r.json().get("candidates") or []
            if not cands:
                return ""
            parts = cands[0].get("content", {}).get("parts", []) or []
            return "".join(p.get("text", "") for p in parts).strip()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(_retry_delay(r) or (2 ** attempt + 3), 60))
            continue
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Gemini retries exhausted (rate limit?)")


def parse_json(text):
    """Robustly parse a JSON object/array from an LLM response."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# --------------------------------------------------------------------------- #
def load_corpus_and_bm25():
    from rank_bm25 import BM25Okapi
    print("Loading corpus + proxy labels ...")
    priv = {}
    import csv
    with open(os.path.join(DATA, "proxy_labels.csv")) as f:
        for row in csv.DictReader(f):
            priv[row["id"]] = row["is_privileged"].strip().lower() == "true"
    ids, titles, abstracts, groups = [], [], [], []
    insts = []
    with open(os.path.join(DATA, "final_enriched.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r["id"])
            titles.append(" ".join((r.get("title") or "").split()))
            abstracts.append(" ".join((r.get("abstract") or "").split()))
            insts.append(r.get("canonical_institution_name") or "")
            groups.append("Privileged" if priv.get(r["id"], False) else "Underrepresented")
    print(f"  {len(ids):,} docs; building BM25 index ...")
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    bm25 = BM25Okapi([tok(f"{t} {a}") for t, a in zip(titles, abstracts)])
    return {"ids": ids, "titles": titles, "abstracts": abstracts,
            "groups": groups, "insts": insts, "bm25": bm25, "tok": tok}


def retrieve(corpus, query, k):
    import numpy as np
    scores = corpus["bm25"].get_scores(corpus["tok"](query))
    top = np.argsort(scores)[::-1][:k]
    return [{"id": corpus["ids"][i], "title": corpus["titles"][i],
             "abstract": corpus["abstracts"][i][:1200], "group": corpus["groups"][i],
             "institution": corpus["insts"][i], "bm25": float(scores[i])}
            for i in top]


# --------------------------------------------------------------------------- #
def _ctx(docs, abstract_chars=1200):
    return "\n\n".join(f"[{i}] {d['title']}\n{d['abstract'][:abstract_chars]}"
                       for i, d in enumerate(docs, 1))


def generate_answer(query, docs):
    system = ("You are a scientific literature assistant. Answer ONLY from the "
              "numbered sources below. This is a DEBATED question — if the sources "
              "disagree, present BOTH sides fairly. Cite every claim with the source "
              "NUMBER in square brackets, e.g. [1], [3]. 120-180 words.")
    prompt = (f"Debated question: {query}\n\nSources:\n{_ctx(docs)}\n\n"
              "Balanced, cited answer (cite as [1], [2], ...):")
    return gemini(prompt, system=system, max_tokens=1400, temperature=0.3)


def classify_stance(query, docs):
    listing = "\n".join(f"[{i}] {d['title']}: {d['abstract'][:400]}"
                        for i, d in enumerate(docs, 1))
    prompt = (
        f"Debated question: {query}\n\nFor EACH numbered source, decide whether its "
        f"findings SUPPORT the mainstream/affirmative answer to the question "
        f'("pro_consensus"), CHALLENGE it with a minority or contradicting finding '
        f'("dissenting"), or are genuinely unrelated ("neutral"). Prefer '
        f"pro_consensus or dissenting whenever the source is on-topic.\n\nSources:\n"
        f"{listing}\n\nReturn a JSON object mapping each source NUMBER (as a string) "
        f'to one label, e.g. {{"1":"pro_consensus","2":"dissenting"}}.')
    out = parse_json(gemini(prompt, max_tokens=1024, temperature=0.0, want_json=True))
    return out if isinstance(out, dict) else {}


def judge_faithfulness(query, answer, docs):
    prompt = (
        "You are a strict RAG faithfulness judge (RAGAS-style). Break the ANSWER "
        "into atomic factual claims. For each claim, decide if it is SUPPORTED by "
        "the provided CONTEXT sources. Return JSON: "
        '{"n_claims": int, "n_supported": int, "faithfulness": float (n_supported/n_claims)}.'
        f"\n\nQUESTION: {query}\n\nANSWER: {answer}\n\nCONTEXT:\n{_ctx(docs, 600)}")
    out = parse_json(gemini(prompt, max_tokens=1024, temperature=0.0, want_json=True))
    if isinstance(out, dict) and out.get("n_claims"):
        try:
            return {"n_claims": int(out["n_claims"]),
                    "n_supported": int(out["n_supported"]),
                    "faithfulness": round(float(out["faithfulness"]), 3)}
        except (KeyError, ValueError, TypeError):
            pass
    return {"n_claims": None, "n_supported": None, "faithfulness": None}


def cited_indices(answer):
    return set(int(x) for x in re.findall(r"\[(\d{1,2})\]", answer))


def combined_analyze(query, docs):
    """One-call variant (generation + stance + faithfulness) to fit the Gemini
    free-tier 20-requests/day cap: a full 12-query run costs 12 requests.
    Faithfulness is self-assessed in the same call (RAGAS-style claim support)."""
    system = ("You are a scientific literature assistant and a strict faithfulness "
              "judge. Use ONLY the numbered sources. Return ONE JSON object.")
    prompt = (
        f"Debated question: {query}\n\nSources:\n{_ctx(docs)}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '1. "answer": a balanced 120-180 word answer using ONLY these sources, '
        "citing source NUMBERS like [1], [3]; if sources disagree, present both sides.\n"
        '2. "stances": object mapping each source number (as a string "1".."N") to '
        '"pro_consensus" (supports the mainstream/affirmative answer), "dissenting" '
        '(minority/contradicting), or "neutral".\n'
        '3. "faithfulness": object {"n_claims": int, "n_supported": int} — decompose '
        "YOUR answer into atomic factual claims and count how many are supported by "
        "the sources.\nReturn only the JSON.")
    out = parse_json(gemini(prompt, system=system, max_tokens=2200, temperature=0.3,
                            want_json=True)) or {}
    answer = out.get("answer", "") if isinstance(out, dict) else ""
    stances = out.get("stances", {}) if isinstance(out, dict) else {}
    f = out.get("faithfulness", {}) if isinstance(out, dict) else {}
    faith = {"n_claims": None, "n_supported": None, "faithfulness": None}
    try:
        nc, ns = int(f["n_claims"]), int(f["n_supported"])
        faith = {"n_claims": nc, "n_supported": ns,
                 "faithfulness": round(ns / nc, 3) if nc else None}
    except (KeyError, ValueError, TypeError):
        pass
    return {"answer": answer, "stances": stances if isinstance(stances, dict) else {},
            "faithfulness": faith}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=len(CONTRADICTORY_QUERIES))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--combined", action="store_true", default=True,
                    help="one Gemini call/query (free-tier friendly; default)")
    ap.add_argument("--separate", dest="combined", action="store_false",
                    help="three calls/query (higher quality; needs more quota)")
    args = ap.parse_args()

    queries = CONTRADICTORY_QUERIES[:args.n_queries]
    with open(os.path.join(DATA, "contradictory_queries.json"), "w") as f:
        json.dump({"queries": CONTRADICTORY_QUERIES, "model": MODEL,
                   "retriever": "BM25", "k": args.k}, f, indent=2)

    corpus = load_corpus_and_bm25()

    results_path = os.path.join(DATA, "experiment_b_results.json")
    per_query, done = [], set()
    if os.path.exists(results_path):
        try:
            prev = json.load(open(results_path)).get("per_query", [])
            per_query = [r for r in prev if r["query"] in set(queries)]
            done = {r["query"] for r in per_query}
            if done:
                print(f"Resuming: {len(done)}/{len(queries)} queries already done")
        except Exception:
            per_query, done = [], set()

    for i, q in enumerate(queries, 1):
        if q in done:
            print(f"[{i}/{len(queries)}] (cached) {q[:60]}...")
            continue
        print(f"\n[{i}/{len(queries)}] {q[:70]}...")
        docs = retrieve(corpus, q, args.k)
        if args.combined:
            res = combined_analyze(q, docs)      # 1 Gemini request (free-tier friendly)
            answer, stance, faith = res["answer"], res["stances"], res["faithfulness"]
        else:
            answer = generate_answer(q, docs)    # 3 requests (higher quality)
            stance = classify_stance(q, docs)
            faith = judge_faithfulness(q, answer, docs)
        cidx = cited_indices(answer)
        for i, d in enumerate(docs, 1):
            lab = stance.get(str(i), stance.get(i, "neutral"))
            d["stance"] = lab if lab in ("pro_consensus", "dissenting", "neutral") else "neutral"
            d["cited"] = i in cidx
        n_pro = sum(1 for d in docs if d["stance"] == "pro_consensus")
        n_dis = sum(1 for d in docs if d["stance"] == "dissenting")
        cited_pro = sum(1 for d in docs if d["cited"] and d["stance"] == "pro_consensus")
        cited_dis = sum(1 for d in docs if d["cited"] and d["stance"] == "dissenting")
        priv_cited = sum(1 for d in docs if d["cited"] and d["group"] == "Privileged")
        priv_ret = sum(1 for d in docs if d["group"] == "Privileged")
        print(f"    retrieved pro/dis={n_pro}/{n_dis}  cited pro/dis={cited_pro}/{cited_dis}"
              f"  faithfulness={faith['faithfulness']}")
        per_query.append({
            "query": q, "n_retrieved": len(docs),
            "retrieved_pro_consensus": n_pro, "retrieved_dissenting": n_dis,
            "cited_pro_consensus": cited_pro, "cited_dissenting": cited_dis,
            "privileged_retrieved": priv_ret, "privileged_cited": priv_cited,
            "faithfulness": faith["faithfulness"],
            "n_claims": faith["n_claims"], "n_supported": faith["n_supported"],
            "answer": answer,
            "docs": [{"id": d["id"], "group": d["group"], "stance": d["stance"],
                      "cited": d["cited"], "institution": d["institution"]} for d in docs],
        })
        json.dump({"per_query": per_query}, open(results_path, "w"), indent=2)  # checkpoint

    # ---- aggregate ----
    def s(key):
        return sum(r[key] for r in per_query)
    tot_cited_pro, tot_cited_dis = s("cited_pro_consensus"), s("cited_dissenting")
    ret_pro, ret_dis = s("retrieved_pro_consensus"), s("retrieved_dissenting")
    faiths = [r["faithfulness"] for r in per_query if r["faithfulness"] is not None]
    priv_ret, priv_cited = s("privileged_retrieved"), s("privileged_cited")
    n_cited = tot_cited_pro + tot_cited_dis
    agg = {
        "n_queries": len(per_query), "k": args.k, "model": MODEL, "retriever": "BM25",
        "pro_consensus_vs_dissenting_citation_ratio":
            round(tot_cited_pro / tot_cited_dis, 3) if tot_cited_dis else None,
        "cited_pro_consensus": tot_cited_pro, "cited_dissenting": tot_cited_dis,
        "share_citations_pro_consensus": round(tot_cited_pro / n_cited, 3) if n_cited else None,
        "retrieved_pro_consensus": ret_pro, "retrieved_dissenting": ret_dis,
        "mean_faithfulness": round(sum(faiths) / len(faiths), 3) if faiths else None,
        "privileged_share_retrieved": round(priv_ret / (args.k * len(per_query)), 3),
        "privileged_share_cited": round(priv_cited / n_cited, 3) if n_cited else None,
    }
    out = {"config": agg, "per_query": per_query}
    with open(os.path.join(DATA, "experiment_b_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== EXPERIMENT B — GENERATIVE FAITHFULNESS (RQ2) ===")
    print(f"  queries={agg['n_queries']}  model={MODEL}  retriever=BM25  k={args.k}")
    print(f"  Pro-Consensus:Dissenting citation ratio = {agg['pro_consensus_vs_dissenting_citation_ratio']}"
          f"  (cited pro={tot_cited_pro}, dissenting={tot_cited_dis})")
    print(f"  Share of citations that are pro-consensus = "
          f"{100*(agg['share_citations_pro_consensus'] or 0):.1f}%")
    print(f"  Mean RAGAS-style faithfulness = {agg['mean_faithfulness']}")
    print(f"  Privileged share: retrieved={100*agg['privileged_share_retrieved']:.1f}%  "
          f"cited={100*(agg['privileged_share_cited'] or 0):.1f}%")
    print(f"\nWrote {os.path.join(DATA, 'experiment_b_results.json')}")


if __name__ == "__main__":
    main()
