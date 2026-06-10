"""
generate.py
Closes the loop on the "RAG" label: retrieve top-k papers for each query
with a dense retriever, then ask Llama-3-8B-Instruct (4-bit) to answer the
query using ONLY the retrieved abstracts as context.

Output: results/generations_<model>_k<k>.jsonl  -- one JSON object per query
        with the query, the retrieved doc ids, whether the source paper was
        in the top-k, and the generated answer. Use this for the Slide 5
        "naive RAG" demo and Appendix examples in the report.

Requires HF login for Meta-Llama-3-8B-Instruct (gated model).

Run:  python src/generate.py                  # default: specter2, k=5, all queries
      python src/generate.py --k 10 --n 20    # k=10, first 20 queries (demo)
      python src/generate.py --model minilm
"""
import argparse
import json

import chromadb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config as C
from encoders import get_encoder


LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = (
    "You are a scientific research assistant. Answer the user's question "
    "using ONLY the provided arXiv abstracts. Cite each claim with the "
    "paper id in square brackets, e.g. [2305.12345]. If the abstracts do "
    "not contain enough information, say so explicitly."
)


def load_llama():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return tok, model


def build_context(docs):
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['abstract']}" for d in docs)


def generate(tok, model, query, docs, max_new_tokens=300):
    user_msg = f"Question: {query}\n\nAbstracts:\n{build_context(docs)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="specter2", choices=["specter2", "minilm"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=0, help="limit to first N queries (0 = all)")
    args = ap.parse_args()

    sample_by_id = {}
    with open(C.SAMPLE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sample_by_id[r["id"]] = r

    with open(C.QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    if args.n > 0:
        queries = queries[:args.n]

    client = chromadb.PersistentClient(path=C.CHROMA_DIR)
    col = client.get_collection(args.model)
    enc = get_encoder(args.model)

    print("Loading Llama-3-8B-Instruct (4-bit) ...")
    tok, model = load_llama()

    out_path = C.RESULTS_DIR / f"generations_{args.model}_k{args.k}.jsonl"
    print(f"Generating {len(queries)} answers -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            vec = enc.encode([""], [q["query"]])[0].tolist()
            res = col.query(query_embeddings=[vec], n_results=args.k, include=[])
            retrieved_ids = res["ids"][0]
            docs = [sample_by_id[pid] for pid in retrieved_ids if pid in sample_by_id]
            answer = generate(tok, model, q["query"], docs)
            f.write(json.dumps({
                "qid": q["qid"],
                "query": q["query"],
                "retriever": args.model,
                "k": args.k,
                "retrieved_ids": retrieved_ids,
                "source_paper_in_topk": q["relevant_ids"][0] in retrieved_ids,
                "answer": answer,
            }) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(queries)}")
    print("Done.")


if __name__ == "__main__":
    main()
