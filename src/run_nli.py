"""Specialised factual-verification baseline: sentence-level NLI in the style of SummaC-ZS.

For every answer sentence the two evidence chunks with the highest MiniLM cosine are used as
premises. Scores (higher = more supported):
  nli_mean_ec : mean over sentences of max over chunks of P(entail) - P(contradict)   (primary)
  nli_min_e   : min  over sentences of max over chunks of P(entail)
  nli_max_c   : -max over sentences/chunks of P(contradict)
Reference-based data: premise = "The answer to the question '<q>' is <reference>.", hypothesis = answer.
Runs on calibration and test splits and on the counterfactual set (the NLI model is never trained here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from common import CFG, PROC, RESULTS, chunk_evidence, read_jsonl, sentences

MAX_SENTS, MAX_CHUNKS = 40, 400
CACHE = PROC / "nli_cache.pkl"


def key(p, h):
    return hashlib.md5((p + "␞" + h).encode("utf8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = CFG["nli"]["model"]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).eval().float()   # checkpoint is fp16; fp16 is ~13x slower on CPU
    load_s = time.perf_counter() - t0
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    i_e, i_c = lab["entailment"], lab["contradiction"]

    ex = [e for e in read_jsonl(PROC / "examples.jsonl") if e["split"] in ("calib", "test")]
    ex += read_jsonl(PROC / "counterfactuals.jsonl")
    retrieval = json.loads((PROC / "retrieval.json").read_text())
    chunks_of, plan, pairs = {}, {}, {}
    for e in ex:
        if e["target"] == "correctness":
            prem = [f"The answer to the question '{e['question']}' is {r}." for r in e["reference"][:3]]
            plan[e["id"]] = [[(p, e["answer"]) for p in prem]]
        else:
            if e["id"] not in retrieval:
                continue                                  # e.g. paraphrases not yet scored by score_cheap.py
            if e["evidence"] not in chunks_of:
                chunks_of[e["evidence"]] = chunk_evidence(e["evidence"])[:MAX_CHUNKS]
            ch = chunks_of[e["evidence"]]
            sents = sentences(e["answer"])[:MAX_SENTS] or [e["answer"]]
            plan[e["id"]] = [[(ch[j], s) for j in top[: CFG["nli"]["top_k_chunks"]]] for s, top in zip(sents, retrieval[e["id"]])]
        for group in plan[e["id"]]:
            for p, h in group:
                pairs.setdefault(key(p, h), (p, h))
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    todo = [(k, v) for k, v in pairs.items() if k not in cache]
    todo.sort(key=lambda kv: len(kv[1][0]) + len(kv[1][1]))
    if a.limit:
        todo = todo[: a.limit]
    print(f"pairs total {len(pairs)} cached {len(pairs) - len(todo)} todo {len(todo)}", flush=True)
    t0 = time.perf_counter()
    bs, n_tok = 32, 0
    for i in range(0, len(todo), bs):
        batch = todo[i: i + bs]
        enc = tok([v[0] for _, v in batch], [v[1] for _, v in batch], truncation=True,
                  max_length=CFG["nli"]["max_length"], padding=True, return_tensors="pt")
        n_tok += int(enc["attention_mask"].sum())
        with torch.inference_mode():
            pr = torch.softmax(model(**enc).logits, dim=-1).numpy()
        for (k, _), row in zip(batch, pr):
            cache[k] = (float(row[i_e]), float(row[i_c]))
        if (i // bs) % 50 == 0:
            print(f"{i}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
        if (i // bs) % 400 == 399:
            pickle.dump(cache, open(CACHE, "wb"))
    secs = time.perf_counter() - t0
    pickle.dump(cache, open(CACHE, "wb"))
    rows = {}
    for eid, groups in plan.items():
        if any(key(p, h) not in cache for g in groups for p, h in g):
            continue
        ent = [max(cache[key(p, h)][0] for p, h in g) for g in groups]
        ec = [max(cache[key(p, h)][0] - cache[key(p, h)][1] for p, h in g) for g in groups]
        con = max(cache[key(p, h)][1] for g in groups for p, h in g)
        rows[eid] = dict(nli_mean_ec=float(np.mean(ec)), nli_min_e=float(min(ent)), nli_max_c=float(-con),
                         nli_pairs=sum(len(g) for g in groups))
    pd.DataFrame.from_dict(rows, orient="index").to_parquet(RESULTS / "scores_nli.parquet")
    if todo:
        (RESULTS / "cost_nli_bulk.json").write_text(json.dumps(
            dict(model=name, load_s=load_s, pairs=len(todo), tokens=n_tok, seconds=secs, threads=a.threads), indent=1))
    print("done", len(rows), "examples;", len(todo), "pairs in", round(secs), "s")


if __name__ == "__main__":
    main()
