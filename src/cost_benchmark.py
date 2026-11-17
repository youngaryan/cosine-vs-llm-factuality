"""Latency / compute benchmark on one machine with nothing else running.

A fixed sample of RAGTruth test examples (equal numbers per task, drawn from the small-judge subset)
is scored one example at a time. For each method we record cold-start (model load) time, per-example
warm latency (median, p95), wall seconds per 1,000 evaluations, process CPU seconds, peak RSS and, for
LLM judges, prompt/output tokens and the effect of Ollama's prompt-prefix cache.

  python src/cost_benchmark.py                 # cheap methods + small judge
  python src/cost_benchmark.py --strong        # also benchmark the stronger judge (slow)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from common import CFG, PROC, RESULTS, SEED, chunk_evidence, read_jsonl, sentences

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20 if psutil else None


def summarise(lat, cpu, extra=None):
    lat = np.asarray(lat)
    d = dict(n=len(lat), median_ms=float(np.median(lat) * 1e3), p95_ms=float(np.percentile(lat, 95) * 1e3),
             mean_ms=float(lat.mean() * 1e3), wall_s_per_1000=float(lat.mean() * 1000),
             cpu_s_per_1000=float(cpu / len(lat) * 1000) if cpu is not None else None, rss_mb=rss_mb())
    d.update(extra or {})
    return d


def timed(fn, items):
    lat = []
    c0 = time.process_time()
    for it in items:
        t = time.perf_counter()
        fn(it)
        lat.append(time.perf_counter() - t)
    return lat, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=40)
    ap.add_argument("--strong", action="store_true")
    ap.add_argument("--judge-n", type=int, default=30)
    a = ap.parse_args()
    import torch

    torch.set_num_threads(6)
    sub = json.loads((PROC / "judge_subsets.json").read_text())["small"]["groups"]
    ex = [e for e in read_jsonl(PROC / "examples.jsonl") if e["split"] == "test" and e["group"] in set(sub) and e["dataset"] == "ragtruth"]
    rng = np.random.default_rng(SEED + 5)
    sample = []
    for dom in ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt"]:
        d = [e for e in ex if e["domain"] == dom]
        seen, picked = set(), []
        for i in rng.permutation(len(d)):                      # one answer per source: no evidence reuse
            if d[i]["group"] not in seen:
                seen.add(d[i]["group"])
                picked.append(d[i])
        sample += picked[: a.n_per_task]
    out = dict(hardware=dict(cpu=platform.processor(), machine=platform.machine(), cores_logical=os.cpu_count(),
                             ram_gb=round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
                             gpu="none used (CPU inference only)", os=platform.platform(), torch=torch.__version__,
                             torch_threads=torch.get_num_threads()),
               sample=dict(n=len(sample), per_task=a.n_per_task,
                           mean_answer_words=float(np.mean([len(e["answer"].split()) for e in sample])),
                           mean_evidence_words=float(np.mean([len(e["evidence"].split()) for e in sample]))),
               methods={})

    # ---- lexical features and structured checks
    import score_cheap as SC
    from common import content_tokens, numbers, tokens

    def lexical(e):
        tok = tokens(e["evidence"])
        ct = content_tokens(e["evidence"])
        cache = {e["evidence"]: dict(tok=tok, tokset=set(tok), nums=set(numbers(e["evidence"])), norm=SC.norm_answer(e["evidence"]),
                                     bigrams=set(zip(ct, ct[1:])), chunks=chunk_evidence(e["evidence"]))}
        SC.lexical_evidence(e, cache)

    lat, cpu = timed(lexical, sample)
    out["methods"]["lexical_and_checks"] = summarise(lat, cpu, dict(cold_start_s=0.0, model_calls=0))

    # ---- embedding cosine (evidence chunks + answer sentences encoded per example; and answer-only when evidence is cached)
    for name, hf in [("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("bge_small", "BAAI/bge-small-en-v1.5")]:
        enc = SC.STEncoder(hf)
        enc.encode(["warm up"])

        def full(e, enc=enc):
            S = enc.encode(sentences(e["answer"])[:40] or [e["answer"]])
            C = enc.encode(chunk_evidence(e["evidence"])[:400])
            (S @ C.T).max(1).mean()

        def answer_only(e, enc=enc):
            enc.encode(sentences(e["answer"])[:40] or [e["answer"]])

        lat, cpu = timed(full, sample)
        out["methods"][f"{name}_cosine"] = summarise(lat, cpu, dict(cold_start_s=enc.load_s, model_calls=2,
                                                                     note="encodes evidence chunks and answer sentences"))
        lat, cpu = timed(answer_only, sample)
        out["methods"][f"{name}_cosine_evidence_cached"] = summarise(lat, cpu, dict(cold_start_s=enc.load_s, model_calls=1,
                                                                                      note="evidence embeddings reused"))
        if name == "minilm":
            minilm = enc

    # ---- risk estimator = lexical + MiniLM (3 configurations) + logistic regression
    def router(e):
        lexical(e)
        S = minilm.encode(sentences(e["answer"])[:40] or [e["answer"]])
        C = minilm.encode(chunk_evidence(e["evidence"])[:400])
        W = minilm.encode([e["answer"], e["evidence"]])
        (S @ C.T).max(1)
        float(W[0] @ W[1])

    lat, cpu = timed(router, sample)
    out["methods"]["risk_full"] = summarise(lat, cpu, dict(cold_start_s=minilm.load_s, model_calls=3,
                                                           note="lexical features + MiniLM doc/mean-max/min-max; LR inference is negligible"))

    # ---- BERTScore precision (roberta-base)
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    t = time.perf_counter()
    btok = AutoTokenizer.from_pretrained(CFG["bertscore"]["model"])
    bmod = AutoModel.from_pretrained(CFG["bertscore"]["model"], output_hidden_states=True).eval()
    bload = time.perf_counter() - t

    def bert(e):
        ids = btok(e["evidence"], add_special_tokens=False)["input_ids"]
        wins = [btok.decode(ids[i: i + 384]) for i in range(0, max(1, len(ids)), 352)]
        embs = []
        with torch.inference_mode():
            for i in range(0, len(wins), 16):
                enc_ = btok(wins[i: i + 16], truncation=True, max_length=386, padding=True, return_tensors="pt")
                embs.append(torch.nn.functional.normalize(bmod(**enc_).hidden_states[10], dim=-1))
            enc_ = btok([e["answer"]], truncation=True, max_length=386, return_tensors="pt")
            av = torch.nn.functional.normalize(bmod(**enc_).hidden_states[10], dim=-1)[0]
        max(float((av @ h.reshape(-1, h.shape[-1]).T).max()) for h in embs)

    lat, cpu = timed(bert, sample)
    out["methods"]["bertscore"] = summarise(lat, cpu, dict(cold_start_s=bload, model_calls=2))
    del bmod

    # ---- NLI (two retrieved chunks per answer sentence; includes the MiniLM retrieval step)
    t = time.perf_counter()
    ntok = AutoTokenizer.from_pretrained(CFG["nli"]["model"])
    nmod = AutoModelForSequenceClassification.from_pretrained(CFG["nli"]["model"], dtype=torch.float32).eval().float()
    nload = time.perf_counter() - t

    def nli(e):
        sents = sentences(e["answer"])[:40] or [e["answer"]]
        ch = chunk_evidence(e["evidence"])[:400]
        sim = minilm.encode(sents) @ minilm.encode(ch).T
        top = np.argsort(-sim, axis=1)[:, :2]
        prem = [ch[j] for row in top for j in row]
        hyp = [s for s, row in zip(sents, top) for _ in row]
        with torch.inference_mode():
            for i in range(0, len(prem), 32):
                enc_ = ntok(prem[i: i + 32], hyp[i: i + 32], truncation=True, max_length=CFG["nli"]["max_length"], padding=True, return_tensors="pt")
                nmod(**enc_)

    lat, cpu = timed(nli, sample)
    out["methods"]["nli"] = summarise(lat, cpu, dict(cold_start_s=nload + minilm.load_s, model_calls=3,
                                                     mean_pairs=float(np.mean([2 * len(sentences(e["answer"])[:40] or [1]) for e in sample]))))
    del nmod

    # ---- LLM judges: uncached prompt, then a second answer for the same source (prefix cache hit)
    import urllib.request

    import run_judge as J

    by_group = {}
    for e in ex:
        by_group.setdefault(e["group"], []).append(e)
    for judge in ["small"] + (["strong"] if a.strong else []):
        model = CFG["judges"][judge]["model"]
        opts = dict(CFG["judges"]["options"])
        urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(
            dict(model=model, keep_alive=0)).encode(), headers={"Content-Type": "application/json"}), timeout=600).read()
        first = J.judge_one(model, sample[0], "v1", opts)                         # cold: includes model load
        cold, warm, cached = first.get("load_s"), [], []
        picks = [sample[i] for i in np.linspace(0, len(sample) - 1, a.judge_n).astype(int)]
        for e in picks:
            other = next((o for o in by_group[e["group"]] if o["id"] != e["id"]), None)
            r1 = J.judge_one(model, e, "v1", opts)
            warm.append(r1)
            if other is not None:
                cached.append(J.judge_one(model, other, "v1", opts))
        def agg(all_rs):
            # token and latency statistics over successful calls only; failed calls are counted and their errors kept
            rs = [r for r in all_rs if r.get("verdict")]
            lat = [r["wall_s"] for r in rs]
            return summarise(lat, None, dict(
                n_calls=len(all_rs), errors=[r.get("error") for r in all_rs if not r.get("verdict")],
                prompt_tokens=float(np.mean([r["prompt_tokens"] for r in rs])),
                cached_prompt_tokens=float(np.mean([r["cached_tokens"] or 0 for r in rs])),
                output_tokens=float(np.mean([r["output_tokens"] for r in rs])),
                prompt_tokens_per_s=float(np.sum([r["prompt_tokens"] - (r["cached_tokens"] or 0) for r in rs]) / max(1e-9, np.sum([r["prompt_s"] for r in rs]))),
                output_tokens_per_s=float(np.sum([r["output_tokens"] for r in rs]) / max(1e-9, np.sum([r["gen_s"] for r in rs]))),
                model_calls=1, failures=int(sum(r.get("verdict") is None for r in all_rs))))
        out["methods"][f"judge_{judge}"] = dict(agg(warm), cold_start_s=cold, model=model, note="new source each call: only the rubric prefix is cached")
        out["methods"][f"judge_{judge}_prefix_cached"] = dict(agg(cached), cold_start_s=cold, model=model, note="second answer for the same source: evidence prefix cached")
        urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(
            dict(model=model, keep_alive=0)).encode(), headers={"Content-Type": "application/json"}), timeout=600).read()
        name = "cost.json" if not a.strong else "cost.json"
        (RESULTS / name).write_text(json.dumps(out, indent=1))
    (RESULTS / "cost.json").write_text(json.dumps(out, indent=1))
    for k, v in out["methods"].items():
        print(f"{k:38s} median {v['median_ms']:9.1f} ms  p95 {v['p95_ms']:9.1f} ms  s/1000 {v['wall_s_per_1000']:9.1f}  cold {v.get('cold_start_s')}")


if __name__ == "__main__":
    main()
