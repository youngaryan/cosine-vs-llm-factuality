"""BERTScore baseline (roberta-base, layer 10, no IDF, no baseline rescaling), implemented directly.

Evidence-grounded data: BERTScore *precision* of the answer against the evidence, i.e. the mean over
answer tokens of the maximum cosine to any evidence token. Evidence longer than the encoder window is
encoded in 384-token windows and the token embeddings are pooled into one candidate set.
Reference-based data: BERTScore F1 against the best-matching reference.
Runs on calibration and test splits and on the counterfactual set.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import CFG, PROC, RESULTS, read_jsonl

WIN = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel, AutoTokenizer

    cfg = CFG["bertscore"]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModel.from_pretrained(cfg["model"], output_hidden_states=True).eval()
    load_s = time.perf_counter() - t0
    layer = cfg["layer"]

    def embed(texts):
        """List of L2-normalised token-embedding matrices (special tokens dropped)."""
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for b in range(0, len(order), 16):
            idx = order[b: b + 16]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=WIN + 2, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = model(**enc).hidden_states[layer]
            h = torch.nn.functional.normalize(h, dim=-1).numpy()
            for j, i in enumerate(idx):
                n = int(enc["attention_mask"][j].sum())
                out[i] = h[j, 1: n - 1].astype(np.float32)
        return out

    def windows(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= WIN:
            return [text]
        return [tok.decode(ids[i: i + WIN]) for i in range(0, len(ids), WIN - 32)]

    ex = [e for e in read_jsonl(PROC / "examples.jsonl") if e["split"] in ("calib", "test")]
    ex += read_jsonl(PROC / "counterfactuals.jsonl")
    if a.limit:
        ex = ex[: a.limit]
    t0 = time.perf_counter()
    rows = {}
    # Evidence-grounded examples are processed one evidence text at a time (evidence encoded once, then all
    # answers written from it), so that only one evidence's token embeddings are held in memory.
    by_ev = {}
    for e in ex:
        if e["target"] != "correctness":
            by_ev.setdefault(e["evidence"], []).append(e)
    for n, (ev, group) in enumerate(sorted(by_ev.items(), key=lambda kv: kv[0])):
        E = np.concatenate(embed(windows(ev)))
        for e, av in zip(group, embed([g["answer"] for g in group])):
            rows[e["id"]] = dict(bertscore=float((av @ E.T).max(1).mean()) if len(av) else 0.0)
        if n % 100 == 0:
            print("evidence", n, len(by_ev), round(time.perf_counter() - t0), flush=True)
    ref = [e for e in ex if e["target"] == "correctness"]
    for b in range(0, len(ref), 256):
        batch = ref[b: b + 256]
        for e, av in zip(batch, embed([e["answer"] for e in batch])):
            best = 0.0
            for rv in (embed(e["reference"][:5]) if len(av) else []):
                if len(rv) == 0:
                    continue
                sim = av @ rv.T
                p, r = sim.max(1).mean(), sim.max(0).mean()
                best = max(best, 2 * p * r / (p + r) if p + r > 0 else 0.0)
            rows[e["id"]] = dict(bertscore=float(best))
        print("references", b, len(ref), round(time.perf_counter() - t0), flush=True)
    secs = time.perf_counter() - t0
    pd.DataFrame.from_dict(rows, orient="index").to_parquet(RESULTS / "scores_bertscore.parquet")
    (RESULTS / "cost_bertscore_bulk.json").write_text(json.dumps(
        dict(model=cfg["model"], load_s=load_s, examples=len(ex), seconds=secs, threads=a.threads), indent=1))
    print("done", len(rows), round(secs))


if __name__ == "__main__":
    main()
