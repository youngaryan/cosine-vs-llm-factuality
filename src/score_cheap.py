"""Lexical scores, structured checks and embedding-cosine scores for every example.

Writes results/scores_cheap.parquet (one row per example) and data/processed/retrieval.json
(top-2 evidence chunks per answer sentence, used by the NLI baseline).

Score orientation: higher = more supported. Evidence-grounded configurations:
  <emb>_doc : cos(answer, evidence) on whole texts (truncated to the encoder's max length)
  <emb>_mm  : mean over answer sentences of the max cosine to any evidence chunk   (primary)
  <emb>_min : min  over answer sentences of the max cosine to any evidence chunk
Reference-based configuration (EVOUNA): <emb>_mm = <emb>_min = <emb>_doc = max over references of
cos(answer, reference); <emb>_qa adds the question to both sides.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from common import (CFG, PROC, RESULTS, chunk_evidence, content_tokens, neg_count, numbers, proper_tokens,
                    read_jsonl, sentences, tokens)

MAX_SENTS, MAX_CHUNKS = 40, 400


def norm_answer(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def f1(a, b) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    ov = sum((ca & cb).values())
    if not ov:
        return 0.0
    p, r = ov / len(a), ov / len(b)
    return 2 * p * r / (p + r)


def lexical_evidence(e, ev_cache):
    ev = ev_cache[e["evidence"]]
    ans_tok, ans_ct = tokens(e["answer"]), content_tokens(e["answer"])
    sents = sentences(e["answer"])[:MAX_SENTS] or [e["answer"]]
    sent_prec = []
    for s in sents:
        ct = content_tokens(s)
        if ct:
            sent_prec.append(sum(t in ev["tokset"] for t in ct) / len(ct))
    nums = numbers(e["answer"])
    n_unsup = sum(n not in ev["nums"] for n in nums)
    pt = proper_tokens(e["answer"])
    e_unsup = sum(t not in ev["tokset"] for t in pt)
    big = list(zip(ans_ct, ans_ct[1:]))
    return dict(
        lex_em=float(norm_answer(e["answer"]) in ev["norm"]),
        lex_f1=f1(ans_tok, ev["tok"]),
        lex_tok_prec=sum(t in ev["tokset"] for t in ans_ct) / len(ans_ct) if ans_ct else 1.0,
        chk_min_sent_prec=min(sent_prec) if sent_prec else 1.0,
        chk_num_unsup=float(n_unsup), chk_num_ratio=n_unsup / len(nums) if nums else 0.0,
        chk_ent_unsup=float(e_unsup), chk_ent_ratio=e_unsup / len(pt) if pt else 0.0,
        chk_novel_bigram=sum(b not in ev["bigrams"] for b in big) / len(big) if big else 0.0,
        len_ans_tok=float(len(ans_tok)), len_ans_sent=float(len(sents)), len_ev_tok=float(len(ev["tok"])),
    )


def lexical_reference(e):
    a_norm, a_tok, a_ct = norm_answer(e["answer"]), tokens(e["answer"]), set(content_tokens(e["answer"]))
    refs = e["reference"]
    a_nums = set(numbers(e["answer"]))
    ref_nums = [set(numbers(r)) for r in refs]
    has_ref_num = any(ref_nums)
    num_ok = any(rn and rn <= a_nums for rn in ref_nums)
    rec = []
    for r in refs:
        ct = content_tokens(r)
        rec.append(sum(t in a_ct for t in ct) / len(ct) if ct else 0.0)
    return dict(
        lex_em=float(any(norm_answer(r) == a_norm for r in refs)),
        lex_contains=float(any(norm_answer(r) and norm_answer(r) in a_norm for r in refs)),
        lex_f1=max(f1(a_tok, tokens(r)) for r in refs),
        lex_tok_prec=max(rec),                                  # reference content-token recall in the answer
        chk_min_sent_prec=max(rec),
        chk_num_unsup=float(has_ref_num and not num_ok), chk_num_ratio=float(has_ref_num and not num_ok),
        chk_ent_unsup=0.0, chk_ent_ratio=0.0, chk_novel_bigram=0.0,
        len_ans_tok=float(len(a_tok)), len_ans_sent=float(len(sentences(e["answer"])) or 1),
        len_ev_tok=float(np.mean([len(tokens(r)) for r in refs])),
    )


class STEncoder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        t = time.perf_counter()
        self.m = SentenceTransformer(name, device="cpu")
        self.load_s = time.perf_counter() - t

    def encode(self, texts):
        return self.m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True).astype(np.float32)


class OllamaEncoder:
    """nomic-embed-text through Ollama; the model card's 'classification: ' prefix is used for both sides."""

    def __init__(self, name="nomic-embed-text"):
        self.name, self.load_s = name, 0.0

    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 48):
            body = dict(model=self.name, input=["classification: " + t[:6000] for t in texts[i: i + 48]], keep_alive="10m")
            req = urllib.request.Request("http://localhost:11434/api/embed", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            v = np.asarray(json.load(urllib.request.urlopen(req, timeout=1800))["embeddings"], dtype=np.float32)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out)


def embed_scores(name, enc, examples, units, want_retrieval):
    texts = {}
    for e in examples:
        u = units[e["id"]]
        for t in u["sents"] + u["chunks"] + u["whole"]:
            texts.setdefault(t, len(texts))
    order = sorted(texts, key=len)                                   # length-sorted batches
    t0 = time.perf_counter()
    vecs = np.zeros((len(order), 0), dtype=np.float32)
    parts = [enc.encode(order[i: i + 4096]) for i in range(0, len(order), 4096)]
    vecs = np.concatenate(parts)
    secs = time.perf_counter() - t0
    row = {t: i for i, t in enumerate(order)}
    out, retr = {}, {}
    for e in examples:
        u = units[e["id"]]
        S = vecs[[row[t] for t in u["sents"]]]
        C = vecs[[row[t] for t in u["chunks"]]]
        sim = S @ C.T
        mx = sim.max(axis=1)
        w = vecs[[row[t] for t in u["whole"]]]
        rec = {f"{name}_mm": float(mx.mean()), f"{name}_min": float(mx.min())}
        if e["target"] == "correctness":
            n = len(e["reference"])
            rec[f"{name}_doc"] = float((w[0] @ w[1: 1 + n].T).max())
            rec[f"{name}_qa"] = float((w[1 + n] @ w[2 + n:].T).max())
            rec[f"{name}_mm"] = rec[f"{name}_min"] = rec[f"{name}_doc"]
        else:
            rec[f"{name}_doc"] = float(w[0] @ w[1])
            rec[f"{name}_qa"] = rec[f"{name}_doc"]
        out[e["id"]] = rec
        if want_retrieval and e["target"] != "correctness":
            retr[e["id"]] = np.argsort(-sim, axis=1)[:, :2].tolist()
    stats = dict(n_texts=len(order), chars=int(sum(map(len, order))), encode_s=secs, load_s=enc.load_s)
    return out, retr, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nomic", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep lexical columns from an existing scores_cheap.parquet; recompute only the embedders")
    ap.add_argument("--encoders", default="minilm,bge_small")
    a = ap.parse_args()
    resume = a.resume and (RESULTS / "scores_cheap.parquet").exists()
    ex = read_jsonl(PROC / "examples.jsonl")
    if (PROC / "counterfactuals.jsonl").exists():
        ex += read_jsonl(PROC / "counterfactuals.jsonl")
    print("examples", len(ex), flush=True)

    ev_cache, units, rows = {}, {}, []
    for e in ex:
        if e["target"] == "correctness":
            lex = {} if resume else lexical_reference(e)
            refs = e["reference"]
            units[e["id"]] = dict(sents=[e["answer"]], chunks=refs,
                                  whole=[e["answer"], *refs, e["question"] + " " + e["answer"],
                                         *[e["question"] + " " + r for r in refs]])
        else:
            if e["evidence"] not in ev_cache:
                tok = tokens(e["evidence"])
                ct = content_tokens(e["evidence"])
                ev_cache[e["evidence"]] = dict(tok=tok, tokset=set(tok), nums=set(numbers(e["evidence"])),
                                               norm=norm_answer(e["evidence"]), bigrams=set(zip(ct, ct[1:])),
                                               chunks=chunk_evidence(e["evidence"])[:MAX_CHUNKS])
            lex = {} if resume else lexical_evidence(e, ev_cache)
            units[e["id"]] = dict(sents=sentences(e["answer"])[:MAX_SENTS] or [e["answer"]],
                                  chunks=ev_cache[e["evidence"]]["chunks"], whole=[e["answer"], e["evidence"]])
        m = e.get("meta", {})
        rows.append(dict(id=e["id"], dataset=e["dataset"], domain=e["domain"], target=e["target"], split=e["split"],
                         group=e["group"], label=e["label"], label_origin=e["label_origin"], generator=e["generator"],
                         cf_type=m.get("type", ""), cf_base=m.get("base", ""), cf_matched=bool(m.get("matched_locus", False)),
                         error_types="|".join(m.get("error_types", [])), label_best=m.get("label_best", -1), **lex))
    df = pd.DataFrame(rows).set_index("id")
    if resume:
        old = pd.read_parquet(RESULTS / "scores_cheap.parquet")
        assert set(old.index) == set(df.index), "example set changed; run without --resume"
        df = old.drop(columns=[c for c in old.columns if c.split("_")[0] in ("minilm", "bge") or c == "chk_neg_mismatch"])
        print("resumed lexical columns from scores_cheap.parquet", flush=True)
    else:
        df = tfidf_column(df, ex)
    cost = json.loads((RESULTS / "cost_embedding_bulk.json").read_text()) if resume and (RESULTS / "cost_embedding_bulk.json").exists() else {}
    run_embedders(df, ex, units, a, cost)


def tfidf_column(df, ex):
    # TF-IDF fitted on training-split texts only
    from sklearn.feature_extraction.text import TfidfVectorizer

    fit = [e["answer"] for e in ex if e["split"] == "train"] + list({e["evidence"] for e in ex if e["split"] == "train" and e["evidence"]})
    tv = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2).fit(fit)
    A = tv.transform([e["answer"] for e in ex])
    tfidf = []
    ev_vec = {}
    for i, e in enumerate(ex):
        if e["target"] == "correctness":
            R = tv.transform(e["reference"])
            tfidf.append(float((A[i] @ R.T).max()) if R.nnz and A[i].nnz else 0.0)
        else:
            if e["evidence"] not in ev_vec:
                ev_vec[e["evidence"]] = tv.transform([e["evidence"]])
            tfidf.append(float((A[i] @ ev_vec[e["evidence"]].T)[0, 0]))
    df["lex_tfidf"] = tfidf
    print("lexical done", flush=True)
    return df


def run_embedders(df, ex, units, a, cost):
    encoders = [("minilm", lambda: STEncoder("sentence-transformers/all-MiniLM-L6-v2"), ex),
                ("bge_small", lambda: STEncoder("BAAI/bge-small-en-v1.5"), ex)]
    if not a.skip_nomic:
        encoders.append(("nomic", OllamaEncoder, [e for e in ex if e["split"] != "train"]))
    want = a.encoders.split(",") + ([] if a.skip_nomic else ["nomic"])
    encoders = [x for x in encoders if x[0] in want]
    assert encoders[0][0] == "minilm", "MiniLM must run first: it provides the retrieval used by NLI and the negation check"
    retrieval = {}
    for name, make, subset in encoders:
        enc = make()
        sc, retr, st = embed_scores(name, enc, subset, units, want_retrieval=(name == "minilm"))
        cost[name] = st
        df = df.join(pd.DataFrame.from_dict(sc, orient="index"))
        if retr:
            retrieval = retr
            df = add_negation(df, ex, units, retrieval)                  # saved before slower encoders run
            (PROC / "retrieval.json").write_text(json.dumps(retrieval))
        print(name, st, flush=True)
        del enc
        df.to_parquet(RESULTS / "scores_cheap.parquet")
        (RESULTS / "cost_embedding_bulk.json").write_text(json.dumps(cost, indent=1))
    print(df.groupby(["domain", "split"]).size())


def add_negation(df, ex, units, retrieval):
    """Negation mismatch between each answer sentence and its best-matching evidence chunk (MiniLM retrieval)."""
    neg = {}
    for e in ex:
        if e["target"] == "correctness":
            neg[e["id"]] = float((neg_count(e["answer"]) > 0) != any(neg_count(r) > 0 for r in e["reference"]))
            continue
        u = units[e["id"]]
        mm = [(neg_count(s) > 0) != (neg_count(u["chunks"][top[0]]) > 0) for s, top in zip(u["sents"], retrieval[e["id"]])]
        neg[e["id"]] = float(np.mean(mm))
    df["chk_neg_mismatch"] = pd.Series(neg)
    return df


if __name__ == "__main__":
    main()
