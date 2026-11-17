"""LLM-as-a-judge via a local Ollama server. Resume-safe; one JSON line per call.

  python src/run_judge.py --select                      # write data/processed/judge_subsets.json
  python src/run_judge.py --judge small --part natural  # RAGTruth + FaithBench + EVOUNA subsets
  python src/run_judge.py --judge small --part cf
  python src/run_judge.py --judge small --part sensitivity --prompt v2

Calls are ordered by source so that consecutive calls share the rubric+evidence prefix and
Ollama can reuse its prompt cache; cached and evaluated prompt tokens are both logged.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import time
import urllib.request

import numpy as np

from common import CFG, PROC, RESULTS, SEED, read_jsonl

LABELS = ["SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"]
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": LABELS}}, "required": ["verdict"]}

RUBRIC = {
    "v1": (
        "You are a strict fact-checking judge. You receive EVIDENCE, sometimes a QUESTION, and an ANSWER. "
        "Decide whether every factual claim in the ANSWER is supported by the EVIDENCE.\n"
        "- SUPPORTED: every factual claim in the answer is stated in, or directly entailed by, the evidence. "
        "Paraphrasing is fine.\n"
        "- UNSUPPORTED: at least one claim contradicts the evidence (for example a wrong number, date, name, "
        "negation, cause, or who did what to whom).\n"
        "- INSUFFICIENT_EVIDENCE: no claim contradicts the evidence, but at least one claim adds information "
        "that the evidence does not contain.\n"
        "Judge only against the evidence, never from your own knowledge. Ignore style, fluency and completeness.\n"
        'Reply with one JSON object: {"verdict": "SUPPORTED" | "UNSUPPORTED" | "INSUFFICIENT_EVIDENCE"}'
    ),
    "v2": (
        "Task: hallucination check. Read the SOURCE material and the RESPONSE written from it. A response is "
        "faithful only if a careful reader could verify each of its statements from the source alone.\n"
        "Choose one label:\n"
        "SUPPORTED = the response is fully faithful to the source.\n"
        "UNSUPPORTED = the response states something that conflicts with the source.\n"
        "INSUFFICIENT_EVIDENCE = nothing conflicts, but the response includes details absent from the source.\n"
        "Do not use outside knowledge. Do not penalise wording or omissions.\n"
        'Answer with JSON only: {"verdict": "<label>"}'
    ),
    "ref": (
        "You are a strict grader of answers to factual questions. You receive a QUESTION, one or more REFERENCE "
        "ANSWERS (any one of them is correct) and a CANDIDATE ANSWER.\n"
        "- SUPPORTED: the candidate gives the same answer as a reference. Different wording and extra correct "
        "detail are fine.\n"
        "- UNSUPPORTED: the candidate gives a different or conflicting answer, or does not answer the question.\n"
        "- INSUFFICIENT_EVIDENCE: the references do not let you decide whether the candidate is right.\n"
        "Grade only against the references, never from your own knowledge.\n"
        'Reply with one JSON object: {"verdict": "SUPPORTED" | "UNSUPPORTED" | "INSUFFICIENT_EVIDENCE"}'
    ),
}


def build_messages(e, prompt):
    if e["target"] == "correctness":
        user = (f"QUESTION: {e['question']}\n\nREFERENCE ANSWERS:\n" + "\n".join(f"- {r}" for r in e["reference"][:8])
                + f"\n\nCANDIDATE ANSWER:\n{e['answer']}")
        return [{"role": "system", "content": RUBRIC["ref"]}, {"role": "user", "content": user}]
    ev, ans = ("SOURCE", "RESPONSE") if prompt == "v2" else ("EVIDENCE", "ANSWER")
    user = f"{ev}:\n{e['evidence']}\n\n"
    if e["question"]:
        user += f"QUESTION: {e['question']}\n\n"
    user += f"{ans}:\n{e['answer']}"
    return [{"role": "system", "content": RUBRIC[prompt]}, {"role": "user", "content": user}]


def p_supported(logprobs):
    """P(SUPPORTED) from the token distribution at the first token of the verdict string, if available."""
    text = ""
    for tok in logprobs or []:
        if re.search(r'"verdict"\s*:\s*"$', text):
            mass = collections.Counter()
            for alt in tok.get("top_logprobs", []):
                t = alt["token"].strip().strip('"').upper()
                if t and "SUPPORTED".startswith(t[:3]) and t[0] == "S":
                    mass["S"] += math.exp(alt["logprob"])
                elif t[:2] in ("UN", "IN") or t[:1] in ("U", "I"):
                    mass["U"] += math.exp(alt["logprob"])
            tot = mass["S"] + mass["U"]
            return mass["S"] / tot if tot > 0 else None
        text += tok["token"]
    return None


def call(model, messages, opts):
    body = dict(model=model, messages=messages, stream=False, format=SCHEMA, options=opts, keep_alive="15m",
                logprobs=True, top_logprobs=8)
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    r["wall_s"] = time.perf_counter() - t0
    return r


def judge_one(model, e, prompt, opts):
    msgs = build_messages(e, prompt)
    rec = dict(id=e["id"], group=e["group"], prompt=prompt if e["target"] != "correctness" else "ref", retries=0)
    for attempt in range(2):
        try:
            r = call(model, msgs, opts)
            raw = r["message"]["content"]
            verdict = json.loads(raw).get("verdict")
            if verdict not in LABELS:
                raise ValueError(f"bad verdict {raw[:80]}")
            rec.update(verdict=verdict, raw=raw, p_supported=p_supported(r.get("logprobs")),
                       prompt_tokens=r.get("prompt_eval_count"), cached_tokens=r.get("prompt_eval_cached_count"),
                       output_tokens=r.get("eval_count"), prompt_s=r.get("prompt_eval_duration", 0) / 1e9,
                       gen_s=r.get("eval_duration", 0) / 1e9, load_s=r.get("load_duration", 0) / 1e9,
                       wall_s=r["wall_s"], error=None)
            return rec
        except Exception as ex:  # noqa: BLE001 - logged and retried once
            rec.update(retries=attempt + 1, error=repr(ex)[:300])
    rec.update(verdict=None)
    return rec


def select():
    rng = np.random.default_rng(SEED + 23)
    ex = [e for e in read_jsonl(PROC / "examples.jsonl") if e["split"] == "test"]
    cf = read_jsonl(PROC / "counterfactuals.jsonl")
    sub = {}
    order = {}
    for dom in ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt", "faithbench"]:
        g = sorted({e["group"] for e in ex if e["domain"] == dom})
        order[dom] = [g[i] for i in rng.permutation(len(g))]
    ev = sorted(e["id"] for e in ex if e["domain"] == "evouna_nq")
    order["evouna"] = [ev[i] for i in rng.permutation(len(ev))]
    para = {r["meta"]["base"] for r in cf if r["meta"]["type"] == "paraphrase"}
    bases = collections.defaultdict(list)
    for r in cf:
        if r["meta"]["type"] == "original" and r["meta"]["base"] in para:
            bases[r["domain"]].append(r["meta"]["base"])
    cf_order = []
    lists = [[b[i] for i in rng.permutation(len(b))] for _, b in sorted((d, sorted(v)) for d, v in bases.items())]
    for k in range(max(map(len, lists), default=0)):
        cf_order += [l[k] for l in lists if k < len(l)]
    for name in ["small", "strong"]:
        c = CFG["judges"]["subsets"][name]
        sub[name] = dict(
            groups=sum((order[d][: c["ragtruth_per_task"]] for d in ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt"]), [])
            + order["faithbench"][: c["faithbench_sources"]],
            evouna_ids=order["evouna"][: c["evouna_examples"]],
            cf_bases=cf_order[: c["cf_bases"]],
        )
    n = CFG["judges"]["prompt_sensitivity"]["ragtruth_per_task"]
    sub["sensitivity"] = dict(groups=sum((order[d][:n] for d in ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt"]), []))
    (PROC / "judge_subsets.json").write_text(json.dumps(sub, indent=1))
    print({k: {kk: len(vv) for kk, vv in v.items()} for k, v in sub.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--judge", choices=["small", "strong"])
    ap.add_argument("--part", choices=["natural", "cf", "sensitivity"])
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.select:
        return select()
    sub = json.loads((PROC / "judge_subsets.json").read_text())
    model = CFG["judges"][a.judge]["model"]
    opts = dict(CFG["judges"]["options"])
    if a.part == "cf":
        want = set(sub[a.judge]["cf_bases"])
        todo = [r for r in read_jsonl(PROC / "counterfactuals.jsonl") if r["meta"]["base"] in want]
    else:
        ex = [e for e in read_jsonl(PROC / "examples.jsonl") if e["split"] == "test"]
        s = sub["sensitivity"] if a.part == "sensitivity" else sub[a.judge]
        groups, ids = set(s["groups"]), set(s.get("evouna_ids", []))
        todo = [e for e in ex if e["group"] in groups or e["id"] in ids]
    todo.sort(key=lambda e: (e["target"], e["group"], e["id"]))
    if a.limit:
        todo = todo[: a.limit]
    out = RESULTS / f"judge_{a.judge}_{a.part}_{a.prompt}.jsonl"
    done = {json.loads(l)["id"] for l in open(out, encoding="utf8")} if out.exists() else set()
    t0 = time.time()
    with open(out, "a", encoding="utf8") as f:
        for i, e in enumerate(todo):
            if e["id"] in done:
                continue
            rec = judge_one(model, e, a.prompt, opts)
            rec.update(judge=a.judge, model=model)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{a.judge} {a.part} {i}/{len(todo)} elapsed {time.time() - t0:.0f}s last={rec.get('verdict')} "
                      f"ptok={rec.get('prompt_tokens')} cached={rec.get('cached_tokens')}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
