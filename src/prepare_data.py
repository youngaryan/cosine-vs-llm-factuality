"""Normalise RAGTruth, FaithBench, EVOUNA-NQ and HaluEval-QA into one schema with grouped splits.

Output: data/processed/examples.jsonl with fields
  id, dataset, domain, target ('faithfulness' | 'correctness'), label_origin, group, split,
  question, evidence, reference (list), answer, label (1 = unsupported / incorrect), generator, meta
and results/dataset_stats.json (counts and exclusions).
"""
from __future__ import annotations

import collections
import csv
import json

import numpy as np

from common import CFG, PROC, RAW, RESULTS, SEED, read_jsonl, write_jsonl


def flatten(obj, prefix=""):
    """Structured (JSON) evidence -> 'path: value.' lines so that it can be chunked like prose."""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines += flatten(v, f"{prefix}{k} ")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines += flatten(v, f"{prefix}{i + 1} " if not isinstance(v, (dict, list)) else prefix)
    elif obj is not None:
        text = str(obj).strip().replace("\n", " ")
        if text:
            lines.append(f"{prefix.strip()}: {text}" + ("" if text[-1] in ".!?" else "."))
    return lines


def ragtruth(rng, stats):
    src = {s["source_id"]: s for s in read_jsonl(RAW / "ragtruth" / "source_info.jsonl")}
    resp = read_jsonl(RAW / "ragtruth" / "response.jsonl")
    by_task = collections.defaultdict(lambda: collections.defaultdict(list))
    excluded = collections.Counter()
    for r in resp:
        if r["quality"] != "good":
            excluded[f"ragtruth_quality_{r['quality']}"] += 1
            continue
        by_task[src[r["source_id"]]["task_type"]][r["source_id"]].append(r)
    cfg = CFG["data"]["ragtruth"]
    out = []
    for task in sorted(by_task):
        groups = by_task[task]
        train_ids = sorted(g for g, rs in groups.items() if rs[0]["split"] == "train")
        test_ids = sorted(g for g, rs in groups.items() if rs[0]["split"] == "test")
        perm = rng.permutation(len(train_ids))
        n_tr, n_ca = cfg["train_sources_per_task"], cfg["calib_sources_per_task"]
        split_of = {train_ids[i]: "train" for i in perm[:n_tr]}
        split_of.update({train_ids[i]: "calib" for i in perm[n_tr : n_tr + n_ca]})
        split_of.update({g: "test" for g in test_ids})
        stats["unused_official_train_sources"][f"ragtruth_{task}"] = len(train_ids) - n_tr - n_ca
        for g, rs in groups.items():
            if g not in split_of:
                continue
            s = src[g]
            info = s["source_info"]
            if task == "QA":
                question, evidence = info["question"], info["passages"]
            elif task == "Summary":
                question, evidence = "", info
            else:
                question, evidence = "", "\n".join(flatten(info))
            for r in rs:
                types = sorted({l["label_type"] for l in r["labels"]})
                out.append(
                    dict(
                        id=f"ragtruth-{r['id']}", dataset="ragtruth", domain=f"ragtruth_{task.lower()}",
                        target="faithfulness", label_origin="human", group=f"ragtruth-{g}", split=split_of[g],
                        question=question, evidence=evidence, reference=[], answer=r["response"].strip(),
                        label=int(len(r["labels"]) > 0), generator=r["model"],
                        meta=dict(error_types=types, n_spans=len(r["labels"]), source=s["source"]),
                    )
                )
    stats["excluded"].update(excluded)
    return out


def faithbench(rng, stats):
    rows = list(csv.DictReader(open(RAW / "faithbench" / "FaithBench.csv", encoding="utf8")))
    sources = sorted({r["source"] for r in rows})
    perm = rng.permutation(len(sources))
    n_ca = CFG["data"]["faithbench"]["calib_sources"]
    split_of = {sources[i]: ("calib" if k < n_ca else "test") for k, i in enumerate(perm)}
    gid = {s: i for i, s in enumerate(sources)}
    bad = {"Unwanted", "Questionable"}
    out = []
    for i, r in enumerate(rows):
        out.append(
            dict(
                id=f"faithbench-{i}", dataset="faithbench", domain="faithbench", target="faithfulness",
                label_origin="human", group=f"faithbench-{gid[r['source']]}", split=split_of[r["source"]],
                question="", evidence=r["source"], reference=[], answer=r["summary"].strip(),
                label=int(r["worst-label"] in bad), generator=r["LLM"],
                meta=dict(worst=r["worst-label"], best=r["best-label"], label_best=int(r["best-label"] in bad)),
            )
        )
    return out


def evouna(rng, stats):
    data = json.load(open(RAW / "evouna" / "NQ.json", encoding="utf8"))
    keep = [(i, d) for i, d in enumerate(data) if not d.get("improper")]
    stats["excluded"]["evouna_improper_questions"] = len(data) - len(keep)
    perm = rng.permutation(len(keep))
    cfg = CFG["data"]["evouna_nq"]
    out = []
    for k, j in enumerate(perm):
        i, d = keep[j]
        split = "train" if k < cfg["train_questions"] else "calib" if k < cfg["train_questions"] + cfg["calib_questions"] else "test"
        refs = [a.strip() for a in d["golden_answer"].split("/") if a.strip()]
        for m in ["fid", "gpt35", "chatgpt", "gpt4", "newbing"]:
            ans, judge = d.get(f"answer_{m}"), d.get(f"judge_{m}")
            if judge is None or not ans or not str(ans).strip():
                stats["excluded"]["evouna_missing_answer_or_judgement"] += 1
                continue
            out.append(
                dict(
                    id=f"evouna-{i}-{m}", dataset="evouna_nq", domain="evouna_nq", target="correctness",
                    label_origin="human", group=f"evouna-{i}", split=split, question=d["question"], evidence="",
                    reference=refs, answer=str(ans).strip(), label=int(not judge), generator=m, meta={},
                )
            )
    return out


def halueval(rng, stats):
    rows = read_jsonl(RAW / "halueval" / "qa_data.json")
    cfg = CFG["data"]["halueval_qa"]
    perm = rng.permutation(len(rows))[: cfg["calib_items"] + cfg["test_items"]]
    stats["unused_official_train_sources"]["halueval_qa_items_not_sampled"] = len(rows) - len(perm)
    out = []
    for k, i in enumerate(perm):
        d = rows[int(i)]
        split = "calib" if k < cfg["calib_items"] else "test"
        for lab, key in [(0, "right_answer"), (1, "hallucinated_answer")]:
            out.append(
                dict(
                    id=f"halueval-{i}-{lab}", dataset="halueval_qa", domain="halueval_qa", target="faithfulness",
                    label_origin="synthetic", group=f"halueval-{i}", split=split, question=d["question"],
                    evidence=d["knowledge"], reference=[], answer=d[key].strip(), label=lab,
                    generator="chatgpt-synthetic" if lab else "gold", meta={},
                )
            )
    return out


def main():
    stats = dict(excluded=collections.Counter(), unused_official_train_sources={})
    examples = []
    for i, fn in enumerate([ragtruth, faithbench, evouna, halueval]):
        examples += fn(np.random.default_rng(SEED + i), stats)
    # leakage check: a group never appears in more than one split
    seen = {}
    for e in examples:
        assert seen.setdefault(e["group"], e["split"]) == e["split"], f"group leak {e['group']}"
    write_jsonl(PROC / "examples.jsonl", examples)
    table = collections.defaultdict(lambda: collections.Counter())
    for e in examples:
        table[(e["domain"], e["split"])]["n"] += 1
        table[(e["domain"], e["split"])]["unsupported"] += e["label"]
    groups = collections.defaultdict(set)
    for e in examples:
        groups[(e["domain"], e["split"])].add(e["group"])
    stats["splits"] = [
        dict(domain=d, split=s, n=c["n"], unsupported=c["unsupported"], groups=len(groups[(d, s)]))
        for (d, s), c in sorted(table.items())
    ]
    stats["excluded"] = dict(stats["excluded"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")
    for r in stats["splits"]:
        print(r)
    print(stats["excluded"], stats["unused_official_train_sources"])


if __name__ == "__main__":
    main()
