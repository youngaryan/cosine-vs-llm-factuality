"""Build every LaTeX table body and every number macro used in the manuscript from results/*.json.

  python src/make_tables.py

Writes paper/generated/tab_*.tex (tabular bodies), paper/generated/numbers.tex (\\newcommand macros),
paper/generated/prompts.tex (the judge prompts, verbatim from src/run_judge.py) and results/numbers.json.
Nothing in the manuscript's results is typed by hand: prose refers to the macros.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import CFG, PROC, RESULTS, ROOT, read_jsonl

GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
MACROS: dict[str, str] = {}

DOM = {"ragtruth_qa": "RT-QA", "ragtruth_summary": "RT-Sum", "ragtruth_data2txt": "RT-D2T", "faithbench": "FaithBench",
       "halueval_qa": "HaluEval-QA", "evouna_nq": "EVOUNA-NQ"}
LAB = {"lex_em": "Exact match", "lex_contains": "Reference containment", "lex_f1": "Token F1",
       "lex_tfidf": "TF--IDF cosine", "lex_tok_prec": "Token precision",
       "minilm_doc": "MiniLM cosine (document)", "minilm_qa": "MiniLM cosine (question+answer)",
       "minilm_mm": "MiniLM cosine (mean-max)", "minilm_min": "MiniLM cosine (min-max)",
       "bge_small_doc": "BGE-small cosine (document)", "bge_small_mm": "BGE-small cosine (mean-max)",
       "bge_small_min": "BGE-small cosine (min-max)", "bertscore": "BERTScore", "nli_mean_ec": "NLI (SummaC-ZS style)",
       "nli_min_e": "NLI (min. entailment)", "risk_cos": "Risk model: cosine only",
       "risk_cos_checks": "Risk model: cosine + checks", "risk_full": "Risk model: full",
       "judge_small": "Judge (3B)", "judge_strong": "Judge (14B)"}
SHORT = {"minilm_mm": "MiniLM", "bge_small_mm": "BGE-small", "bertscore": "BERTScore", "nli_mean_ec": "NLI",
         "risk_full": "Risk (full)", "risk_cos": "Risk (cos)", "risk_cos_checks": "Risk (cos+chk)",
         "lex_tok_prec": "Tok. prec.", "lex_tfidf": "TF--IDF"}


def R(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")                                  # no "-.000" for values that round to zero
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def ci(c, d=3, p=False):
    if not c:
        return ""
    return f"[{pct(c[0])}, {pct(c[1])}]" if p else f"[{f(c[0], d)}, {f(c[1], d)}]"


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s.encode("ascii", "replace").decode().replace("?", "'") if not s.isascii() else s


def M(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in MACROS, name
    MACROS[name] = str(value)


def pv(p, nb):
    return f"< {1 / nb:.4f}".rstrip("0") if p < 1 / nb else f"= {p:.3f}"


def write(name, lines):
    (GEN / name).write_text("\n".join(lines) + "\n", encoding="utf8")


def bold_best(vals, fmt, higher=True):
    ok = [v for v in vals if v is not None]
    if not ok:
        return [fmt(v) for v in vals]
    best = max(ok) if higher else min(ok)
    return [(r"\textbf{" + fmt(v) + "}") if v is not None and abs(v - best) < 1e-12 else fmt(v) for v in vals]


# ------------------------------------------------------------------------------------------------- datasets
def t_datasets():
    ds, cf = R("dataset_stats"), R("cf_stats")
    sp = {(r["domain"], r["split"]): r for r in ds["splits"]}
    info = {"ragtruth_qa": ("faithfulness", "human"), "ragtruth_summary": ("faithfulness", "human"),
            "ragtruth_data2txt": ("faithfulness", "human"), "faithbench": ("faithfulness", "human"),
            "halueval_qa": ("faithfulness", "synthetic"), "evouna_nq": ("correctness", "human")}
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Domain & Labels & Train & Calib. & Test & Test groups & Unsup.\ (\%) \\", r"\midrule"]
    tot = 0
    for d, (tgt, lab) in info.items():
        g = lambda s, k="n": sp.get((d, s), {}).get(k)
        te = sp[(d, "test")]
        tot += te["n"]
        if d == "evouna_nq":
            L.append(r"\midrule")
        L.append(f"{DOM[d]} & {lab} & {g('train') or '--'} & {g('calib')} & {te['n']} & {te['groups']} & {pct(te['unsupported'] / te['n'])} \\\\")
    L += [r"\midrule",
          f"Counterfactual (RT test) & rule & -- & -- & {cf['pairs']} & {cf['bases']} & 100.0 \\\\",
          f"\\quad paraphrase controls & LLM+rule & -- & -- & {cf['paraphrase']['kept']} & {cf['paraphrase']['kept']} & 0.0 \\\\",
          r"\bottomrule", r"\end{tabular}"]
    write("tab_datasets.tex", L)
    M("NTestNatural", f"{tot:,}")
    M("NTestFaith", f"{sum(sp[(d, 'test')]['n'] for d in info if d != 'evouna_nq'):,}")
    M("NTestRagtruth", f"{sum(sp[(d, 'test')]['n'] for d in ['ragtruth_qa', 'ragtruth_summary', 'ragtruth_data2txt']):,}")
    M("NTestEvouna", f"{sp[('evouna_nq', 'test')]['n']:,}")
    M("NCfPairs", f"{cf['pairs']:,}")
    M("NCfBases", f"{cf['bases']:,}")
    M("NCfMatched", f"{cf['matched_locus']:,}")
    M("NCfPara", cf["paraphrase"]["kept"])
    att = cf["paraphrase"]["kept"] + sum(cf["paraphrase"]["rejected"].values())
    M("NCfParaAttempted", att)
    M("CfParaKeptPct", pct(cf["paraphrase"]["kept"] / att, 0))
    for t, n in cf["by_type"].items():
        M("NCf" + t.replace("_", " ").title().replace(" ", ""), f"{n:,}")
    M("NExclRefusal", ds["excluded"]["ragtruth_quality_incorrect_refusal"])
    M("NExclTrunc", ds["excluded"]["ragtruth_quality_truncated"])
    M("NExclImproper", ds["excluded"]["evouna_improper_questions"])


# ---------------------------------------------------------------------------------------------- main table
MAIN_ROWS = ["lex_tok_prec", "lex_tfidf", "minilm_doc", "minilm_mm", "minilm_min", "bge_small_mm", "bertscore", "nli_mean_ec",
             "risk_cos", "risk_cos_checks", "risk_full"]


def t_main():
    m = R("main_performance")
    ft, pooled = m["full_test"], m["pooled_ragtruth"]
    doms = ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt", "faithbench", "halueval_qa"]
    L = [r"\begin{tabular}{l" + "rr" * 5 + "rl}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DOM[d]}}}" for d in doms) + r" & \multicolumn{2}{c}{RAGTruth pooled} \\",
         "Method & " + " & ".join(["AUC & BAcc"] * 5) + r" & BAcc & 95\% CI \\", r"\midrule"]
    cols = []
    for d in doms:
        cols.append(bold_best([ft[d].get(s, {}).get("auroc") for s in MAIN_ROWS], f))
        cols.append(bold_best([ft[d].get(s, {}).get("balanced_accuracy") for s in MAIN_ROWS], f))
    cols.append(bold_best([pooled.get(s, {}).get("balanced_accuracy") for s in MAIN_ROWS], f))
    for i, s in enumerate(MAIN_ROWS):
        if s in ("minilm_doc", "bertscore", "risk_cos"):
            L.append(r"\midrule")
        L.append(LAB[s] + " & " + " & ".join(c[i] for c in cols) + " & " + ci(pooled.get(s, {}).get("bacc_ci")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main.tex", L)
    for s, tag in [("minilm_mm", "Cos"), ("minilm_doc", "CosDoc"), ("bge_small_mm", "Bge"), ("bertscore", "Bert"), ("nli_mean_ec", "Nli"),
                   ("risk_full", "Risk"), ("risk_cos", "RiskCos"), ("risk_cos_checks", "RiskChk"), ("lex_tok_prec", "TokPrec")]:
        if s not in pooled:
            continue
        M(f"Pool{tag}Bacc", f(pooled[s]["balanced_accuracy"]))
        M(f"Pool{tag}BaccCi", ci(pooled[s]["bacc_ci"]))
        M(f"Pool{tag}Auc", f(pooled[s]["auroc"]))
        M(f"Pool{tag}Far", pct(pooled[s]["far"]))
        for d, dt in [("ragtruth_qa", "Qa"), ("ragtruth_summary", "Sum"), ("ragtruth_data2txt", "Dtt"), ("faithbench", "Fb"), ("halueval_qa", "Halu")]:
            if s in ft[d]:
                M(f"{dt}{tag}Auc", f(ft[d][s]["auroc"]))
                M(f"{dt}{tag}AucCi", ci(ft[d][s].get("auroc_ci")))
                M(f"{dt}{tag}Bacc", f(ft[d][s]["balanced_accuracy"]))
    fb = m.get("faithbench_best_label", {})
    for s, tag in [("minilm_mm", "Cos"), ("nli_mean_ec", "Nli"), ("risk_full", "Risk")]:
        if s in fb:
            M(f"FbBest{tag}Auc", f(fb[s]["auroc"]))
    if "judge_small" in fb:
        M("FbBestJudgeSmallBacc", f(fb["judge_small"]["balanced_accuracy"]))

    # EVOUNA (reference-based correctness)
    ev = ft["evouna_nq"]
    rows = ["lex_em", "lex_contains", "lex_f1", "lex_tfidf", "minilm_doc", "minilm_qa", "bge_small_doc", "bertscore", "nli_mean_ec",
            "risk_cos", "risk_full"]
    rows = [r for r in rows if r in ev]
    a = bold_best([ev[s]["auroc"] for s in rows], f)
    b = bold_best([ev[s]["balanced_accuracy"] for s in rows], f)
    L = [r"\begin{tabular}{lrrlrr}", r"\toprule", r"Method & AUC & BAcc & 95\% CI & FAR & FRR \\", r"\midrule"]
    for i, s in enumerate(rows):
        if s in ("minilm_doc", "risk_cos"):
            L.append(r"\midrule")
        L.append(f"{LAB[s]} & {a[i]} & {b[i]} & {ci(ev[s].get('bacc_ci'))} & {pct(ev[s]['far'])} & {pct(ev[s]['frr'])} \\\\")
    for judge in ["small", "strong"]:
        js = m["judged_subset"].get(judge, {}).get("evouna_nq")
        if not js:
            continue
        j = js[f"judge_{judge}"]
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{Judged subset, $n={j['n']}$}}}} \\")
        L.append(f"{LAB['judge_' + judge]} & {f(j.get('auroc'))} & {f(j['balanced_accuracy'])} & {ci(j.get('bacc_ci'))} & {pct(j['far'])} & {pct(j['frr'])} \\\\")
        for s in ["lex_contains", "minilm_doc", "risk_full"]:
            if s in js:
                L.append(f"{LAB[s]} & {f(js[s]['auroc'])} & {f(js[s]['balanced_accuracy'])} & {ci(js[s].get('bacc_ci'))} & {pct(js[s]['far'])} & {pct(js[s]['frr'])} \\\\")
        T = "Small" if judge == "small" else "Strong"
        M(f"EvJudge{T}Bacc", f(j["balanced_accuracy"]))
        M(f"EvJudge{T}BaccCi", ci(j.get("bacc_ci")))
        M(f"EvJudge{T}N", j["n"])
        pdv = m["paired_vs_judge"][judge]["evouna_nq"]
        for s, tag in [("lex_contains", "Contain"), ("minilm_doc", "Cos"), ("risk_full", "Risk")]:
            if s in pdv:
                M(f"Ev{tag}Minus{T}", f(pdv[s]["diff"]))
                M(f"Ev{tag}Minus{T}Ci", ci(pdv[s]["ci"]))
            if s in js:
                M(f"Ev{tag}Sub{T}Bacc", f(js[s]["balanced_accuracy"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_evouna.tex", L)
    for s, tag in [("lex_em", "Em"), ("lex_contains", "Contain"), ("lex_f1", "Tokf"), ("minilm_doc", "Cos"), ("risk_full", "Risk"), ("bertscore", "Bert"), ("nli_mean_ec", "Nli")]:
        if s in ev:
            M(f"Ev{tag}Auc", f(ev[s]["auroc"]))
            M(f"Ev{tag}Bacc", f(ev[s]["balanced_accuracy"]))
            M(f"Ev{tag}Far", pct(ev[s]["far"]))


def t_judged():
    m = R("main_performance")
    L = [r"\begin{tabular}{llrlrrrrrl}", r"\toprule",
         r" & & & \multicolumn{3}{c}{Judge} & \multicolumn{3}{c}{BAcc on the same examples} & Risk $-$ Judge \\",
         r"Judge & Domain & $n$ & BAcc [95\% CI] & FAR & FRR & MiniLM & NLI & Risk & $\Delta$BAcc [95\% CI] \\", r"\midrule"]
    for judge in ["small", "strong"]:
        js = m["judged_subset"].get(judge)
        if not js:
            continue
        T = "Small" if judge == "small" else "Strong"
        first = True
        for d in ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt", "faithbench"]:
            if d not in js:
                continue
            r = js[d]
            j = r[f"judge_{judge}"]
            pdv = m["paired_vs_judge"][judge][d]
            g = lambda s: f(r[s]["balanced_accuracy"]) if s in r else "--"
            dr = pdv.get("risk_full")
            L.append(f"{LAB['judge_' + judge] if first else ''} & {DOM[d]} & {j['n']} & {f(j['balanced_accuracy'])} {ci(j.get('bacc_ci'))} & {pct(j['far'])} & {pct(j['frr'])} & "
                     f"{g('minilm_mm')} & {g('nli_mean_ec')} & {g('risk_full')} & {f(dr['diff']) if dr else '--'} {ci(dr['ci']) if dr else ''} \\\\")
            first = False
            dt = {"ragtruth_qa": "Qa", "ragtruth_summary": "Sum", "ragtruth_data2txt": "Dtt", "faithbench": "Fb"}[d]
            M(f"{dt}Judge{T}Bacc", f(j["balanced_accuracy"]))
            M(f"{dt}Judge{T}Far", pct(j["far"]))
            M(f"{dt}Judge{T}Frr", pct(j["frr"]))
            M(f"{dt}Judge{T}N", j["n"])
        L.append(r"\midrule")
        meta = m["judge_meta"][judge]
        tot = sum(v["n"] for v in meta.values())
        M(f"Judge{T}Calls", f"{tot:,}")
        M(f"Judge{T}Failures", sum(v["failures"] for v in meta.values()))
        ins = sum(v["verdicts"].get("INSUFFICIENT_EVIDENCE", 0) for v in meta.values())
        M(f"Judge{T}InsufPct", pct(ins / tot))
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("tab_judged.tex", L)
    ag = m.get("judge_agreement") or {}
    for d, dt in [("ragtruth_qa", "Qa"), ("ragtruth_summary", "Sum"), ("ragtruth_data2txt", "Dtt"), ("faithbench", "Fb"), ("evouna_nq", "Ev")]:
        if d in ag:
            M(f"{dt}JudgeKappa", f(ag[d]["kappa_small_strong"], 2))
            M(f"{dt}StrongMinusSmall", f(ag[d]["diff_strong_minus_small"]["diff"]))
            M(f"{dt}StrongMinusSmallCi", ci(ag[d]["diff_strong_minus_small"]["ci"]))


def t_errors():
    e = R("error_types")
    keys = ["minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_full"]
    hasj = [j for j in ["small", "strong"] if any(f"judge_{j}" in v for v in e.values())]
    L = [r"\begin{tabular}{lr" + "r" * len(keys) + "r" * len(hasj) + "}", r"\toprule",
         "Annotated error type & $n$ & " + " & ".join(SHORT[k] for k in keys) + "".join(f" & {LAB['judge_' + j]}" for j in hasj) + r" \\", r"\midrule"]
    for name, row in e.items():
        if name == "Conflict only":
            L.append(r"\midrule")
        cells = [pct(row[k]["recall"]) if row["n"] else "--" for k in keys]
        jc = [(f"{pct(row['judge_' + j]['recall'])} ({row['judge_' + j]['n']})" if f"judge_{j}" in row else "--") for j in hasj]
        L.append(f"{name} & {row['n']} & " + " & ".join(cells + jc) + r" \\")
        tag = "".join(w.title() for w in name.split())
        for k, kt in [("minilm_mm", "Cos"), ("nli_mean_ec", "Nli"), ("risk_full", "Risk")]:
            if row["n"]:
                M(f"Err{tag}{kt}", pct(row[k]["recall"]))
                M(f"Err{tag}{kt}Far", pct(row[k]["far_at_alpha_threshold"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_errors.tex", L)


# ------------------------------------------------------------------------------------------ counterfactuals
TYPES = ["negation", "number", "date", "entity", "role_swap", "direction", "causal", "addition"]
TYPE_LAB = {"negation": "Neg.", "number": "Num.", "date": "Date", "entity": "Entity", "role_swap": "Role", "direction": "Dir.",
            "causal": "Causal", "addition": "Add."}


def t_cf():
    c = R("counterfactual")
    nb = (R("analysis_missing_columns") or {}).get("n_boot", CFG["stats"]["bootstrap"])
    bs = c["by_scorer"]
    rows = [s for s in ["lex_tok_prec", "minilm_doc", "minilm_mm", "minilm_min", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_cos", "risk_cos_checks", "risk_full"] if s in bs]
    n = bs["minilm_mm"]["bacc"]
    for block, key, name in [("drop", "score_drop_rate", "tab_cf_drop.tex"), ("det", "paired_detection", "tab_cf_detect.tex")]:
        L = [r"\begin{tabular}{l" + "r" * (len(TYPES) + 1) + ("rr" if block == "drop" else "rr") + "}", r"\toprule",
             "Method & " + " & ".join(TYPE_LAB[t] for t in TYPES) + " & All"
             + (r" & Inv. & Para.\ $\Delta$" if block == "drop" else r" & Orig.\ acc. & Para.\ acc.") + r" \\",
             "$n$ pairs & " + " & ".join(str(n[t]["n"]) for t in TYPES) + f" & {n['ALL']['n']} & " +
             (f"{bs['minilm_mm']['inversion']['n']} & {bs['minilm_mm']['paraphrase']['n']}" if block == "drop" else f"{n['ALL']['n']} & {bs['minilm_mm']['paraphrase']['n']}") + r" \\", r"\midrule"]
        for s in rows:
            b = bs[s]["bacc"]
            cells = [pct(b[t][key]) if t in b else "--" for t in TYPES] + [pct(b["ALL"][key])]
            if block == "drop":
                cells += [pct(bs[s]["inversion"]["rate"]) if "inversion" in bs[s] else "--",
                          f(bs[s]["paraphrase"]["delta_sd"], 2) if "paraphrase" in bs[s] else "--"]
            else:
                cells += [pct(b["ALL"]["original_acceptance"]), pct(bs[s]["paraphrase"]["acceptance"]) if "paraphrase" in bs[s] else "--"]
            if s in ("bertscore", "risk_cos"):
                L.append(r"\midrule")
            L.append(LAB[s] + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(name, L)
    for s, tag in [("minilm_mm", "Cos"), ("bge_small_mm", "Bge"), ("bertscore", "Bert"), ("nli_mean_ec", "Nli"), ("risk_full", "Risk"), ("lex_tok_prec", "TokPrec"), ("risk_cos_checks", "RiskChk")]:
        if s not in bs:
            continue
        b = bs[s]["bacc"]["ALL"]
        M(f"Cf{tag}Detect", pct(b["paired_detection"]))
        M(f"Cf{tag}DetectCi", ci(b["paired_detection_ci"], p=True))
        M(f"Cf{tag}FlipRej", pct(b["flip_rejection"]))
        M(f"Cf{tag}OrigAcc", pct(b["original_acceptance"]))
        M(f"Cf{tag}Drop", pct(b["score_drop_rate"]))
        M(f"Cf{tag}DetectFar", pct(bs[s]["far"]["ALL"]["paired_detection"]))
        M(f"Cf{tag}OrigAccFar", pct(bs[s]["far"]["ALL"]["original_acceptance"]))
        for t in TYPES:
            if t in bs[s]["bacc"]:
                tt = t.replace("_", " ").title().replace(" ", "")
                M(f"Cf{tag}Detect{tt}", pct(bs[s]["bacc"][t]["paired_detection"]))
                M(f"Cf{tag}Drop{tt}", pct(bs[s]["bacc"][t]["score_drop_rate"]))
                M(f"Cf{tag}Sd{tt}", f(bs[s]["delta_sd_by_type"][t], 2))
        if "inversion" in bs[s]:
            M(f"Cf{tag}Inv", pct(bs[s]["inversion"]["rate"]))
            M(f"Cf{tag}InvCi", ci(bs[s]["inversion"]["ci"], p=True))
            M(f"Cf{tag}ParaAcc", pct(bs[s]["paraphrase"]["acceptance"]))
            M(f"Cf{tag}ParaOrigAcc", pct(bs[s]["paraphrase"]["original_acceptance"]))
            M(f"Cf{tag}ParaFalseRej", pct(bs[s]["paraphrase"]["false_rejection_given_original_accepted"]))
            M(f"Cf{tag}ParaSd", f(bs[s]["paraphrase"]["delta_sd"], 2))
        if "homogeneity" in bs[s]:
            h = bs[s]["homogeneity"]
            M(f"Cf{tag}Chi", f"{h['chi2']:.1f}")
            M(f"Cf{tag}CramerV", f(h["cramers_v"], 2))
        dl = bs[s]["dilution"]
        ks = list(dl)
        M(f"Cf{tag}DilFirst", f(dl[ks[1]]["mean_drop_sd"], 2))
        M(f"Cf{tag}DilLast", f(dl[ks[-1]]["mean_drop_sd"], 2))
    M("NCfInv", bs["minilm_mm"]["inversion"]["n"])
    M("CfDilFirstBin", list(bs["minilm_mm"]["dilution"])[1].replace("-", "--"))

    # judges and cheap scorers on identical pairs
    js = c["judges"]
    if js:
        L = [r"\begin{tabular}{llr" + "r" * (len(TYPES) + 1) + "rr}", r"\toprule",
             "Pairs & Evaluator & $n$ & " + " & ".join(TYPE_LAB[t] for t in TYPES) + r" & All & Orig.\ acc. & Para.\ acc. \\", r"\midrule"]
        for judge in ["small", "strong"]:
            if judge not in js:
                continue
            T = "Small" if judge == "small" else "Strong"
            v = js[judge]["verdict"]
            para = js[judge].get("paraphrase", {})
            L.append(f"{LAB['judge_' + judge]} subset & {LAB['judge_' + judge]} & {v['ALL']['n']} & " + " & ".join(pct(v[t]["paired_detection"], 0) if t in v else "--" for t in TYPES)
                     + f" & {pct(v['ALL']['paired_detection'])} & {pct(v['ALL']['original_acceptance'])} & {pct(para.get('acceptance'))} \\\\")
            L.append(" & $n$ per type & & " + " & ".join(str(v[t]["n"]) if t in v else "--" for t in TYPES) + f" & & & {para.get('n', '--')} \\\\")
            for s, r in js[judge]["same_pairs"].items():
                L.append(f" & {SHORT[s]} & {v['ALL']['n']} & " + " & ".join([""] * len(TYPES)) + f" & {pct(r['paired_detection'])} & & \\\\")
                tg = {"minilm_mm": "Cos", "bge_small_mm": "Bge", "bertscore": "Bert", "nli_mean_ec": "Nli", "risk_full": "Risk"}[s]
                M(f"CfJ{T}Minus{tg}", pct(r["judge_minus_scorer"]))
                M(f"CfJ{T}Minus{tg}Ci", ci(r["ci"], p=True))
                M(f"CfJ{T}Minus{tg}P", pv(r["p_boot"], nb))
                M(f"CfJ{T}Same{tg}Detect", pct(r["paired_detection"]))
            L.append(r"\midrule")
            M(f"CfJ{T}N", v["ALL"]["n"])
            M(f"CfJ{T}Bases", js[judge]["n_originals"])
            M(f"CfJ{T}Detect", pct(v["ALL"]["paired_detection"]))
            M(f"CfJ{T}DetectCi", ci(v["ALL"]["paired_detection_ci"], p=True))
            M(f"CfJ{T}FlipRej", pct(v["ALL"]["flip_rejection"]))
            M(f"CfJ{T}OrigAcc", pct(js[judge]["original_acceptance_all"]))
            if para:
                M(f"CfJ{T}ParaAcc", pct(para["acceptance"]))
                M(f"CfJ{T}ParaN", para["n"])
            iv = js[judge].get("inversion_verdict")
            if iv:
                M(f"CfJ{T}InvN", iv["n"])
                M(f"CfJ{T}CorrectOrder", pct(iv["correct_order"]))
                M(f"CfJ{T}WrongOrder", pct(iv["flip_accepted_paraphrase_rejected"]))
                M(f"CfJ{T}PsupInv", pct(iv["psup_inversion"]))
        L[-1] = r"\bottomrule"
        L.append(r"\end{tabular}")
        write("tab_cf_judge.tex", L)


def t_examples():
    """Representative matched-locus triplets with evaluator outputs, chosen deterministically (first by id per type)."""
    p = RESULTS / "scores_all.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p)
    cf = df[df["dataset"] == "counterfactual"]
    recs = {r["id"]: r for r in read_jsonl(PROC / "counterfactuals.jsonl")}
    para = cf[cf["cf_type"] == "paraphrase"].set_index("cf_base")
    orig = cf[cf["cf_type"] == "original"].set_index("cf_base")
    jc = "judge_small_verdict" if "judge_small_verdict" in cf else None
    L = [r"\begin{tabular}{p{0.07\textwidth}p{0.58\textwidth}rrrl}", r"\toprule",
         r"Variant & Target sentence & MiniLM & NLI & Risk & Judge (3B) \\", r"\midrule"]
    short = {"SUPPORTED": "SUP", "UNSUPPORTED": "UNSUP", "INSUFFICIENT_EVIDENCE": "INSUF"}
    n = 0
    for t in ["number", "negation", "entity", "role_swap", "causal"]:
        e = cf[(cf["cf_type"] == t) & cf["cf_matched"] & cf["cf_base"].isin(para.index)]
        if jc:
            e = e[e[jc].notna() & e["cf_base"].isin(para[para[jc].notna()].index)]
        e = e[e.index.map(lambda i: len(recs[i]["meta"]["edited_sentence"]) < 230)]
        if not len(e):
            continue
        i = sorted(e.index)[0]
        b = e.loc[i, "cf_base"]
        m = recs[i]["meta"]
        pid = para.loc[[b]].index[0]
        prow, orow, frow = para.loc[b], orig.loc[b], e.loc[i]
        ptxt = recs[f"cf-{b}-paraphrase"]["meta"].get("edited_sentence") or recs[f"cf-{b}-paraphrase"]["meta"].get("paraphrase_sentence", "")
        for lab, txt, row in [("original", m["original_sentence"], orow), ("paraphrase", ptxt, prow), (t.replace("_", " "), m["edited_sentence"], frow)]:
            jv = short.get(row[jc], "--") if jc and isinstance(row[jc], str) else "--"
            L.append(f"{lab} & {esc(txt)} & {f(row['minilm_mm'])} & {f(row.get('nli_mean_ec'))} & {f(-row['risk_full'])} & {jv} \\\\")
        L.append(r"\midrule")
        n += 1
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("tab_examples.tex", L)


# ------------------------------------------------------------------------------------------------ transfer
def t_transfer():
    t = R("transfer")
    a = str(CFG["router"]["primary_alpha"])
    doms = ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt", "faithbench", "halueval_qa"]
    for s, name in [("minilm_mm", "tab_transfer_cos.tex"), ("risk_full", "tab_transfer_risk.tex"), ("nli_mean_ec", "tab_transfer_nli.tex")]:
        if s not in t["matrix"]:
            continue
        mat = t["matrix"][s]
        L = [r"\begin{tabular}{l" + "r" * len(doms) + "}", r"\toprule",
             r"Calibrated on $\downarrow$ / tested on $\rightarrow$ & " + " & ".join(DOM[d] for d in doms) + r" \\", r"\midrule"]
        for src in doms:
            cells = []
            for tgt in doms:
                c = mat[f"{src}->{tgt}"]
                v = pct(c[f"far@{a}"])
                if c[f"far_ci@{a}"][0] > float(a):
                    v = r"\textbf{" + v + "}"
                cells.append(f"{v} ({pct(c[f'accept_rate@{a}'], 0)})")
            L.append(f"{DOM[src]} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(name, L)
    for s, tag in [("minilm_mm", "Cos"), ("bge_small_mm", "Bge"), ("bertscore", "Bert"), ("nli_mean_ec", "Nli"), ("risk_full", "Risk"), ("risk_cos", "RiskCos")]:
        if s not in t["h3"]:
            continue
        h = t["h3"][s][a]
        M(f"Tr{tag}OffExceed", h["offdiag_far_exceeds_alpha_ci"])
        M(f"Tr{tag}OffMax", pct(h["offdiag_far_max"]))
        M(f"Tr{tag}OffMedian", pct(h["offdiag_far_median"]))
        M(f"Tr{tag}DiagExceed", h["diag_far_exceeds_alpha_ci"])
        M(f"Tr{tag}DiagMax", pct(h["diag_far_max"]))
        M(f"Tr{tag}BaccDrop", f(h["mean_bacc_drop_offdiag"]))
        mat = t["matrix"][s]
        rt = ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt"]
        off = [mat[f"{x}->{y}"] for x in rt for y in rt if x != y]
        M(f"Tr{tag}RtExceed", sum(c[f"far_ci@{a}"][0] > float(a) for c in off))
        M(f"Tr{tag}RtMax", pct(max(c[f"far@{a}"] for c in off)))
        M(f"Tr{tag}ToFbMax", pct(max(mat[f"{x}->faithbench"][f"far@{a}"] for x in rt)))
        M(f"Tr{tag}ToFbMin", pct(min(mat[f"{x}->faithbench"][f"far@{a}"] for x in rt)))
        M(f"Tr{tag}ToHaluMax", pct(max(mat[f"{x}->halueval_qa"][f"far@{a}"] for x in rt)))
        M(f"Tr{tag}ToHaluMin", pct(min(mat[f"{x}->halueval_qa"][f"far@{a}"] for x in rt)))
    M("TrOffCells", t["h3"]["minilm_mm"][a]["offdiag_cells"])
    # generator and length shift
    L = [r"\begin{tabular}{lllrrr}", r"\toprule", r"Shift & Method & Calibrated $\rightarrow$ tested & FAR [95\% CI] & FRR & AUC \\", r"\midrule"]
    for name, lab in [("generator_shift", "Generator"), ("length_shift", "Length")]:
        for s in ["minilm_mm", "nli_mean_ec", "risk_full"]:
            if s not in t[name]:
                continue
            for k, c in t[name][s].items():
                a_, b_ = k.split("->")
                L.append(f"{lab} & {SHORT[s]} & {esc(a_)} $\\rightarrow$ {esc(b_)} & {pct(c['far'])} {ci(c['far_ci'], p=True)} & {pct(c['frr'])} & {f(c['auroc'])} \\\\")
                tg = {"minilm_mm": "Cos", "nli_mean_ec": "Nli", "risk_full": "Risk"}[s]
                M(f"Sh{lab[:3]}{tg}{a_.replace('_', ' ').title().replace(' ', '')}To{b_.replace('_', ' ').title().replace(' ', '')}", pct(c["far"]))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("tab_shift.tex", L)
    rp = t["router_trained_per_domain"]
    rt = ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt"]
    M("TrRouterSrcOffMax", pct(max(rp[f"{x}->{y}"]["far"] for x in rt for y in rt if x != y)))
    M("TrRouterSrcOffAucMin", f(min(rp[f"{x}->{y}"]["auroc"] for x in rt for y in rt if x != y)))
    M("TrRouterSrcDiagAucMin", f(min(rp[f"{x}->{x}"]["auroc"] for x in rt)))


# ---------------------------------------------------------------------------------- router and cascade
def t_router():
    r = R("router_analysis")
    ab = r["ablation"]
    L = [r"\begin{tabular}{lrlrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{RAGTruth test AUC} & FaithB. & HaluEval & \multicolumn{2}{c}{Counterfactual} \\",
         r"Feature set & AUC & $\Delta$ vs.\ full [95\% CI] & AUC & AUC & drop\,\% & AUC \\", r"\midrule"]
    for v, row in ab.items():
        if v in ("minus_cosine", "only_cosine"):
            L.append(r"\midrule")
        name = "All groups (full)" if v == "full" else v.replace("minus_", "$-$ ").replace("only_", "only ")
        d = f"{f(-row['auroc_loss_vs_full'])} {ci([-row['auroc_loss_ci'][1], -row['auroc_loss_ci'][0]])}" if "auroc_loss_vs_full" in row else ""
        L.append(f"{name} & {f(row['ragtruth_auroc'])} & {d} & {f(row['faithbench_auroc'])} & {f(row['halueval_qa_auroc'])} & {pct(row['cf_score_drop_rate'])} & {f(row['cf_auroc_edit_vs_original'])} \\\\")
        tag = "".join(w.title() for w in v.split("_"))
        M(f"Abl{tag}Auc", f(row["ragtruth_auroc"]))
        M(f"Abl{tag}CfAuc", f(row["cf_auroc_edit_vs_original"]))
        M(f"Abl{tag}FbAuc", f(row["faithbench_auroc"]))
        M(f"Abl{tag}HaluAuc", f(row["halueval_qa_auroc"]))
        if "auroc_loss_vs_full" in row:
            M(f"Abl{tag}Loss", f(row["auroc_loss_vs_full"]))
            M(f"Abl{tag}LossCi", ci(row["auroc_loss_ci"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", L)
    for d, dt in [("ragtruth_qa", "Qa"), ("ragtruth_summary", "Sum"), ("ragtruth_data2txt", "Dtt"), ("faithbench", "Fb"), ("halueval_qa", "Halu")]:
        c = r["calibration"][d]["risk_full"]
        M(f"Cal{dt}Ece", f(c["ece"]))
        M(f"Cal{dt}Brier", f(c["brier"]))
        g = r["gbm_check"][d]
        M(f"Gbm{dt}Auc", f(g["gbm_auroc"]))
        M(f"Lr{dt}Auc", f(g["lr_auroc"]))
    co = r["coefficients"]["risk_full"]
    top = sorted(co.items(), key=lambda kv: -abs(kv[1]))
    L = [r"\begin{tabular}{lr}", r"\toprule", r"Feature (standardised) & Coefficient \\", r"\midrule"]
    L += [f"\\texttt{{{esc(k)}}} & {v:+.3f} \\\\" for k, v in top]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_coef.tex", L)


def t_cascade():
    k = R("cascade")
    nb = (R("analysis_missing_columns") or {}).get("n_boot", CFG["stats"]["bootstrap"])
    gates = [("minilm_mm", "Cos"), ("bge_small_mm", "Bge"), ("nli_mean_ec", "Nli"), ("risk_cos", "RiskCos"), ("risk_cos_checks", "RiskChk"), ("risk_full", "Risk")]
    # cheap stage on full test
    L = [r"\begin{tabular}{llrlrlr}", r"\toprule",
         r"$\alpha$ & Gate & Resolved\,\% & 95\% CI & Cheap FAR\,\% & 95\% CI & Cheap FRR\,\% \\", r"\midrule"]
    for a in k["cheap_stage_full_test"]:
        for s, tg in gates:
            c = k["cheap_stage_full_test"][a].get(s)
            if not c:
                continue
            L.append(f"{a} & {LAB[s]} & {pct(c['coverage'])} & {ci(c['coverage_ci'], p=True)} & {pct(c['cheap_far'])} & {ci(c['cheap_far_ci'], p=True)} & {pct(c['cheap_frr'])} \\\\")
            an = {"0.05": "Five", "0.1": "Ten", "0.2": "Twenty"}[a]
            M(f"Cs{tg}{an}Cov", pct(c["coverage"]))
            M(f"Cs{tg}{an}CovCi", ci(c["coverage_ci"], p=True))
            M(f"Cs{tg}{an}Far", pct(c["cheap_far"]))
            M(f"Cs{tg}{an}FarCi", ci(c["cheap_far_ci"], p=True))
            M(f"Cs{tg}{an}Frr", pct(c["cheap_frr"]))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("tab_cheapstage.tex", L)
    h = k["h4"]["0.05"]
    for s, tg in [("risk_full", "Risk"), ("risk_cos_checks", "RiskChk"), ("nli_mean_ec", "Nli"), ("bge_small_mm", "Bge")]:
        x = h[f"{s}_minus_minilm_mm_coverage"]
        M(f"Hfour{tg}Diff", pct(x["diff"]))
        M(f"Hfour{tg}DiffCi", ci(x["ci"], p=True))
        M(f"Hfour{tg}P", pv(x["p_boot"], nb))
    # system on judged subsets
    for judge in ["small", "strong"]:
        sj = k["system_judged_subset"].get(judge)
        if not sj:
            continue
        T = "Small" if judge == "small" else "Strong"
        aj = sj["always_judge"]
        L = [r"\begin{tabular}{llrlrrlr}", r"\toprule",
             r"Policy & Gate & Judged\,\% & BAcc [95\% CI] & FAR\,\% & FRR\,\% & $\Delta$BAcc vs.\ always-judge & Random \\", r"\midrule",
             f"Always judge & -- & 100.0 & {f(aj['balanced_accuracy'])} {ci(aj.get('bacc_ci'))} & {pct(aj['far'])} & {pct(aj['frr'])} & -- & -- \\\\", r"\midrule"]
        for s in ["lex_tok_prec", "minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_full"]:
            c = sj.get(f"always_cheap_{s}")
            if c:
                L.append(f"Never judge & {LAB[s]} & 0.0 & {f(c['balanced_accuracy'])} {ci(c['bacc_ci'])} & {pct(c['far'])} & {pct(c['frr'])} & {f(c['vs_judge']['diff'])} {ci(c['vs_judge']['ci'])} & -- \\\\")
                tg = {"lex_tok_prec": "TokPrec", "minilm_mm": "Cos", "bge_small_mm": "Bge", "bertscore": "Bert", "nli_mean_ec": "Nli", "risk_full": "Risk"}[s]
                M(f"Sys{T}Never{tg}Bacc", f(c["balanced_accuracy"]))
                M(f"Sys{T}Never{tg}Diff", f(c["vs_judge"]["diff"]))
                M(f"Sys{T}Never{tg}DiffCi", ci(c["vs_judge"]["ci"]))
        for a in ["0.05", "0.1", "0.2"]:
            L.append(r"\midrule")
            for s, tg in gates:
                c = sj.get(a, {}).get(s)
                if not c or s in ("bge_small_mm", "risk_cos"):
                    continue
                L.append(f"Gate, $\\alpha={a}$ & {LAB[s]} & {pct(c['escalation'])} & {f(c['balanced_accuracy'])} {ci(c['bacc_ci'])} & {pct(c['far'])} & {pct(c['frr'])} & "
                         f"{f(c['bacc_minus_always_judge'])} {ci(c['bacc_minus_always_judge_ci'])} & {f(c['random_escalation']['balanced_accuracy'])} \\\\")
                an = {"0.05": "Five", "0.1": "Ten", "0.2": "Twenty"}[a]
                M(f"Sys{T}{tg}{an}Esc", pct(c["escalation"]))
                M(f"Sys{T}{tg}{an}Saved", pct(1 - c["escalation"]))
                M(f"Sys{T}{tg}{an}Bacc", f(c["balanced_accuracy"]))
                M(f"Sys{T}{tg}{an}BaccCi", ci(c["bacc_ci"]))
                M(f"Sys{T}{tg}{an}Far", pct(c["far"]))
                M(f"Sys{T}{tg}{an}FarCi", ci(c["far_ci"], p=True))
                M(f"Sys{T}{tg}{an}Diff", f(c["bacc_minus_always_judge"]))
                M(f"Sys{T}{tg}{an}DiffCi", ci(c["bacc_minus_always_judge_ci"]))
                M(f"Sys{T}{tg}{an}Rand", f(c["random_escalation"]["balanced_accuracy"]))
                M(f"Sys{T}{tg}{an}JudgeSec", f"{c['judge_seconds_per_1000']:.0f}")
        L += [r"\bottomrule", r"\end{tabular}"]
        write(f"tab_cascade_{judge}.tex", L)
        M(f"Sys{T}N", sj["n"])
        M(f"Sys{T}JudgeBacc", f(aj["balanced_accuracy"]))
        M(f"Sys{T}JudgeBaccCi", ci(aj.get("bacc_ci")))
        M(f"Sys{T}JudgeFar", pct(aj["far"]))
        M(f"Sys{T}JudgeFrr", pct(aj["frr"]))
        cu = k["curves"].get(judge, {})
        for s, tg in [("minilm_mm", "Cos"), ("risk_full", "Risk"), ("nli_mean_ec", "Nli")]:
            if s in cu:
                g, r_ = np.array(cu[s]["gated"]), np.array(cu[s]["random"])
                M(f"Cur{T}{tg}MaxGain", f(float((g - r_).max())))
                M(f"Cur{T}{tg}MeanGain", f(float((g - r_)[1:-1].mean())))
                M(f"Cur{T}{tg}Aurc", f(cu[s]["aurc"]))
    for src, st in [("in_domain", "In"), ("from_ragtruth_summary", "Ood")]:
        for s, tg in [("minilm_mm", "Cos"), ("risk_full", "Risk")]:
            c = k["ood_faithbench"][src][s]
            M(f"Fb{st}{tg}Cov", pct(c["coverage"]))
            M(f"Fb{st}{tg}Far", pct(c["cheap_far"]))
            M(f"Fb{st}{tg}Frr", pct(c["cheap_frr"]))
    for s, tg in [("minilm_doc", "Cos"), ("lex_f1", "Tokf"), ("risk_full", "Risk")]:
        c = k["evouna"][s]
        M(f"EvCs{tg}Cov", pct(c["coverage"]))
        M(f"EvCs{tg}Far", pct(c["cheap_far"]))
        M(f"EvCs{tg}Frr", pct(c["cheap_frr"]))
        if "system_bacc" in c:
            M(f"EvCs{tg}SysBacc", f(c["system_bacc"]))
            M(f"EvCs{tg}Esc", pct(c["escalation"]))


def t_cost():
    c = R("cost")
    if not c:
        return
    me = c["methods"]
    names = [("lexical_and_checks", "Lexical features + checks"), ("minilm_cosine", "MiniLM cosine"),
             ("minilm_cosine_evidence_cached", r"\quad evidence embeddings cached"), ("bge_small_cosine", "BGE-small cosine"),
             ("bge_small_cosine_evidence_cached", r"\quad evidence embeddings cached"), ("risk_full", "Risk model: full"),
             ("bertscore", "BERTScore (roberta-base)"), ("nli", "NLI (DeBERTa-v3-base)"),
             ("judge_small", "Judge (3B), new source"), ("judge_small_prefix_cached", r"\quad same source (prefix cached)"),
             ("judge_strong", "Judge (14B), new source"), ("judge_strong_prefix_cached", r"\quad same source (prefix cached)")]
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Method & Median ms & p95 ms & Wall s / 1k & CPU s / 1k & Cold start s & Prompt tok. & Out tok. \\", r"\midrule"]
    base = me["minilm_cosine"]["wall_s_per_1000"]
    for key, lab in names:
        if key not in me:
            continue
        v = me[key]
        if key in ("bertscore", "judge_small", "judge_strong"):
            L.append(r"\midrule")
        g = lambda x, d=0: "--" if v.get(x) is None else f"{v[x]:,.{d}f}"
        L.append(f"{lab} & {g('median_ms', 1 if v['median_ms'] < 100 else 0)} & {g('p95_ms', 1 if v['p95_ms'] < 100 else 0)} & {g('wall_s_per_1000', 1 if v['wall_s_per_1000'] < 100 else 0)} & {g('cpu_s_per_1000')} & {g('cold_start_s', 1)} & {g('prompt_tokens')} & {g('output_tokens', 1)} \\\\")
        tag = "".join(w.title() for w in key.split("_"))
        M(f"Cost{tag}Median", g("median_ms", 1 if v["median_ms"] < 100 else 0))
        M(f"Cost{tag}Wall", g("wall_s_per_1000", 1 if v["wall_s_per_1000"] < 100 else 0))
        M(f"Cost{tag}Ratio", f"{v['wall_s_per_1000'] / base:,.0f}" if v["wall_s_per_1000"] / base >= 10 else f"{v['wall_s_per_1000'] / base:.1f}")
        if v.get("prompt_tokens") is not None:
            M(f"Cost{tag}Ptok", g("prompt_tokens"))
            M(f"Cost{tag}TokPerS", g("prompt_tokens_per_s"))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_cost.tex", L)
    hw = c["hardware"]
    M("HwCpu", esc(hw["cpu"]))
    M("HwCores", hw["cores_logical"])
    M("HwRam", hw["ram_gb"])
    M("HwThreads", hw["torch_threads"])
    M("CostSampleN", c["sample"]["n"])
    M("CostAnsWords", f"{c['sample']['mean_answer_words']:.0f}")
    M("CostEvWords", f"{c['sample']['mean_evidence_words']:.0f}")


def t_sensitivity():
    s = R("prompt_sensitivity")
    if not s:
        return
    M("SensN", s["n"])
    M("SensAgree", pct(s["agreement"]))
    M("SensKappa", f(s["kappa_v1_v2"], 2))
    M("SensBaccOne", f(s["v1"]["balanced_accuracy"]))
    M("SensBaccTwo", f(s["v2"]["balanced_accuracy"]))
    M("SensFarOne", pct(s["v1"]["far"]))
    M("SensFarTwo", pct(s["v2"]["far"]))
    M("SensFrrOne", pct(s["v1"]["frr"]))
    M("SensFrrTwo", pct(s["v2"]["frr"]))


def t_hyp():
    """Preregistered hypothesis tests. A p-value at the resampling floor is printed as an upper bound."""
    h = R("hypotheses")
    nb = (R("analysis_missing_columns") or {}).get("n_boot", CFG["stats"]["bootstrap"])
    raw, holm = h["raw_p"], h["holm_adjusted_p"]
    for key, tg in [("H1_judge_vs_cosine", "HoneJudge"), ("H1_nli_vs_cosine", "HoneNli"), ("H3_transfer", "Hthree"), ("H4_coverage", "Hfour")]:
        if key not in raw:
            continue
        floor = raw[key] <= 1 / nb + 1e-12
        M(f"{tg}Holm", (r"\le " if floor else "= ") + f"{holm[key]:.3f}")
        M(f"{tg}Raw", (r"\le " if floor else "= ") + f"{raw[key]:.3f}")
    h1 = h["H1"]
    jm = h1.get("judge_minus_cosine")
    if jm:
        M("HoneJudgeMinusCos", pct(jm["judge_minus_scorer"]))
        M("HoneJudgeMinusCosCi", ci(jm["ci"], p=True))
    x = h["H1_nli_minus_cosine_all_pairs"]
    M("HoneNliMinusCos", pct(x["diff"]))
    M("HoneNliMinusCosCi", ci(x["ci"], p=True))
    for s, tg in [("minilm_mm", "Cos"), ("bge_small_mm", "Bge"), ("bertscore", "Bert"), ("nli_mean_ec", "Nli"), ("risk_full", "Risk")]:
        y = h["H2"].get(s)
        if y:
            M(f"Htwo{tg}Perm", pv(y["p_perm_within_base"], nb))
            M(f"Htwo{tg}RejLo", pct(y["rejection_range"][0]))
            M(f"Htwo{tg}RejHi", pct(y["rejection_range"][1]))
    t3 = h["H3"]
    M("HthreeExceed", t3["offdiag_far_exceeds_alpha_ci"])
    M("HthreeCells", t3["offdiag_cells"])
    M("HthreePBonf", f"{t3['p_bonferroni']:.3f}")
    s4 = h["H4"]["system"]
    M("HfourSysDiff", f(s4["bacc_minus_always_judge"]))
    M("HfourSysDiffCi", ci(s4["bacc_minus_always_judge_ci"]))


def t_prompts():
    import run_judge as J

    L = []
    for k, title in [("v1", "Evidence-grounded rubric (primary, v1)"), ("v2", "Evidence-grounded rubric (sensitivity variant, v2)"), ("ref", "Reference-based rubric (EVOUNA-NQ)")]:
        L.append(rf"\paragraph{{{title}.}}")
        L.append(r"\begin{quote}\small\ttfamily\raggedright")
        L.append(esc(J.RUBRIC[k]).replace("\n", r"\\" + "\n").replace('"', r"\textquotedbl{}"))
        L.append(r"\end{quote}")
    write("prompts.tex", L)
    M("JudgeSmallModel", esc(CFG["judges"]["small"]["model"]))
    M("JudgeStrongModel", esc(CFG["judges"]["strong"]["model"]))
    M("Seed", CFG["seed"] if "seed" in CFG else J.SEED)
    M("NBoot", f"{CFG['stats']['bootstrap']:,}")


def main():
    miss = R("analysis_missing_columns") or {}
    for fn in [t_datasets, t_main, t_judged, t_errors, t_cf, t_examples, t_transfer, t_router, t_cascade, t_cost, t_sensitivity, t_hyp, t_prompts]:
        try:
            fn()
        except Exception as ex:  # a missing input must be visible, not silently skipped
            print(f"!! {fn.__name__}: {type(ex).__name__}: {ex}")
    lines = ["% generated by src/make_tables.py -- do not edit"]
    if miss.get("missing") or miss.get("n_boot") != CFG["stats"]["bootstrap"]:
        lines.append(f"% WARNING: built from a debugging analysis: {miss}")
        print("WARNING: numbers come from a debugging analysis:", miss)
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    write("numbers.tex", lines)
    (RESULTS / "numbers.json").write_text(json.dumps(MACROS, indent=1, sort_keys=True), encoding="utf8")
    print(len(MACROS), "macros;", len(list(GEN.glob("tab_*.tex"))), "tables written to", GEN)


if __name__ == "__main__":
    main()
