"""All statistical analyses. Reads score tables and judge logs, writes results/*.json.

  python src/analyze.py            # everything
  python src/analyze.py --fast     # 200 bootstrap resamples (debugging only)
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import analysis_lib as L
from common import (CFG, PROC, RESULTS, SEED, auroc, binary_metrics, cluster_bootstrap, ece_brier, read_jsonl,
                    thr_far, thr_frr, thr_max_bacc)

ALPHAS = CFG["router"]["alphas"]
ALPHA = CFG["router"]["primary_alpha"]
BETA = CFG["router"]["beta_false_reject"]
EVID_SCORERS = ["lex_f1", "lex_tfidf", "lex_tok_prec", "minilm_doc", "minilm_mm", "minilm_min", "bge_small_doc",
                "bge_small_mm", "bge_small_min", "bertscore", "nli_mean_ec", "nli_min_e", "risk_cos",
                "risk_cos_checks", "risk_full"]
REF_SCORERS = ["lex_em", "lex_contains", "lex_f1", "lex_tfidf", "minilm_doc", "minilm_qa", "bge_small_doc",
               "bertscore", "nli_mean_ec", "risk_cos", "risk_full"]
KEY = ["minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_full"]
NB = None


def split(df, domain, part):
    return df[(df["domain"] == domain) & (df["split"] == part) & (df["dataset"] != "counterfactual")]


# =============================================================== 1. main comparison (RQ1)
def main_performance(df):
    out = {"full_test": {}, "judged_subset": {}, "paired_vs_judge": {}, "judge_meta": {}, "pooled_ragtruth": {}}
    for dom in L.FAITH_DOMAINS + ["evouna_nq"]:
        te, ca = split(df, dom, "test"), split(df, dom, "calib")
        scorers = REF_SCORERS if dom == "evouna_nq" else EVID_SCORERS
        out["full_test"][dom] = {s: L.eval_scorer(te, ca, s, n_boot=NB) for s in scorers if s in df and te[s].notna().any()}
        for judge in ["small", "strong"]:
            pc = f"judge_{judge}_pred"
            if pc not in df or te[pc].notna().sum() == 0:
                continue
            sub = te[te[pc].notna()]
            res = {f"judge_{judge}": L.eval_pred(sub, pc, n_boot=NB)}
            ps = sub[f"judge_{judge}_psup"]
            if ps.notna().all():
                res[f"judge_{judge}"]["auroc"] = auroc(sub["label"].values, -ps.values)
            for s in scorers:
                if s in df and sub[s].notna().all():
                    res[s] = L.eval_scorer(sub, ca, s, n_boot=NB)
            out["judged_subset"].setdefault(judge, {})[dom] = res
            pd_ = {}
            for s in [x for x in ["minilm_mm" if dom != "evouna_nq" else "minilm_doc", "nli_mean_ec", "risk_full",
                                  "lex_contains"] if x in res]:
                pred_s = (sub[s].values < res[s]["threshold"]).astype(int)
                pd_[s] = L.paired_diff(sub, pred_s, sub[pc].values, n_boot=NB)
            out["paired_vs_judge"].setdefault(judge, {})[dom] = pd_
            v = sub[f"judge_{judge}_verdict"].fillna("FAILED")
            ins = v == "INSUFFICIENT_EVIDENCE"
            out["judge_meta"].setdefault(judge, {})[dom] = dict(
                verdicts=v.value_counts().to_dict(), n=int(len(sub)),
                insufficient_share=float(ins.mean()),
                insufficient_label_unsupported=float(sub.loc[ins, "label"].mean()) if ins.any() else None,
                failures=int((v == "FAILED").sum()))
    # pooled RAGTruth (per-domain thresholds, pooled decisions)
    te = pd.concat([split(df, d, "test") for d in L.RAG_DOMAINS])
    for s in EVID_SCORERS:
        pred = np.zeros(len(te), dtype=int)
        for d in L.RAG_DOMAINS:
            ca = split(df, d, "calib")
            m = (te["domain"] == d).values
            pred[m] = (te.loc[m, s].values < thr_max_bacc(ca["label"].values, ca[s].values)).astype(int)
        y, g = te["label"].values, te["group"].values
        r = {k: v for k, v in binary_metrics(y, pred).items()}
        r["auroc"] = auroc(y, -te[s].values)
        _, lo, hi, _ = cluster_bootstrap(g, lambda i: binary_metrics(y[i], pred[i])["balanced_accuracy"], NB)
        r["bacc_ci"] = [float(lo), float(hi)]
        out["pooled_ragtruth"][s] = r
    # both judges on their common subset
    if "judge_strong_pred" in df:
        both = df[df["judge_strong_pred"].notna() & df["judge_small_pred"].notna() & (df["dataset"] != "counterfactual")]
        out["judge_agreement"] = {
            d: dict(n=int(len(b)), kappa_small_strong=binary_metrics(b["judge_strong_pred"].values, b["judge_small_pred"].values)["kappa"],
                    diff_strong_minus_small=L.paired_diff(b, b["judge_strong_pred"].values, b["judge_small_pred"].values, n_boot=NB))
            for d, b in both.groupby("domain")}
    # FaithBench sensitivity to the label definition (best annotator label instead of worst)
    fb = split(df, "faithbench", "test").copy()
    fbc = split(df, "faithbench", "calib").copy()
    fb["label"], fbc["label"] = fb["label_best"], fbc["label_best"]
    out["faithbench_best_label"] = {s: L.eval_scorer(fb, fbc, s, boot=False) for s in KEY if s in df}
    for judge in ["small", "strong"]:
        pc = f"judge_{judge}_pred"
        if pc in fb and fb[pc].notna().any():
            out["faithbench_best_label"][f"judge_{judge}"] = L.eval_pred(fb[fb[pc].notna()], pc, boot=False)
    L.dump("main_performance.json", out)
    return out


# ============================================== 2. error types on natural RAGTruth test data
def error_types(df):
    te = pd.concat([split(df, d, "test") for d in L.RAG_DOMAINS]).copy()
    thr = {(d, s): thr_max_bacc(split(df, d, "calib")["label"].values, split(df, d, "calib")[s].values)
           for d in L.RAG_DOMAINS for s in KEY}
    thr_a = {(d, s): thr_far(split(df, d, "calib")["label"].values, split(df, d, "calib")[s].values, ALPHA)
             for d in L.RAG_DOMAINS for s in KEY}
    out = {}
    types = ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]
    buckets = {t: te[te["error_types"].str.split("|").apply(lambda x: t in x)] for t in types}
    buckets["Conflict only"] = te[te["error_types"].apply(lambda x: bool(x) and all("Conflict" in t for t in x.split("|")))]
    buckets["Baseless only"] = te[te["error_types"].apply(lambda x: bool(x) and all("Baseless" in t for t in x.split("|")))]
    buckets["Subtle only"] = te[te["error_types"].apply(lambda x: bool(x) and all("Subtle" in t for t in x.split("|")))]
    for name, b in buckets.items():
        row = dict(n=int(len(b)))
        for s in KEY:
            rej = np.array([v < thr[(d, s)] for v, d in zip(b[s].values, b["domain"].values)])
            acc_a = np.array([v >= thr_a[(d, s)] for v, d in zip(b[s].values, b["domain"].values)])
            row[s] = dict(recall=float(rej.mean()) if len(b) else None,
                          far_at_alpha_threshold=float(acc_a.mean()) if len(b) else None)
            if len(b) > 5:
                _, lo, hi, _ = cluster_bootstrap(b["group"].values, lambda i: rej[i].mean(), NB)
                row[s]["recall_ci"] = [float(lo), float(hi)]
        for judge in ["small", "strong"]:
            pc = f"judge_{judge}_pred"
            if pc in b and b[pc].notna().any():
                jb = b[b[pc].notna()]
                row[f"judge_{judge}"] = dict(recall=float(jb[pc].mean()), n=int(len(jb)))
        out[name] = row
    L.dump("error_types.json", out)
    return out


# ======================================================= 3. counterfactual pairs (RQ2, H1, H2)
def counterfactual(df):
    cf = df[df["dataset"] == "counterfactual"].copy()
    orig = cf[cf["cf_type"] == "original"].set_index("cf_base")
    para = cf[cf["cf_type"] == "paraphrase"].set_index("cf_base")
    edits = cf[~cf["cf_type"].isin(["original", "paraphrase"])]
    meta = {r["id"]: r["meta"] for r in read_jsonl(PROC / "counterfactuals.jsonl")}
    scorers = [s for s in ["lex_f1", "lex_tfidf", "lex_tok_prec", "minilm_doc", "minilm_mm", "minilm_min",
                           "bge_small_mm", "bge_small_min", "bertscore", "nli_mean_ec", "nli_min_e", "risk_cos",
                           "risk_cos_checks", "risk_full"] if s in cf]
    thr = {}
    for d in L.RAG_DOMAINS:
        ca = split(df, d, "calib")
        for s in scorers:
            c = ca.dropna(subset=[s])
            thr[(d, s, "bacc")] = thr_max_bacc(c["label"].values, c[s].values)
            thr[(d, s, "far")] = thr_far(c["label"].values, c[s].values, ALPHA)
            thr[(d, s, "sd")] = float(c[s].std())
    types = sorted(edits["cf_type"].unique())
    out = dict(n_bases=int(len(orig)), n_pairs=int(len(edits)), n_paraphrases=int(len(para)), by_scorer={}, judges={})

    def table(e, o_base, score_e, score_o, rej_e, acc_o, groups):
        """Pair-level statistics with cluster-bootstrap CIs."""
        res = {}
        for t in ["ALL"] + types:
            m = np.ones(len(e), bool) if t == "ALL" else (e["cf_type"].values == t)
            if m.sum() == 0:
                continue
            g = groups[m]
            def stat(i, m=m):
                re_, ao = rej_e[m][i], acc_o[m][i]
                return np.array([re_.mean(), ao.mean(), (re_ & ao).mean(), (score_e[m][i] < score_o[m][i]).mean()])
            p, lo, hi, _ = cluster_bootstrap(g, stat, NB)
            res[t] = dict(n=int(m.sum()), flip_rejection=float(p[0]), flip_rejection_ci=[float(lo[0]), float(hi[0])],
                          original_acceptance=float(p[1]), paired_detection=float(p[2]),
                          paired_detection_ci=[float(lo[2]), float(hi[2])],
                          score_drop_rate=float(p[3]), score_drop_ci=[float(lo[3]), float(hi[3])])
        return res

    for s in scorers:
        e = edits.dropna(subset=[s])
        o = orig.loc[e["cf_base"].values]
        se, so = e[s].values, o[s].values
        res = {}
        for op in ["bacc", "far"]:
            t_e = np.array([thr[(d, s, op)] for d in e["domain"].values])
            res[op] = table(e, o, se, so, se < t_e, so >= t_e, e["group"].values)
        sd = np.array([thr[(d, s, "sd")] for d in e["domain"].values])
        res["delta_sd_by_type"] = {t: float(((so - se) / sd)[e["cf_type"].values == t].mean()) for t in types}
        # H2: homogeneity of rejection across types (chi-square; pairs from one base are not independent)
        t_e = np.array([thr[(d, s, "bacc")] for d in e["domain"].values])
        ct = pd.crosstab(e["cf_type"].values, se < t_e)
        if ct.shape[1] == 2:
            chi, p, dof, _ = chi2_contingency(ct.values)
            res["homogeneity"] = dict(chi2=float(chi), dof=int(dof), p=float(p),
                                      cramers_v=float(np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1)))))
            # preregistered test: permutation of outcomes among the pairs of one base (respects the clustering)
            rej = (se < t_e).astype(float)
            tcode = pd.factorize(e["cf_type"].values)[0]
            bcode = pd.factorize(e["cf_base"].values)[0]
            nt = np.bincount(tcode)
            pbar = rej.mean()

            def hstat(r):
                return float((nt * (np.bincount(tcode, weights=r) / nt - pbar) ** 2).sum() / (pbar * (1 - pbar)))
            obs = hstat(rej)
            prng = np.random.default_rng(SEED + 7)
            srt = np.argsort(bcode, kind="stable")
            ge = 0
            for _ in range(NB):
                order = np.lexsort((prng.random(len(rej)), bcode))
                rp = np.empty_like(rej)
                rp[srt] = rej[order]
                ge += hstat(rp) >= obs - 1e-12
            res["homogeneity"]["p_perm_within_base"] = float((ge + 1) / (NB + 1))
            res["homogeneity"]["rejection_range"] = [float(min(np.bincount(tcode, weights=rej) / nt)), float(max(np.bincount(tcode, weights=rej) / nt))]
        # paraphrase invariance and matched-locus inversion
        if len(para):
            pp = para.dropna(subset=[s])
            po = orig.loc[pp.index]
            t_p = np.array([thr[(d, s, "bacc")] for d in pp["domain"].values])
            acc_p, acc_o = pp[s].values >= t_p, po[s].values >= t_p
            _, lo, hi, _ = cluster_bootstrap(pp["group"].values, lambda i: acc_p[i].mean(), NB)
            res["paraphrase"] = dict(n=int(len(pp)), acceptance=float(acc_p.mean()), acceptance_ci=[float(lo), float(hi)],
                                     original_acceptance=float(acc_o.mean()),
                                     false_rejection_given_original_accepted=float((~acc_p[acc_o]).mean()) if acc_o.any() else None,
                                     delta_sd=float(((po[s].values - pp[s].values) / np.array([thr[(d, s, "sd")] for d in pp["domain"].values])).mean()))
            m = e[e["cf_matched"] & e["cf_base"].isin(pp.index) & (e["cf_type"] != "addition")]
            if len(m):
                sp = pp.loc[m["cf_base"].values, s].values
                sf = m[s].values
                inv = sf >= sp
                _, lo, hi, _ = cluster_bootstrap(m["group"].values, lambda i: inv[i].mean(), NB)
                res["inversion"] = dict(n=int(len(m)), rate=float(inv.mean()), ci=[float(lo), float(hi)],
                                        by_type={t: dict(n=int((m["cf_type"] == t).sum()), rate=float(inv[(m["cf_type"] == t).values].mean()))
                                                 for t in types if (m["cf_type"] == t).any()})
        # dilution: score drop (in calibration SD) by answer length
        nsent = np.array([meta[i]["n_sentences"] for i in e.index])
        drop = (so - se) / sd
        bins = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 99)]
        t_b = np.array([thr[(d, s, "bacc")] for d in e["domain"].values])
        res["dilution"] = {f"{a}-{b}": dict(n=int(((nsent >= a) & (nsent <= b)).sum()),
                                            mean_drop_sd=float(drop[(nsent >= a) & (nsent <= b)].mean()),
                                            flip_rejection=float((se < t_b)[(nsent >= a) & (nsent <= b)].mean()),
                                            paired_detection=float(((se < t_b) & (so >= t_b))[(nsent >= a) & (nsent <= b)].mean()))
                           for a, b in bins if ((nsent >= a) & (nsent <= b)).any()}
        out["by_scorer"][s] = res

    # H1 contrast on all pairs: paired detection of each key scorer minus primary cosine
    out["vs_cosine_all_pairs"] = {}
    for s in [x for x in ["bge_small_mm", "bertscore", "nli_mean_ec", "risk_cos_checks", "risk_full"] if x in scorers]:
        e = edits.dropna(subset=[s, "minilm_mm"])
        o = orig.loc[e["cf_base"].values]
        det = {}
        for x in [s, "minilm_mm"]:
            t_e = np.array([thr[(d, x, "bacc")] for d in e["domain"].values])
            det[x] = (e[x].values < t_e) & (o[x].values >= t_e)
        p, lo, hi, samp = cluster_bootstrap(e["group"].values, lambda i: det[s][i].mean() - det["minilm_mm"][i].mean(), NB)
        out["vs_cosine_all_pairs"][s] = dict(diff=float(p), ci=[float(lo), float(hi)],
                                             p_boot=float(min(1.0, 2 * min((samp <= 0).mean(), (samp >= 0).mean()))))

    # judges (verdict-based) with cheap scorers on the same pairs
    for judge in ["small", "strong"]:
        pc = f"judge_{judge}_pred"
        if pc not in cf or cf[pc].notna().sum() == 0:
            continue
        e = edits[edits[pc].notna() & edits["cf_base"].isin(orig[orig[pc].notna()].index)]
        o = orig.loc[e["cf_base"].values]
        rej_e, acc_o = e[pc].values == 1, o[pc].values == 0
        pe, po_ = e[f"judge_{judge}_psup"].values.astype(float), o[f"judge_{judge}_psup"].values.astype(float)
        res = dict(verdict=table(e, o, pe, po_, rej_e, acc_o, e["group"].values))
        res["same_pairs"] = {}
        for s in [x for x in ["minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_full"] if x in scorers]:
            t_e = np.array([thr[(d, s, "bacc")] for d in e["domain"].values])
            det_s = (e[s].values < t_e) & (o[s].values >= t_e)
            det_j = rej_e & acc_o
            p, lo, hi, samp = cluster_bootstrap(e["group"].values, lambda i: det_j[i].mean() - det_s[i].mean(), NB)
            res["same_pairs"][s] = dict(paired_detection=float(det_s.mean()), flip_rejection=float((e[s].values < t_e).mean()),
                                        judge_minus_scorer=float(p), ci=[float(lo), float(hi)],
                                        p_boot=float(min(1.0, 2 * min((samp <= 0).mean(), (samp >= 0).mean()))))
        pj = para[para[pc].notna()]
        if len(pj):
            oj = orig.loc[pj.index]
            res["paraphrase"] = dict(n=int(len(pj)), acceptance=float((pj[pc] == 0).mean()),
                                     original_acceptance=float((oj[pc] == 0).mean()))
            m = e[e["cf_matched"] & e["cf_base"].isin(pj.index) & (e["cf_type"] != "addition")]
            if len(m):
                vp = pj.loc[m["cf_base"].values, pc].values
                res["inversion_verdict"] = dict(n=int(len(m)), flip_accepted_paraphrase_rejected=float(((m[pc].values == 0) & (vp == 1)).mean()),
                                                correct_order=float(((m[pc].values == 1) & (vp == 0)).mean()),
                                                psup_inversion=float((m[f"judge_{judge}_psup"].values >= pj.loc[m["cf_base"].values, f"judge_{judge}_psup"].values).mean()))
        res["n_originals"] = int(orig[pc].notna().sum())
        res["original_acceptance_all"] = float((orig.loc[orig[pc].notna(), pc] == 0).mean())
        out["judges"][judge] = res
    L.dump("counterfactual.json", out)
    return out


# ========================================================== 4. threshold transfer (RQ3, H3)
def transfer(df):
    out = {"matrix": {}, "h3": {}, "generator_shift": {}, "length_shift": {}, "router_trained_per_domain": {}}
    scorers = [s for s in ["minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec", "risk_cos", "risk_full"] if s in df]
    for s in scorers:
        mat = {}
        for src in L.FAITH_DOMAINS:
            ca = split(df, src, "calib").dropna(subset=[s])
            tb = thr_max_bacc(ca["label"].values, ca[s].values)
            tf = {a: thr_far(ca["label"].values, ca[s].values, a) for a in ALPHAS}
            for tgt in L.FAITH_DOMAINS:
                te = split(df, tgt, "test").dropna(subset=[s])
                y, v, g = te["label"].values, te[s].values, te["group"].values
                cell = dict(bacc=binary_metrics(y, (v < tb).astype(int))["balanced_accuracy"])
                for a in ALPHAS:
                    acc = v >= tf[a]
                    def stat(i, acc=acc):
                        yy = y[i]
                        return np.array([acc[i][yy == 1].mean() if (yy == 1).any() else np.nan,
                                         (~acc[i])[yy == 0].mean() if (yy == 0).any() else np.nan])
                    p, lo, hi, samp = cluster_bootstrap(g, stat, NB)
                    cell[f"p_far_le_alpha@{a}"] = float(np.nanmean(np.asarray(samp)[:, 0] <= a))   # one-sided bootstrap p
                    cell[f"far@{a}"] = float(p[0])
                    cell[f"far_ci@{a}"] = [float(lo[0]), float(hi[0])]
                    cell[f"frr@{a}"] = float(p[1])
                    cell[f"accept_rate@{a}"] = float(acc.mean())
                mat[f"{src}->{tgt}"] = cell
        out["matrix"][s] = mat
        off = [k for k in mat if k.split("->")[0] != k.split("->")[1]]
        diag = [k for k in mat if k.split("->")[0] == k.split("->")[1]]
        out["h3"][s] = {
            str(a): dict(
                offdiag_cells=len(off),
                offdiag_far_exceeds_alpha_ci=int(sum(mat[k][f"far_ci@{a}"][0] > a for k in off)),
                offdiag_far_max=float(max(mat[k][f"far@{a}"] for k in off)),
                p_bonferroni=float(min(1.0, len(off) * max(1.0 / NB, min(mat[k][f"p_far_le_alpha@{a}"] for k in off)))),
                offdiag_exceed_cells=[k for k in off if mat[k][f"far_ci@{a}"][0] > a],
                offdiag_far_median=float(np.median([mat[k][f"far@{a}"] for k in off])),
                diag_far_exceeds_alpha_ci=int(sum(mat[k][f"far_ci@{a}"][0] > a for k in diag)),
                diag_far_max=float(max(mat[k][f"far@{a}"] for k in diag)),
                mean_bacc_drop_offdiag=float(np.mean([mat[f"{k.split('->')[1]}->{k.split('->')[1]}"]["bacc"] - mat[k]["bacc"] for k in off])),
            ) for a in ALPHAS}
    # risk estimator trained on a single RAGTruth task, tested on the others
    for src in L.RAG_DOMAINS:
        tmp = df.copy()
        L.fit_routers(tmp, train_domains=[src], variants={"risk_full": L.ROUTER_VARIANTS["risk_full"]}, tag="_src")
        ca = split(tmp, src, "calib")
        tf = thr_far(ca["label"].values, ca["risk_full_src"].values, ALPHA)
        tb = thr_max_bacc(ca["label"].values, ca["risk_full_src"].values)
        for tgt in L.FAITH_DOMAINS:
            te = split(tmp, tgt, "test")
            y, v = te["label"].values, te["risk_full_src"].values
            out["router_trained_per_domain"][f"{src}->{tgt}"] = dict(
                auroc=auroc(y, -v), bacc=binary_metrics(y, (v < tb).astype(int))["balanced_accuracy"],
                far=float((v >= tf)[y == 1].mean()), frr=float((v < tf)[y == 0].mean()))
    # generator-family and length shifts inside RAGTruth (pooled tasks)
    ca = pd.concat([split(df, d, "calib") for d in L.RAG_DOMAINS])
    te = pd.concat([split(df, d, "test") for d in L.RAG_DOMAINS])
    fam = lambda g: "openai" if g.startswith("gpt") else "open_weight"
    med = ca["len_ans_tok"].median()
    shifts = {"generator_shift": (ca["generator"].map(fam), te["generator"].map(fam)),
              "length_shift": (np.where(ca["len_ans_tok"] <= med, "short", "long"), np.where(te["len_ans_tok"] <= med, "short", "long"))}
    for name, (gc, gt) in shifts.items():
        gc, gt = np.asarray(gc), np.asarray(gt)
        for s in scorers:
            for src in sorted(set(gc)):
                c = ca[gc == src].dropna(subset=[s])
                tf = thr_far(c["label"].values, c[s].values, ALPHA)
                tb = thr_max_bacc(c["label"].values, c[s].values)
                for tgt in sorted(set(gt)):
                    t = te[gt == tgt].dropna(subset=[s])
                    y, v, g = t["label"].values, t[s].values, t["group"].values
                    acc = v >= tf
                    p, lo, hi, _ = cluster_bootstrap(g, lambda i: acc[i][y[i] == 1].mean(), NB)
                    out[name].setdefault(s, {})[f"{src}->{tgt}"] = dict(
                        n=int(len(t)), far=float(p), far_ci=[float(lo), float(hi)], frr=float((~acc)[y == 0].mean()),
                        bacc=binary_metrics(y, (v < tb).astype(int))["balanced_accuracy"], auroc=auroc(y, -v),
                        base_rate=float(y.mean()))
    L.dump("transfer.json", out)
    return out


# ============================================== 5. calibration and ablation of the risk estimator
def router_analysis(df, routers):
    out = {"calibration": {}, "ablation": {}, "coefficients": {k: r.coefficients() for k, r in routers.items()}}
    for dom in L.FAITH_DOMAINS:
        te = split(df, dom, "test")
        for name in L.ROUTER_VARIANTS:
            e, b = ece_brier(te["label"].values, te[name + "_prob"].values)
            out["calibration"].setdefault(dom, {})[name] = dict(ece=e, brier=b, base_rate=float(te["label"].mean()))
    # leave-one-feature-group-out ablation (trained on RAGTruth train, evaluated on each test domain)
    variants = {"full": list(L.GROUPS)}
    variants.update({f"minus_{g}": [x for x in L.GROUPS if x != g] for g in L.GROUPS})
    variants.update({f"only_{g}": [g] for g in L.GROUPS})
    tmp = df.copy()
    L.fit_routers(tmp, variants=variants, tag="_abl")
    pooled = pd.concat([split(tmp, d, "test") for d in L.RAG_DOMAINS])
    cf = tmp[(tmp["dataset"] == "counterfactual") & ~tmp["cf_type"].isin(["original", "paraphrase"])]
    cfo = tmp[(tmp["dataset"] == "counterfactual") & (tmp["cf_type"] == "original")].set_index("cf_base")
    for v in variants:
        col = v + "_abl"
        y, s, g = pooled["label"].values, pooled[col].values, pooled["group"].values
        p, lo, hi, _ = cluster_bootstrap(g, lambda i: auroc(y[i], -s[i]), NB)
        row = dict(ragtruth_auroc=float(p), ragtruth_auroc_ci=[float(lo), float(hi)])
        for d in ["faithbench", "halueval_qa"]:
            t = split(tmp, d, "test")
            row[f"{d}_auroc"] = auroc(t["label"].values, -t[col].values)
        row["cf_score_drop_rate"] = float((cf[col].values < cfo.loc[cf["cf_base"].values, col].values).mean())
        both = pd.concat([cf[[col]].assign(y=1), cfo[[col]].assign(y=0)])
        row["cf_auroc_edit_vs_original"] = auroc(both["y"].values, -both[col].values)
        out["ablation"][v] = row
    full = out["ablation"]["full"]["ragtruth_auroc"]
    yv, gv = pooled["label"].values, pooled["group"].values
    for v in variants:
        if v.startswith("minus_"):
            a, b = pooled["full_abl"].values, pooled[v + "_abl"].values
            p, lo, hi, _ = cluster_bootstrap(gv, lambda i: auroc(yv[i], -a[i]) - auroc(yv[i], -b[i]), NB)
            out["ablation"][v]["auroc_loss_vs_full"] = float(p)
            out["ablation"][v]["auroc_loss_ci"] = [float(lo), float(hi)]
    # non-linear sensitivity check: gradient boosting on the same features
    from sklearn.ensemble import HistGradientBoostingClassifier

    feats = L.feature_list(L.GROUPS)
    tr = df[(df["split"] == "train") & df["domain"].isin(L.RAG_DOMAINS)]
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, random_state=SEED).fit(tr[feats].values, tr["label"].values)
    out["gbm_check"] = {d: dict(gbm_auroc=auroc(split(df, d, "test")["label"].values, gb.predict_proba(split(df, d, "test")[feats].values)[:, 1]),
                                lr_auroc=auroc(split(df, d, "test")["label"].values, -split(df, d, "test")["risk_full"].values))
                        for d in L.FAITH_DOMAINS}
    L.dump("router_analysis.json", out)
    return out


# ================================================= 6. selective cascade (RQ4, H4, RQ5)
def cascade_decisions(score, t_acc, t_rej):
    """0 = accept cheaply, 1 = reject cheaply, -1 = escalate."""
    t_rej = min(t_rej, t_acc)
    return np.where(score >= t_acc, 0, np.where(score < t_rej, 1, -1))


def cascade(df):
    out = {"cheap_stage_full_test": {}, "system_judged_subset": {}, "h4": {}, "curves": {}, "ood_faithbench": {}}
    gates = ["minilm_mm", "bge_small_mm", "risk_cos", "risk_cos_checks", "risk_full", "nli_mean_ec"]
    rng = np.random.default_rng(SEED + 99)

    def thresholds(calibs, s, alpha):
        return {d: (thr_far(c["label"].values, c[s].values, alpha), thr_frr(c["label"].values, c[s].values, BETA)) for d, c in calibs.items()}

    def decide(te, thr, s):
        dec = np.zeros(len(te), dtype=int)
        for d, (ta, tr) in thr.items():
            m = (te["domain"] == d).values
            dec[m] = cascade_decisions(te.loc[m, s].values, ta, tr)
        return dec

    # ---- cheap stage on the full RAGTruth test set (no judge needed)
    calibs = {d: split(df, d, "calib") for d in L.RAG_DOMAINS}
    te = pd.concat([split(df, d, "test") for d in L.RAG_DOMAINS])
    y, g = te["label"].values, te["group"].values
    decs = {}
    for alpha in ALPHAS:
        for s in gates:
            dec = decide(te, thresholds(calibs, s, alpha), s)
            decs[(alpha, s)] = dec
            def stat(i, dec=dec):
                d_, y_ = dec[i], y[i]
                return np.array([(d_ != -1).mean(), (d_ == 0)[y_ == 1].mean(), (d_ == 1)[y_ == 0].mean(), (d_ == 0).mean()])
            p, lo, hi, _ = cluster_bootstrap(g, stat, NB)
            out["cheap_stage_full_test"].setdefault(str(alpha), {})[s] = dict(
                n=int(len(te)), coverage=float(p[0]), coverage_ci=[float(lo[0]), float(hi[0])],
                cheap_far=float(p[1]), cheap_far_ci=[float(lo[1]), float(hi[1])],
                cheap_frr=float(p[2]), cheap_frr_ci=[float(lo[2]), float(hi[2])], cheap_accept_share=float(p[3]),
                by_domain={d: dict(coverage=float((dec[(te["domain"] == d).values] != -1).mean()),
                                   cheap_far=float((dec == 0)[(te["domain"] == d).values & (y == 1)].mean()))
                           for d in L.RAG_DOMAINS})
        for s in ["risk_cos_checks", "risk_full", "nli_mean_ec", "bge_small_mm"]:
            a_, b_ = decs[(alpha, s)], decs[(alpha, "minilm_mm")]
            p, lo, hi, samp = cluster_bootstrap(g, lambda i: (a_[i] != -1).mean() - (b_[i] != -1).mean(), NB)
            out["h4"].setdefault(str(alpha), {})[f"{s}_minus_minilm_mm_coverage"] = dict(
                diff=float(p), ci=[float(lo), float(hi)], p_boot=float(min(1.0, 2 * min((samp <= 0).mean(), (samp >= 0).mean()))))

    # ---- full system on the judged subset
    for judge in ["small", "strong"]:
        pc = f"judge_{judge}_pred"
        if pc not in df:
            continue
        sub = te[te[pc].notna()]
        if not len(sub):
            continue
        ys, gs, jp = sub["label"].values, sub["group"].values, sub[pc].values.astype(int)
        jt = sub[f"judge_{judge}_wall_s"].values
        res = {"n": int(len(sub)), "always_judge": dict(L.eval_pred(sub, pc, n_boot=NB), escalation=1.0)}
        for s in [x for x in ["lex_tok_prec", "lex_tfidf", "minilm_mm", "bge_small_mm", "bertscore", "nli_mean_ec",
                              "risk_cos", "risk_cos_checks", "risk_full"] if x in sub and sub[x].notna().all()]:
            pred = np.zeros(len(sub), dtype=int)
            for d, c in calibs.items():
                m = (sub["domain"] == d).values
                pred[m] = (sub.loc[m, s].values < thr_max_bacc(c["label"].values, c[s].values)).astype(int)
            _, lo, hi, _ = cluster_bootstrap(gs, lambda i, pred=pred: binary_metrics(ys[i], pred[i])["balanced_accuracy"], NB)
            res[f"always_cheap_{s}"] = dict(binary_metrics(ys, pred), escalation=0.0, bacc_ci=[float(lo), float(hi)],
                                            auroc=auroc(ys, -sub[s].values),
                                            vs_judge=L.paired_diff(sub, pred, jp, n_boot=NB))
        for alpha in ALPHAS:
            for s in gates:
                dec = decide(sub, thresholds(calibs, s, alpha), s)
                final = np.where(dec == -1, jp, dec)
                def stat(i, final=final, dec=dec):
                    bm = binary_metrics(ys[i], final[i])
                    return np.array([bm["balanced_accuracy"], bm["far"], bm["frr"], (dec[i] == -1).mean(),
                                     bm["balanced_accuracy"] - binary_metrics(ys[i], jp[i])["balanced_accuracy"]])
                p, lo, hi, _ = cluster_bootstrap(gs, stat, NB)
                r = dict(balanced_accuracy=float(p[0]), bacc_ci=[float(lo[0]), float(hi[0])], far=float(p[1]),
                         far_ci=[float(lo[1]), float(hi[1])], frr=float(p[2]), escalation=float(p[3]),
                         escalation_ci=[float(lo[3]), float(hi[3])], bacc_minus_always_judge=float(p[4]),
                         bacc_minus_always_judge_ci=[float(lo[4]), float(hi[4])],
                         accuracy=binary_metrics(ys, final)["accuracy"], macro_f1=binary_metrics(ys, final)["macro_f1"],
                         judge_seconds_per_1000=float(jt[dec == -1].sum() / len(sub) * 1000))
                # random-escalation control: same number of judge calls, cheap max-bacc decision elsewhere
                k = int((dec == -1).sum())
                cheap = np.zeros(len(sub), dtype=int)
                for d, c in calibs.items():
                    m = (sub["domain"] == d).values
                    cheap[m] = (sub.loc[m, s].values < thr_max_bacc(c["label"].values, c[s].values)).astype(int)
                rb, rf = [], []
                for _ in range(300):
                    pick = rng.permutation(len(sub))[:k]
                    f2 = cheap.copy()
                    f2[pick] = jp[pick]
                    bm = binary_metrics(ys, f2)
                    rb.append(bm["balanced_accuracy"])
                    rf.append(bm["far"])
                r["random_escalation"] = dict(balanced_accuracy=float(np.mean(rb)), bacc_sd=float(np.std(rb)), far=float(np.mean(rf)))
                # uncertainty-gated control at the same budget: escalate the k examples closest to the max-bacc threshold
                res.setdefault(str(alpha), {})[s] = r
        out["system_judged_subset"][judge] = res

        # ---- curves: system balanced accuracy vs. share escalated (margin-gated vs random)
        curves = {}
        fr = np.linspace(0, 1, 21)
        for s in ["minilm_mm", "risk_full", "nli_mean_ec"]:
            margin = np.zeros(len(sub))
            cheap = np.zeros(len(sub), dtype=int)
            for d, c in calibs.items():
                m = (sub["domain"] == d).values
                t = thr_max_bacc(c["label"].values, c[s].values)
                margin[m] = np.abs(sub.loc[m, s].values - t) / (c[s].std() + 1e-12)
                cheap[m] = (sub.loc[m, s].values < t).astype(int)
            order = np.argsort(margin)
            gated, rand = [], []
            for f in fr:
                k = int(round(f * len(sub)))
                f1_ = cheap.copy()
                f1_[order[:k]] = jp[order[:k]]
                gated.append(binary_metrics(ys, f1_)["balanced_accuracy"])
                acc = []
                for _ in range(100):
                    pick = rng.permutation(len(sub))[:k]
                    f2 = cheap.copy()
                    f2[pick] = jp[pick]
                    acc.append(binary_metrics(ys, f2)["balanced_accuracy"])
                rand.append(float(np.mean(acc)))
            # selective risk of the cheap method alone vs. coverage (most confident first)
            conf = np.argsort(-margin)
            cov = np.linspace(0.05, 1, 20)
            sel = [float((cheap[conf[: max(1, int(c_ * len(sub)))]] != ys[conf[: max(1, int(c_ * len(sub)))]]).mean()) for c_ in cov]
            curves[s] = dict(fraction=fr.tolist(), gated=gated, random=rand, coverage=cov.tolist(), selective_error=sel,
                             aurc=float(np.mean(sel)))
        out["curves"][judge] = curves

    # ---- out-of-domain cascade: RAGTruth-summary thresholds applied to FaithBench
    fb = split(df, "faithbench", "test")
    for src, cal in [("in_domain", {"faithbench": split(df, "faithbench", "calib")}),
                     ("from_ragtruth_summary", {"faithbench": split(df, "ragtruth_summary", "calib")})]:
        for s in ["minilm_mm", "risk_full"]:
            dec = decide(fb, thresholds(cal, s, ALPHA), s)
            yy = fb["label"].values
            r = dict(coverage=float((dec != -1).mean()), cheap_far=float((dec == 0)[yy == 1].mean()),
                     cheap_frr=float((dec == 1)[yy == 0].mean()))
            if "judge_small_pred" in fb and fb["judge_small_pred"].notna().all():
                final = np.where(dec == -1, fb["judge_small_pred"].values.astype(int), dec)
                r.update(system_bacc=binary_metrics(yy, final)["balanced_accuracy"], system_far=binary_metrics(yy, final)["far"])
            out["ood_faithbench"].setdefault(src, {})[s] = r

    # ---- reference-based (EVOUNA): same construction
    ev_c, ev_t = {"evouna_nq": split(df, "evouna_nq", "calib")}, split(df, "evouna_nq", "test")
    out["evouna"] = {}
    for s in ["minilm_doc", "lex_f1", "risk_full"]:
        dec = decide(ev_t, thresholds(ev_c, s, ALPHA), s)
        yy = ev_t["label"].values
        r = dict(coverage=float((dec != -1).mean()), cheap_far=float((dec == 0)[yy == 1].mean()), cheap_frr=float((dec == 1)[yy == 0].mean()))
        if "judge_small_pred" in ev_t:
            m = ev_t["judge_small_pred"].notna().values
            if m.any():
                final = np.where(dec[m] == -1, ev_t.loc[m, "judge_small_pred"].values.astype(int), dec[m])
                r.update(n_judged=int(m.sum()), system_bacc=binary_metrics(yy[m], final)["balanced_accuracy"],
                         system_far=binary_metrics(yy[m], final)["far"], escalation=float((dec[m] == -1).mean()),
                         always_judge_bacc=binary_metrics(yy[m], ev_t.loc[m, "judge_small_pred"].values.astype(int))["balanced_accuracy"])
        out["evouna"][s] = r
    L.dump("cascade.json", out)
    return out


# ======================================================================= 7. prompt sensitivity
def prompt_sensitivity(df):
    a, b = L.load_judge("small", "natural", "v1"), L.load_judge("small", "sensitivity", "v2")
    if not len(b):
        return {}
    ids = [i for i in b.index if i in a.index]
    y = df.loc[ids, "label"].values
    pa, pb = a.loc[ids, "pred"].values, b.loc[ids, "pred"].values
    out = dict(n=len(ids), v1=binary_metrics(y, pa), v2=binary_metrics(y, pb),
               agreement=float((pa == pb).mean()), kappa_v1_v2=binary_metrics(pa, pb)["kappa"],
               verdicts_v1=a.loc[ids, "verdict"].value_counts().to_dict(), verdicts_v2=b.loc[ids, "verdict"].value_counts().to_dict())
    L.dump("prompt_sensitivity.json", out)
    return out


def hypotheses(main, cfres, tr, ca):
    """Summary of preregistered tests with Holm correction over the primary family."""
    p = {}
    h = {}
    j = cfres["judges"].get("small", {})
    if j:
        for s in ["minilm_mm"]:
            p["H1_judge_vs_cosine"] = j["same_pairs"][s]["p_boot"]
        cos, nli = cfres["by_scorer"]["minilm_mm"]["bacc"]["ALL"], cfres["by_scorer"].get("nli_mean_ec", {}).get("bacc", {}).get("ALL")
        h["H1"] = dict(cosine_paired_detection=cos["paired_detection"], cosine_ci=cos["paired_detection_ci"],
                       nli_paired_detection=nli and nli["paired_detection"], nli_ci=nli and nli["paired_detection_ci"],
                       judge_small_paired_detection=j["verdict"]["ALL"]["paired_detection"],
                       judge_minus_cosine=j["same_pairs"]["minilm_mm"],
                       cosine_paraphrase_acceptance=cfres["by_scorer"]["minilm_mm"].get("paraphrase"),
                       cosine_inversion=cfres["by_scorer"]["minilm_mm"].get("inversion"))
    h["H2"] = {s: cfres["by_scorer"][s].get("homogeneity") for s in KEY if s in cfres["by_scorer"]}
    h["H3"] = tr["h3"]["minilm_mm"][str(ALPHA)]
    p["H3_transfer"] = h["H3"]["p_bonferroni"]
    v = cfres.get("vs_cosine_all_pairs", {}).get("nli_mean_ec")
    if v:
        p["H1_nli_vs_cosine"] = v["p_boot"]
        h["H1_nli_minus_cosine_all_pairs"] = v
    h["H4"] = dict(coverage_diff=ca["h4"][str(ALPHA)]["risk_full_minus_minilm_mm_coverage"],
                   system=ca["system_judged_subset"].get("small", {}).get(str(ALPHA), {}).get("risk_full"))
    p["H4_coverage"] = ca["h4"][str(ALPHA)]["risk_full_minus_minilm_mm_coverage"]["p_boot"]
    p = {k: max(v, 1.0 / NB) for k, v in p.items()}                 # a bootstrap p-value cannot be below 1/n_boot
    h["holm_adjusted_p"] = L.holm(p)
    h["raw_p"] = p
    h["note"] = ("Bootstrap p-values are bounded below by 1/n_boot; H3 is assessed by confidence intervals of "
                 "achieved FAR per transfer cell rather than by a single p-value.")
    L.dump("hypotheses.json", h)


def main():
    global NB
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    NB = 200 if a.fast else CFG["stats"]["bootstrap"]
    df = L.attach_judges(L.load_scores())
    L.dump("analysis_missing_columns.json", dict(missing=L.MISSING, n_boot=NB))
    if L.MISSING:
        print("WARNING: debugging run, missing score columns:", L.MISSING, flush=True)
    routers = L.fit_routers(df)
    L.fit_ref_router(df)
    df.to_parquet(RESULTS / "scores_all.parquet")
    m = main_performance(df); print("main done", flush=True)
    error_types(df); print("error types done", flush=True)
    c = counterfactual(df); print("cf done", flush=True)
    t = transfer(df); print("transfer done", flush=True)
    router_analysis(df, routers); print("router done", flush=True)
    k = cascade(df); print("cascade done", flush=True)
    prompt_sensitivity(df)
    hypotheses(m, c, t, k)
    print("all analyses written to", RESULTS)


if __name__ == "__main__":
    main()
