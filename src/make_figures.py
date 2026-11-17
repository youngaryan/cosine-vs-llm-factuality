"""Figures for the manuscript, drawn only from results/*.json and results/scores_all.parquet.

  python src/make_figures.py      # writes figures/*.pdf and figures/*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CFG, FIGS, RESULTS

FIGS.mkdir(exist_ok=True)
# categorical palette (validated for colour-vision deficiency; identity is also carried by markers and direct labels)
C = {"cos": "#2a78d6", "risk": "#eb6834", "nli": "#1baf7a", "bert": "#eda100", "jsmall": "#e87ba4", "jstrong": "#008300",
     "lex": "#6b7280", "ink": "#1f2937", "grid": "#e5e7eb"}
MK = {"cos": "o", "risk": "s", "nli": "^", "bert": "D", "jsmall": "P", "jstrong": "X", "lex": "v"}
SC = {"minilm_mm": ("cos", "MiniLM cosine"), "bge_small_mm": ("cos", "BGE-small cosine"), "risk_full": ("risk", "Risk model"),
      "nli_mean_ec": ("nli", "NLI"), "bertscore": ("bert", "BERTScore"), "lex_tok_prec": ("lex", "Token precision")}
DOM = {"ragtruth_qa": "RT-QA", "ragtruth_summary": "RT-Sum", "ragtruth_data2txt": "RT-D2T", "faithbench": "FaithBench",
       "halueval_qa": "HaluEval"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9ca3af",
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"], "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans", "lines.linewidth": 1.6, "lines.markersize": 5})


def R(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig_pareto():
    """Balanced accuracy against measured wall-clock cost per 1,000 evaluations, one panel per judged subset."""
    k, cost = R("cascade"), R("cost")
    if not k or not cost or "small" not in k["system_judged_subset"]:
        return print("skip pareto")
    me = cost["methods"]
    cheap_cost = {"lex_tok_prec": "lexical_and_checks", "minilm_mm": "minilm_cosine", "bge_small_mm": "bge_small_cosine",
                  "bertscore": "bertscore", "nli_mean_ec": "nli", "risk_full": "risk_full"}
    judges = [j for j in ["small", "strong"] if j in k["system_judged_subset"] and f"judge_{j}" in me]
    fig, axs = plt.subplots(1, len(judges), figsize=(6.6, 3.2))
    axs = np.atleast_1d(axs)
    handles = {}
    for ax, judge in zip(axs, judges):
        sj = k["system_judged_subset"][judge]
        jk = "jsmall" if judge == "small" else "jstrong"
        for s, ck in cheap_cost.items():
            r = sj.get(f"always_cheap_{s}")
            if r and ck in me:
                key, lab = SC[s]
                x, y = me[ck]["wall_s_per_1000"], r["balanced_accuracy"]
                h = ax.errorbar(x, y, yerr=[[y - r["bacc_ci"][0]], [r["bacc_ci"][1] - y]], fmt=MK[key], color=C[key], capsize=2, lw=1,
                                mfc="white" if s == "bge_small_mm" else C[key], zorder=3)
                handles.setdefault(lab, h)
        jx = me[f"judge_{judge}"]["wall_s_per_1000"]
        aj = sj["always_judge"]
        h = ax.errorbar(jx, aj["balanced_accuracy"], yerr=[[aj["balanced_accuracy"] - aj["bacc_ci"][0]], [aj["bacc_ci"][1] - aj["balanced_accuracy"]]],
                        fmt=MK[jk], color=C[jk], capsize=2, lw=1, ms=7, zorder=3)
        handles.setdefault(f"Judge ({'3B' if judge == 'small' else '14B'}), every example", h)
        for s, ck in [("minilm_mm", "minilm_cosine"), ("risk_full", "risk_full")]:
            key = SC[s][0]
            xs, ys = [], []
            for a in ["0.2", "0.1", "0.05"]:
                r = sj.get(a, {}).get(s)
                if r:
                    xs.append(me[ck]["wall_s_per_1000"] + r["escalation"] * jx)
                    ys.append(r["balanced_accuracy"])
            h, = ax.plot(xs, ys, "-", color=C[key], lw=1.2, marker=MK[key], mfc="white", zorder=2)
            handles.setdefault(SC[s][1] + r" gate + judge ($\alpha$ = .20, .10, .05)", h)
        ax.set_xscale("log")
        ax.set_xlim(0.7, jx * 2)
        name = "3B" if judge == "small" else "14B"
        ax.set_title(f"({'a' if judge == 'small' else 'b'}) {name} judge subset, $n$ = {sj['n']}", fontsize=8, loc="left")
        ax.set_xlabel("Wall-clock s per 1,000 (CPU, log)")
    axs[0].set_ylabel("Balanced accuracy")
    fig.legend(list(handles.values()), list(handles.keys()), fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, "fig_pareto")


def fig_escalation():
    k = R("cascade")
    if not k or "small" not in k.get("curves", {}):
        return print("skip escalation")
    cu = k["curves"]["small"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    handles = []
    for s in ["minilm_mm", "nli_mean_ec", "risk_full"]:
        if s not in cu:
            continue
        key, lab = SC[s]
        x = 100 * np.array(cu[s]["fraction"])
        h, = axs[0].plot(x, cu[s]["gated"], "-", color=C[key], marker=MK[key], markevery=4, label=lab)
        axs[0].plot(x, cu[s]["random"], "--", color=C[key], lw=1.0)
        axs[1].plot(100 * np.array(cu[s]["coverage"]), 100 * np.array(cu[s]["selective_error"]), "-", color=C[key], marker=MK[key],
                    markevery=4)
        handles.append(h)
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color=C["lex"], ls="-", label="escalate smallest margin first"),
                Line2D([], [], color=C["lex"], ls="--", lw=1.0, label="escalate at random (mean)")]
    axs[0].set_xlabel("Share of examples sent to the judge (%)")
    axs[0].set_ylabel("System balanced accuracy")
    axs[0].set_title("(a) Escalation: gated vs. random", fontsize=8, loc="left")
    axs[1].set_xlabel("Coverage of the cheap method (%), most confident first")
    axs[1].set_ylabel("Selective error (%)")
    axs[1].set_title("(b) Risk-coverage of cheap methods", fontsize=8, loc="left")
    fig.legend(handles=handles, fontsize=6.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig_escalation")


def fig_transfer():
    t = R("transfer")
    a = str(CFG["router"]["primary_alpha"])
    doms = list(DOM)
    panels = [s for s in ["minilm_mm", "nli_mean_ec", "risk_full"] if s in t["matrix"]]
    fig, axs = plt.subplots(1, len(panels), figsize=(6.8, 2.5), sharey=True)
    for ax, s in zip(np.atleast_1d(axs), panels):
        mat = np.array([[t["matrix"][s][f"{x}->{y}"][f"far@{a}"] for y in doms] for x in doms])
        lo = np.array([[t["matrix"][s][f"{x}->{y}"][f"far_ci@{a}"][0] for y in doms] for x in doms])
        im = ax.imshow(100 * mat, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(doms)):
            for j in range(len(doms)):
                v = 100 * mat[i, j]
                ax.text(j, i, f"{v:.0f}" + ("*" if lo[i, j] > float(a) else ""), ha="center", va="center", fontsize=7,
                        color="white" if v > 55 else C["ink"], fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(doms)), [DOM[d] for d in doms], rotation=40, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(doms)), [DOM[d] for d in doms], fontsize=6.5)
        ax.set_title(SC[s][1], fontsize=8)
        ax.grid(False)
        ax.set_xlabel("tested on")
    np.atleast_1d(axs)[0].set_ylabel("threshold calibrated on")
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="False acceptance rate (%)")
    save(fig, "fig_transfer")


def fig_cf_types():
    c = R("counterfactual")
    types = ["negation", "number", "date", "entity", "role_swap", "direction", "causal", "addition"]
    labs = ["Negation", "Number", "Date", "Entity", "Role swap", "Direction", "Causal", "Addition"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    y = np.arange(len(types))[::-1]
    series = [(s,) + SC[s] for s in ["minilm_mm", "bertscore", "nli_mean_ec", "risk_full"] if s in c["by_scorer"]]
    for ax, key, title in [(axs[0], "score_drop_rate", "(a) Flip lowers the score (%)"), (axs[1], "paired_detection", "(b) Original accepted and flip rejected (%)")]:
        for i, (s, ck, lab) in enumerate(series):
            b = c["by_scorer"][s]["bacc"]
            ax.plot([100 * b[t][key] for t in types], y + (i - 1.5) * 0.12, MK[ck], color=C[ck], label=lab, ls="none")
        if key == "paired_detection":
            for judge, ck, lab in [("small", "jsmall", "Judge (3B), subset"), ("strong", "jstrong", "Judge (14B), subset")]:
                if judge in c["judges"]:
                    v = c["judges"][judge]["verdict"]
                    ax.plot([100 * v[t][key] if t in v else np.nan for t in types], y + (0.42 if judge == "small" else -0.42) * 0.8, MK[ck], color=C[ck],
                            label=lab, ls="none", ms=6)
        ax.set_xlim(-3, 103)
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
        for yy in y[:-1]:
            ax.axhline(yy - 0.5, color=C["grid"], lw=0.6)
    axs[0].set_yticks(y, labs)
    axs[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "fig_cf_types")


def fig_locus():
    """(a) change in MiniLM cosine for paraphrases vs. factual flips of the same sentence; (b) dilution by answer length."""
    c = R("counterfactual")
    p = RESULTS / "scores_all.parquet"
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    if p.exists():
        df = pd.read_parquet(p)
        cf = df[df["dataset"] == "counterfactual"]
        o = cf[cf["cf_type"] == "original"].set_index("cf_base")["minilm_mm"]
        pa = cf[cf["cf_type"] == "paraphrase"].set_index("cf_base")["minilm_mm"]
        fl = cf[cf["cf_matched"] & cf["cf_base"].isin(pa.index) & ~cf["cf_type"].isin(["original", "paraphrase", "addition"])]
        d_para = (pa - o.loc[pa.index]).values
        d_flip = fl["minilm_mm"].values - o.loc[fl["cf_base"].values].values
        lim = np.percentile(np.abs(np.concatenate([d_para, d_flip])), 99)
        bins = np.linspace(-lim, lim, 41)
        axs[0].hist(d_para, bins=bins, density=True, color=C["nli"], alpha=0.55, label=f"Meaning-preserving paraphrase (n = {len(d_para)})")
        axs[0].hist(d_flip, bins=bins, density=True, histtype="step", color=C["risk"], lw=1.6, label=f"Single-fact flip, same sentence (n = {len(d_flip)})")
        axs[0].axvline(0, color=C["ink"], lw=0.6)
        axs[0].set_xlabel("Change in MiniLM mean-max cosine vs. original answer")
        axs[0].set_ylabel("Density")
        axs[0].legend(fontsize=6.5, loc="upper left")
        axs[0].set_title("(a) Paraphrase vs. factual flip", fontsize=8, loc="left")
    for s in ["minilm_mm", "bertscore", "nli_mean_ec", "risk_full"]:
        if s not in c["by_scorer"]:
            continue
        ck, lab = SC[s]
        dl = c["by_scorer"][s]["dilution"]
        ks = list(dl)
        axs[1].plot(range(len(ks)), [dl[k]["mean_drop_sd"] for k in ks], "-", color=C[ck], marker=MK[ck], label=lab)
    axs[1].set_xticks(range(len(ks)), [k.replace("9-99", "9+").replace("1-1", "1") for k in ks])
    axs[1].set_xlabel("Sentences in the answer")
    axs[1].set_ylabel("Mean score drop after flip\n(calibration-set SD units)")
    axs[1].legend(fontsize=6.5)
    axs[1].set_title("(b) Dilution of a single-sentence error", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "fig_locus")


if __name__ == "__main__":
    for fn in [fig_pareto, fig_escalation, fig_transfer, fig_cf_types, fig_locus]:
        try:
            fn()
        except Exception as ex:
            print(f"!! {fn.__name__}: {type(ex).__name__}: {ex}")
