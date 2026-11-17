"""Loading of all score tables, the risk estimator (router) and shared evaluation helpers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (CFG, RESULTS, SEED, auprc, auroc, binary_metrics, cluster_bootstrap, read_jsonl, thr_far,
                    thr_frr, thr_max_bacc)

FAITH_DOMAINS = ["ragtruth_qa", "ragtruth_summary", "ragtruth_data2txt", "faithbench", "halueval_qa"]
RAG_DOMAINS = FAITH_DOMAINS[:3]
PRIMARY_COS = "minilm_mm"
MISSING: list[str] = []

# feature groups of the risk estimator (all available at inference time)
GROUPS = {
    "cosine": ["minilm_mm", "minilm_min", "minilm_doc"],
    "lexical": ["lex_tok_prec", "lex_f1", "lex_tfidf", "chk_min_sent_prec", "chk_novel_bigram"],
    "number": ["chk_num_ratio", "log_num_unsup"],
    "entity": ["chk_ent_ratio", "log_ent_unsup"],
    "negation": ["chk_neg_mismatch"],
    "length": ["log_len_ans", "len_ans_sent", "log_len_ev"],
}
REF_FEATURES = ["minilm_doc", "minilm_qa", "lex_f1", "lex_contains", "lex_em", "lex_tok_prec", "lex_tfidf",
                "chk_num_unsup", "chk_neg_mismatch", "log_len_ans"]
ROUTER_VARIANTS = {
    "risk_cos": ["cosine"],                                             # cosine features only
    "risk_cos_checks": ["cosine", "number", "entity", "negation"],       # + lightweight factual checks
    "risk_full": list(GROUPS),                                           # + lexical grounding and length
}

SCORER_LABELS = {
    "lex_em": "Exact match", "lex_contains": "Lexical containment", "lex_f1": "Token F1",
    "lex_tfidf": "TF-IDF cosine", "lex_tok_prec": "Token precision",
    "minilm_doc": "MiniLM cosine (doc)", "minilm_mm": "MiniLM cosine (mean-max)", "minilm_min": "MiniLM cosine (min-max)",
    "bge_small_doc": "BGE-small cosine (doc)", "bge_small_mm": "BGE-small cosine (mean-max)",
    "bge_small_min": "BGE-small cosine (min-max)", "bertscore": "BERTScore (roberta-base)",
    "nli_mean_ec": "NLI (SummaC-ZS style)", "nli_min_e": "NLI (min entailment)",
    "risk_cos": "Risk model: cosine only", "risk_cos_checks": "Risk model: cosine + checks",
    "risk_full": "Risk model: full", "judge_small": "Judge llama3.2:3b", "judge_strong": "Judge qwen2.5:14b",
}


def load_scores() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / "scores_cheap.parquet")
    for extra in ["scores_nli.parquet", "scores_bertscore.parquet"]:
        p = RESULTS / extra
        if p.exists():
            df = df.join(pd.read_parquet(p))
    # A partially finished pipeline can be analysed for debugging; missing score columns are filled with a constant,
    # listed in MISSING and written to results/analysis_missing_columns.json. A final analysis must have MISSING == [].
    for c in ["chk_neg_mismatch", "bge_small_mm", "bge_small_min", "bge_small_doc", "bertscore", "nli_mean_ec", "nli_min_e"]:
        if c not in df:
            df[c] = 0.0
            MISSING.append(c)
    df["log_num_unsup"] = np.log1p(df["chk_num_unsup"])
    df["log_ent_unsup"] = np.log1p(df["chk_ent_unsup"])
    df["log_len_ans"] = np.log1p(df["len_ans_tok"])
    df["log_len_ev"] = np.log1p(df["len_ev_tok"])
    if "lex_contains" not in df:
        df["lex_contains"] = np.nan
    return df


def load_judge(judge: str, part: str, prompt: str = "v1") -> pd.DataFrame:
    p = RESULTS / f"judge_{judge}_{part}_{prompt}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    j = pd.DataFrame(read_jsonl(p)).drop_duplicates("id").set_index("id")
    j["pred"] = (j["verdict"] != "SUPPORTED").astype(int)             # INSUFFICIENT / failure -> unsupported
    j["failed"] = j["verdict"].isna()
    return j


def attach_judges(df: pd.DataFrame) -> pd.DataFrame:
    for judge in ["small", "strong"]:
        parts = [load_judge(judge, part) for part in ["natural", "cf"]]
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        j = pd.concat(parts)
        j = j[~j.index.duplicated()]
        df[f"judge_{judge}_pred"] = j["pred"].reindex(df.index)
        df[f"judge_{judge}_verdict"] = j["verdict"].reindex(df.index)
        df[f"judge_{judge}_psup"] = j["p_supported"].reindex(df.index)
        for c in ["prompt_tokens", "cached_tokens", "output_tokens", "wall_s", "prompt_s", "gen_s"]:
            df[f"judge_{judge}_{c}"] = j[c].reindex(df.index)
    return df


# ------------------------------------------------------------------------------ router
class Router:
    """Logistic regression risk estimator, P(unsupported), isotonic-calibrated on a calibration set."""

    def __init__(self, features):
        self.features = list(features)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
        self.iso = None

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame | None = None):
        self.model.fit(train[self.features].fillna(0.0).values, train["label"].values)
        if calib is not None and calib["label"].nunique() == 2:
            raw = self.raw(calib)
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, calib["label"].values)
        return self

    def raw(self, df):
        return self.model.predict_proba(df[self.features].fillna(0.0).values)[:, 1]

    def risk(self, df):
        """Uncalibrated probability: used for ranking and thresholds (strictly monotone, no ties)."""
        return self.raw(df)

    def prob(self, df):
        """Isotonic-calibrated probability: used only for calibration metrics."""
        r = self.raw(df)
        return self.iso.predict(r) if self.iso is not None else r

    def coefficients(self):
        lr = self.model.named_steps["logisticregression"]
        return dict(zip(self.features, lr.coef_[0].round(4).tolist()))


def feature_list(groups):
    return [f for g in groups for f in GROUPS[g]]


def fit_routers(df: pd.DataFrame, train_domains=RAG_DOMAINS, variants=ROUTER_VARIANTS, tag=""):
    """Fit evidence-mode routers on the training split of `train_domains`; add support-score columns in place."""
    tr = df[(df["split"] == "train") & df["domain"].isin(train_domains)]
    ca = df[(df["split"] == "calib") & df["domain"].isin(train_domains)]
    faith = df["target"] == "faithfulness"
    routers = {}
    for name, groups in variants.items():
        r = Router(feature_list(groups)).fit(tr, ca)
        routers[name] = r
        df.loc[faith, name + tag] = -r.risk(df[faith])               # support orientation
        df.loc[faith, name + tag + "_prob"] = r.prob(df[faith])
    return routers


def fit_ref_router(df: pd.DataFrame):
    tr = df[(df["split"] == "train") & (df["domain"] == "evouna_nq")]
    ca = df[(df["split"] == "calib") & (df["domain"] == "evouna_nq")]
    m = df["domain"] == "evouna_nq"
    out = {}
    for name, feats in {"risk_cos": ["minilm_doc", "minilm_qa"], "risk_full": REF_FEATURES}.items():
        r = Router(feats).fit(tr, ca)
        df.loc[m, name] = -r.risk(df[m])
        df.loc[m, name + "_prob"] = r.prob(df[m])
        out[name] = r
    df.loc[m, "risk_cos_checks"] = df.loc[m, "risk_full"]
    df.loc[m, "risk_cos_checks_prob"] = df.loc[m, "risk_full_prob"]
    return out


# ------------------------------------------------------------------------- evaluation
def eval_scorer(test: pd.DataFrame, calib: pd.DataFrame, col: str, boot: bool = True, n_boot=None) -> dict:
    """Threshold-free metrics on test plus metrics at the calibration max-balanced-accuracy threshold."""
    t = test.dropna(subset=[col])
    c = calib.dropna(subset=[col])
    y, s, g = t["label"].values, t[col].values.astype(float), t["group"].values
    out = dict(n=int(len(t)), n_unsupported=int(y.sum()), auroc=auroc(y, -s), auprc=auprc(y, -s))
    if len(c) and c["label"].nunique() == 2:
        thr = thr_max_bacc(c["label"].values, c[col].values.astype(float))
        pred = (s < thr).astype(int)
        out.update(threshold=thr, **{k: v for k, v in binary_metrics(y, pred).items() if k != "n"})
        if boot:
            def stat(idx):
                return np.array([auroc(y[idx], -s[idx]), binary_metrics(y[idx], pred[idx])["balanced_accuracy"]])
            _, lo, hi, _ = cluster_bootstrap(g, stat, n_boot)
            out.update(auroc_ci=[float(lo[0]), float(hi[0])], bacc_ci=[float(lo[1]), float(hi[1])])
    return out


def eval_pred(test: pd.DataFrame, pred_col: str, boot: bool = True, n_boot=None) -> dict:
    t = test.dropna(subset=[pred_col])
    y, pred, g = t["label"].values, t[pred_col].values.astype(int), t["group"].values
    out = dict(n=int(len(t)), n_unsupported=int(y.sum()), **{k: v for k, v in binary_metrics(y, pred).items() if k != "n"})
    if boot and len(t):
        _, lo, hi, _ = cluster_bootstrap(g, lambda idx: binary_metrics(y[idx], pred[idx])["balanced_accuracy"], n_boot)
        out["bacc_ci"] = [float(lo), float(hi)]
    return out


def paired_diff(test: pd.DataFrame, pred_a, pred_b, metric="balanced_accuracy", n_boot=None) -> dict:
    """Cluster-bootstrap CI for metric(a) - metric(b) on the same examples."""
    y, g = test["label"].values, test["group"].values
    a, b = np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)

    def stat(idx):
        return binary_metrics(y[idx], a[idx])[metric] - binary_metrics(y[idx], b[idx])[metric]

    p, lo, hi, samples = cluster_bootstrap(g, stat, n_boot)
    pval = float(min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean())))
    return dict(diff=float(p), ci=[float(lo), float(hi)], p_boot=pval)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def dump(name: str, obj):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=default), encoding="utf8")
