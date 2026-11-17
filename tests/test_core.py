"""Unit tests and sanity checks for metrics, thresholds, text utilities, labels and perturbations.

Run from the project root:  python -m pytest -q tests
Tests that need generated data are skipped when the files are absent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import (PROC, auroc, binary_metrics, chunk_evidence, cluster_bootstrap, ece_brier, neg_count,  # noqa: E402
                    numbers, read_jsonl, sentences, thr_far, thr_frr, thr_max_bacc)


# ----------------------------------------------------------------------------- metrics
def test_binary_metrics_positive_class_is_unsupported():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    pred = [1, 1, 0, 0, 0, 0, 0, 1]
    m = binary_metrics(y, pred)
    assert m["recall"] == pytest.approx(2 / 3)           # hallucination recall
    assert m["far"] == pytest.approx(1 / 3)              # unsupported answers accepted
    assert m["frr"] == pytest.approx(1 / 5)              # supported answers rejected
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_kappa_perfect_and_chance():
    assert binary_metrics([0, 1, 0, 1], [0, 1, 0, 1])["kappa"] == pytest.approx(1.0)
    assert binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])["kappa"] == pytest.approx(0.0)


def test_auroc_matches_sklearn_and_handles_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)            # many ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_ece_brier_perfect_calibration():
    ece, brier = ece_brier([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
    assert ece == pytest.approx(0) and brier == pytest.approx(0)


# -------------------------------------------------------------------------- thresholds
def test_thr_far_respects_alpha_on_calibration_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(size=2000) - 0.8 * y                  # support score: lower for unsupported
    for alpha in (0.05, 0.1, 0.2):
        t = thr_far(y, s, alpha)
        far = (s[y == 1] >= t).mean()
        assert far <= alpha + 1e-12
        assert far >= alpha - 2 / (y == 1).sum()         # and not needlessly conservative


def test_thr_far_with_ties_never_exceeds_alpha():
    y = np.array([1] * 10 + [0] * 10)
    s = np.array([0.5] * 10 + [0.9] * 10)
    t = thr_far(y, s, 0.2)
    assert (s[y == 1] >= t).mean() <= 0.2


def test_thr_frr_bounds_false_rejections():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    s = rng.normal(size=1000) - y
    t = thr_frr(y, s, 0.1)
    assert (s[y == 0] < t).mean() <= 0.1 + 1e-12


def test_thr_max_bacc_separates_clean_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    t = thr_max_bacc(y, s)
    assert ((s < t).astype(int) == y).all()


def test_cluster_bootstrap_ci_contains_point():
    g = np.repeat(np.arange(50), 4)
    x = np.random.default_rng(3).normal(size=200)
    p, lo, hi, _ = cluster_bootstrap(g, lambda i: x[i].mean(), n_boot=300)
    assert lo <= p <= hi


# ------------------------------------------------------------------------ text utilities
def test_sentence_split_and_chunking():
    text = "Dr. Smith arrived at 5 p.m. on Monday. He left on Tuesday! Was it late? Yes."
    s = sentences(text)
    assert s[0].startswith("Dr. Smith") and len(s) >= 3
    ch = chunk_evidence("One two three. Four five six. Seven eight nine.", window=2)
    assert ch == ["One two three. Four five six.", "Four five six. Seven eight nine."]


def test_number_and_negation_extraction():
    assert set(numbers("It cost $1,200.50 and rose 20% in 2019, twice.")) >= {"1200.5", "20", "2019"}
    assert neg_count("He did not go and never called.") == 2
    assert neg_count("She can't swim.") >= 1
    assert neg_count("The treatment reduced mortality.") == 0


# --------------------------------------------------------------- perturbation generators
def test_perturbation_rules_change_exactly_the_intended_fact():
    bc = pytest.importorskip("build_counterfactuals")
    rng = np.random.default_rng(0)
    ev = "The treatment reduced mortality by 20% in 2019 because doses increased on Monday."
    low = ev.lower()
    s2, info = bc.ed_number("The treatment reduced mortality by 20% overall.", ev, low, rng, {})
    assert info["old"] == "20" and info["new"] != "20" and info["new"] not in ev
    s2, info = bc.ed_date("The study ran in 2019.", ev, low, rng, {})
    assert info["old"] == "2019" and info["new"] not in ev
    s2, info = bc.ed_negation("The treatment reduced mortality by 20%.", ev, low, rng, {})
    assert "did not reduce" in s2
    s2, info = bc.ed_negation("The drug is not approved.", ev, low, rng, {})
    assert s2 == "The drug is approved."
    s2, info = bc.ed_direction("Doses increased over time.", ev, low, rng, {})
    assert "decreased" in s2
    s2, info = bc.ed_causal("Mortality fell because doses increased.", ev, low, rng, {})
    assert "although" in s2
    assert bc.ed_negation("It may reduce mortality.", ev, low, rng, {}) is None          # hedged claims are skipped


# ------------------------------------------------------------------ generated-data checks
@pytest.mark.skipif(not (PROC / "examples.jsonl").exists(), reason="run src/prepare_data.py first")
def test_splits_have_no_group_leakage_and_labels_are_binary():
    seen = {}
    for e in read_jsonl(PROC / "examples.jsonl"):
        assert e["label"] in (0, 1)
        assert seen.setdefault(e["group"], e["split"]) == e["split"]
        if e["target"] == "correctness":
            assert e["reference"] and not e["evidence"]
        else:
            assert e["evidence"]


@pytest.mark.skipif(not (PROC / "counterfactuals.jsonl").exists(), reason="run src/build_counterfactuals.py first")
def test_counterfactuals_are_single_edits_of_faithful_test_responses():
    ex = {e["id"]: e for e in read_jsonl(PROC / "examples.jsonl")}
    rows = read_jsonl(PROC / "counterfactuals.jsonl")
    orig = {r["meta"]["base"]: r for r in rows if r["meta"]["type"] == "original"}
    for r in rows:
        base = ex[r["meta"]["base"]]
        assert base["split"] == "test" and base["label"] == 0 and base["dataset"] == "ragtruth"
        assert r["group"] == base["group"] and r["evidence"] == base["evidence"]
        t = r["meta"]["type"]
        if t == "original":
            assert r["answer"] == base["answer"] and r["label"] == 0
        elif t == "paraphrase":
            assert r["label"] == 0 and r["answer"] != base["answer"]
            assert r["answer"].replace(r["meta"]["edited_sentence"], r["meta"]["original_sentence"], 1) == base["answer"]
        elif t == "addition":
            assert r["label"] == 1 and r["answer"].startswith(base["answer"].rstrip())
        else:
            assert r["label"] == 1 and r["answer"] != base["answer"]
            assert r["answer"].replace(r["meta"]["edited_sentence"], r["meta"]["original_sentence"], 1) == base["answer"]
            assert r["meta"]["base"] in orig


@pytest.mark.skipif(not (PROC / "judge_subsets.json").exists(), reason="run src/run_judge.py --select first")
def test_strong_judge_subset_is_nested_in_small_subset():
    sub = json.loads((PROC / "judge_subsets.json").read_text())
    assert set(sub["strong"]["groups"]) <= set(sub["small"]["groups"])
    assert set(sub["strong"]["evouna_ids"]) <= set(sub["small"]["evouna_ids"])
    assert set(sub["strong"]["cf_bases"]) <= set(sub["small"]["cf_bases"])
