# Novelty audit

Audit date: 16 November 2026. Every record below was opened on its official page
(ACL Anthology, arXiv, IJCAI, ICLR, NeurIPS proceedings) during this audit. Descriptions
paraphrase the abstract or HTML full text; numbers are quoted only where they were read on
the page. "Not read" means the detail was not on the page that was fetched, not that it is
absent from the paper. This is a bounded search, not proof that no overlapping work exists.

## 1. What is already established (not claimed as new here)

| Idea | Precedent |
|---|---|
| Embedding/semantic similarity is not factual correctness | Silva & Sanfilippo (IJCAI 2026); Sinha (arXiv 2512.15068); Steck et al. (WWW 2024 Companion) |
| Embedding cosine vs. NLI vs. LLM judge for hallucination detection | Sinha 2025 (BGE-base cosine, DeBERTa-v3-large-MNLI, lexical, GPT-4o-mini judge; HaluEval AUC 0.72 / 0.81 / 0.65 / 0.94) |
| Perturbing answers to test factuality metrics | FactCC (Kryscinski et al. 2020); FRANK (Pagnoni et al. 2021); CheckList (Ribeiro et al. 2020); Gharsallah et al. (arXiv 2609.15561); Nguyen et al. (ICCCI 2026) |
| Embeddings are insensitive to negation | Cao (ECAI 2025, arXiv 2504.00584); Anschütz et al. (INLG 2023) |
| Selective / cascaded judging with risk control | Jung et al. (ICLR 2025); FrugalGPT (Chen et al. 2023); Geifman & El-Yaniv (2017); Learn-then-Test and Conformal Risk Control (Angelopoulos et al.) |
| Cheap detector → LLM judge cascades for hallucination | Liu et al. (arXiv 2610.37229); Li et al. JEV-as-a-Judge (arXiv 2609.26550); Do Rosario & Atef (arXiv 2610.20634); Valentin et al. (KDD GenAI Eval WS 2024) |
| Combining cheap metrics with LLM judges cost-consciously | Rahnamoun & Shamsfard (COLING 2025) |
| Hallucination detectors fail out of distribution | Dubanowska et al. (Findings EMNLP 2025, hidden-state probes); Godbole & Jia (Findings ACL 2025, LLM/NLI metrics); HALT-RAG (arXiv 2509.07475, per-task thresholds) |
| Calibrated multi-score combination | Valentin et al. 2024; Li, Magesh & Veeravalli (Findings ACL 2026); HALT-RAG |

## 2. Closest prior work, paper by paper

| Paper | Task / data | Evaluators | Perturbation test | Threshold transfer across domains | Implemented cheap→LLM cascade | Measured compute cost |
|---|---|---|---|---|---|---|
| Sinha 2025 (arXiv) | Conformal hallucination detection; NQ-Open synthetic swaps, HaluEval | cosine (BGE-base), NLI, lexical, combiner, GPT-4o-mini judge (n=200) | No (synthetic answer swaps as data, not paired tests) | **No** — 60/40 split inside each dataset; shift named as future work | **No** — proposed as future work | API price estimates |
| Gharsallah et al. 2026 (arXiv; ext. of EvalLLM 2025) | Meta-evaluation of factuality metrics; 100 NQ pairs → 500 | 6 open-weight LLM judges, RAGAS, LLM+NLI | Yes (paraphrase + 4 severity levels, LLM-edited) | No | No | No |
| Silva & Sanfilippo 2026 (IJCAI) | Legal RAG, 3,012 synthetic instances | BERTScore vs GPT-4o-mini judge | No | No (single domain) | No | Not read |
| Rahnamoun & Shamsfard 2025 (COLING) | Open-domain QA evaluation | lexical, semantic, LLM judges, layered fusion | No | Not read | Layered metric scheme | Not read |
| Jung et al. 2025 (ICLR) | Preference judging with human-agreement guarantee | LLM→LLM cascade | No | No | Yes (LLM→LLM) | Yes |
| Liu et al. 2026 (arXiv) | 9,852 items, five hallucination types | decision models vs generative judge | No | Not read | Yes (classifier cascade) | Yes (1–3% of judge cost) |
| Do Rosario & Atef 2026 (arXiv) | 495 LLM-AggreFact claims | decision model vs frontier LLM | No | No | Yes; **random-escalation control matched the confidence gate** | Yes (USD) |
| Li et al. 2026, JEV (arXiv) | General judging, 1,610 held-out pairs | decision judge → reasoning judge | Style-adversarial pairs | Frozen threshold on new workloads | Yes | Yes |
| Nguyen et al. 2026 (ICCCI) | Vietnamese hallucination detection, 1,969 minimal-edit pairs | fine-tuned Qwen detectors | Yes (fix / delete / add / replace) | No | No | No |
| Dubanowska et al. 2025 (Findings EMNLP) | RAGTruth and others | hidden-state probes | No | Yes (near-chance OOD) | No | No |
| Godbole & Jia 2025 (Findings ACL) | 5 metrics × 11 datasets | LLM/NLI factuality metrics | Paraphrase bias analysis | Cross-dataset disagreement | No | No |
| PARALLAX 2026 (arXiv) | 22 methods × 6 corpora | incl. text-embedding similarity baseline | Perturbation probes mentioned in a search snippet only (not verified) | — | No | No |

## 3. Gap that this study addresses

No paper located in this audit does all of the following together, and Sinha (2025) names
the first two as future work:

1. **Matched-locus counterfactual triplets on long, human-labelled faithful RAG responses.**
   For one response we edit one sentence in two ways — a meaning-preserving paraphrase and a
   single-fact flip — and score both against the same evidence with thresholds frozen on
   natural data. This yields an *inversion rate*: how often an evaluator ranks the factual
   flip as better supported than the faithful paraphrase. Gharsallah et al. test LLM judges
   on short NQ answers with LLM-made edits and no similarity baseline; FactCC/FRANK train or
   benchmark metrics rather than probing thresholded cosine; Nguyen et al. probe fine-tuned
   detectors in Vietnamese.
2. **A cross-domain transfer matrix for thresholds set to a false-acceptance target**, across
   three RAGTruth tasks, FaithBench and HaluEval-QA, for cosine, a feature-based risk model,
   and NLI; plus shifts across generator model families and answer length.
3. **An implemented embedding-first cascade with measured local compute cost**, compared
   against a cosine-only cascade, an always-judge policy and a random-escalation control
   (the control that Do Rosario & Atef showed can erase an apparent gating benefit).

## 4. What is *not* claimed

* The cascade is a standard two-threshold selective scheme. It is not a new algorithm; the
  question tested is whether cheap structured features change coverage at a fixed
  false-acceptance budget, and whether the thresholds survive a shift.
* No distribution-free guarantee is claimed out of distribution.
* The diagnostics "paired flip detection rate", "high-similarity hallucination acceptance",
  "paraphrase invariance", "threshold transfer" and "cost-constrained coverage" are
  re-statements of pair accuracy, false-acceptance rate above a threshold, false-rejection
  rate, transfer gap and selective coverage. Only the matched-locus inversion rate is, to our
  knowledge, not reported in the works above.
* Counterfactual pairs are rule-generated and rule-validated; they are not human-annotated.

## 5. Bibliographic caveats found during verification

* Gharsallah et al.: the arXiv record (14 Sep 2026) is an extended version of an EvalLLM 2025
  (CORIA-TALN) paper; the workshop proceedings record was not located, so the arXiv version is cited.
* Sinha 2025 and Liu et al. 2026 are preprints with no peer-reviewed venue shown.
* BERTScore: ICLR 2020 venue rests on the authors' arXiv comment (OpenReview blocked the fetch).
* C-Pack, EVOUNA, MT-Bench, Cao, Anschütz et al.: venue taken from the authors' arXiv comment.
* FrugalGPT, Learn-then-Test, Conformal Risk Control: cited as arXiv; journal versions not confirmed.
* Rahnamoun & Shamsfard: no DOI shown on the Anthology page. Jung et al.: no pages/DOI on the ICLR page.
