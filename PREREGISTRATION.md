# Analysis plan, fixed before any test-split outcome was computed

Written 16 November 2026, after inspecting dataset schemas and label counts only. No
evaluator had been run on any test example when this file was written. Later departures are
logged in `EXPERIMENT_STATUS.md` under "Deviations".

## Scale
The machine is a CPU-only laptop (6 cores, 30 GB RAM, no discrete GPU). Cheap evaluators run
on full test sets. LLM judges run on stratified, source-clustered subsets and are reported as
a **reduced-scale pilot**; the stronger judge runs on a smaller subset still.

## Targets and labels
Positive class everywhere: **unsupported** (1). Supported = 0. Scores are oriented so that a
higher score means *more supported*; an example is accepted when score ≥ threshold.

| Dataset | Target | Label = 1 when | Label origin |
|---|---|---|---|
| RAGTruth (QA, Summary, Data2txt) | evidence-grounded faithfulness | ≥ 1 annotated hallucination span | human |
| FaithBench | evidence-grounded faithfulness | worst annotator label ∈ {Unwanted, Questionable} (sensitivity: best label) | human |
| EVOUNA-NQ | reference-based factual correctness | human judgement = incorrect | human |
| HaluEval-QA | evidence-grounded faithfulness | answer is the generated hallucinated one | **synthetic** |
| Counterfactual set | evidence-grounded faithfulness | rule-perturbed copy of a human-labelled faithful RAGTruth response | **rule-generated, rule-validated** |

RAGTruth responses with quality ≠ "good" and EVOUNA items flagged improper or with a missing
judgement are excluded and counted.

## Splits (grouped by source document / question)
* RAGTruth official train → sampled sources split into `train` (router fitting) and `calib`
  (thresholds, probability calibration). Official test → `test`.
* FaithBench, EVOUNA-NQ, HaluEval-QA: seeded group split into train/calib/test.
* Counterfactuals are built only from RAGTruth **test** sources; perturbation rules are
  developed by reading examples from **train** sources only.
* Seed 20261116 for all sampling.

## Evaluators
Lexical (exact match, token F1, TF-IDF cosine, evidence token precision); embedding cosine
with all-MiniLM-L6-v2 (primary), bge-small-en-v1.5, bge-base-en-v1.5, in document-level,
mean-of-max and min-of-max chunk configurations; BERTScore (roberta-base); NLI
(DeBERTa-v3-base-mnli-fever-anli, sentence-vs-retrieved-chunk); small judge llama3.2:3b;
stronger judge qwen2.5:14b; cosine-threshold cascade; proposed feature-based risk cascade;
random-escalation control.

**Primary cosine configuration**: MiniLM mean-of-max over evidence chunks (evidence-based)
and candidate-vs-reference cosine (reference-based).

## Judge label mapping
SUPPORTED → 0. UNSUPPORTED → 1. INSUFFICIENT_EVIDENCE → 1 for the binary metric (the
datasets label unsupported additions as hallucinations), and counted separately. Unparseable
output after one retry → counted as a judge failure and scored as 1.

## Operating points
Thresholds are chosen on `calib` only: (a) maximum balanced accuracy; (b) the largest
acceptance region whose calibration false-acceptance rate (FAR = accepted unsupported /
all unsupported) is ≤ α, α ∈ {0.05 (primary), 0.10, 0.20}.

## Hypotheses and how each is tested
* **H1** — With thresholds from natural calibration data, cosine accepts faithful
  paraphrases but fails to reject single-fact flips. Test: paired flip-detection rate of
  primary cosine vs NLI and vs the small judge; cluster-bootstrap CI of the difference.
* **H2** — Detection differs by perturbation type. Test: per-type flip rejection rates with
  CIs and a permutation test of homogeneity per evaluator.
* **H3** — A FAR ≤ 5 % cosine threshold from one domain exceeds 5 % FAR on at least one other
  domain (lower 95 % bound > 5 %).
* **H4** — At the same cheap-stage FAR budget, the feature-based risk score resolves a larger
  share of examples without a judge call than the cosine-only cascade (CI of the coverage
  difference excludes 0), with system balanced accuracy no more than 2 points below
  always-judge. A random-escalation control with the same escalation share is reported.

A null or negative outcome on any hypothesis is reported as such.

## Statistics
Cluster bootstrap by source (2,000 resamples, percentile intervals), paired where methods
share examples. Holm correction over the family {H1 (2 contrasts), H3, H4}. Effect sizes are
reported as absolute differences. Calibration: Brier score and 10-bin ECE.

## Cost
Wall-clock and token counts recorded per call on the same machine. Embedding cost includes
encoding; cold-start (model load) is reported separately from warm cost. Judge calls reuse
the prompt-prefix cache when consecutive calls share evidence; cached and uncached token
counts are both logged. No currency figures are reported because no paid API is used.
