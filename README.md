# When Is Cosine Similarity Enough?

Code, results and manuscript for an empirical study of cheap factuality evaluators (lexical
overlap, embedding cosine, BERTScore, NLI, a feature-based risk model) against local LLM
judges, on human-labelled hallucination data, rule-built counterfactual pairs, cross-domain
threshold transfer and a cost-aware cascade.

Two evaluation targets are kept apart throughout:

* **evidence-grounded faithfulness** — is the answer supported by the evidence it was written
  from? (RAGTruth, FaithBench; HaluEval-QA as a synthetic out-of-domain target)
* **reference-based correctness** — does the answer match a reference answer? (EVOUNA-NQ)

Positive class = *unsupported*. All scores are oriented "higher = more supported".

## What is here

| Path | Content |
|---|---|
| `paper/main.tex`, `paper/sections/`, `paper/references.bib` | Manuscript (ACL style). Numbers and tables are `\input` from `paper/generated/`, which `src/make_tables.py` writes from `results/*.json` |
| `PREREGISTRATION.md` | Analysis plan and hypotheses, written before any test outcome was computed |
| `NOVELTY_AUDIT.md` | Literature audit and what is / is not claimed as new |
| `EXPERIMENT_STATUS.md` | What was run, what was not, deviations from the plan |
| `INTERNAL_REVIEW.md` | Answers to the twelve quality-control questions and the claims revised as a result |
| `configs/experiment.yaml` | Seeds, models, subset sizes, thresholds |
| `data/README.md` | Dataset sources, licences, download commands, label rules |
| `src/` | Pipeline (see below) |
| `tests/` | Unit tests: metrics, thresholds, bootstrap, perturbation rules, split leakage |
| `results/` | Score tables (`*.parquet`), judge logs (`judge_*.jsonl`, one line per call), analysis outputs (`*.json`) |
| `figures/` | Figures (PDF and PNG) |

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
ollama pull llama3.2:3b && ollama pull qwen2.5:14b && ollama pull phi4-mini:3.8b
```

```bash
python src/run_all.py
```

`run_all.py` downloads the four public datasets, builds splits and counterfactuals, runs every
evaluator, the analysis, the cost benchmark, tables, figures and tests. Useful variants:

```bash
python src/run_all.py --no-judges
```

```bash
python src/run_all.py --from analyze
```

To build the PDF (needs [Tectonic](https://tectonic-typesetting.github.io), which fetches LaTeX packages on first use):

```bash
cd paper && tectonic -X compile main.tex
```

To build the arXiv upload (`arxiv/cosine_factuality_arxiv.tar.gz`; see `arxiv/SUBMISSION.md`):

```bash
python src/make_arxiv.py
```

`python src/run_all.py --list` prints the step names. Each step is an ordinary script
(`src/prepare_data.py`, `src/build_counterfactuals.py`, `src/score_cheap.py`, `src/run_nli.py`,
`src/run_bertscore.py`, `src/run_judge.py`, `src/analyze.py`, `src/cost_benchmark.py`,
`src/make_tables.py`, `src/make_figures.py`); their docstrings give the options.

To build the PDF (no LaTeX compiler was available on the machine that ran the study):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Notes on exact reproduction

* Everything ran on CPU. Expect roughly a working day of wall-clock time; the LLM judges
  dominate. Judge runs are resume-safe.
* Sampling uses seed 20261116. Judge decoding uses temperature 0 and the same seed, but
  bit-identical LLM output across machines and Ollama versions is not guaranteed; the raw
  verdict of every call is saved in `results/judge_*.jsonl` so the analysis can be re-run
  without calling a model.
* The paraphrase controls come from an LLM and are therefore subject to the same caveat;
  the accepted paraphrases are saved in `data/processed/counterfactuals.jsonl` when the
  pipeline is run. Raw and processed data are not redistributed here (see `data/README.md`).
* Behind a TLS-inspecting proxy, `pip install truststore` lets the Hugging Face downloads
  use the system certificate store (`src/common.py` picks it up automatically).
* The cost benchmark must run with nothing else on the machine.

## Integrity statement

Labels used for evaluation are the datasets' human annotations, except HaluEval-QA
(synthetic) and the counterfactual pairs (rule-generated and rule-validated, **not**
human-annotated). LLM-judge output is never used as ground truth. The LLM-judge experiments
are a reduced-scale pilot on fixed subsets; subset sizes are stated wherever a judge result
is reported.
