# Experiment status

Last updated: 17 November 2026. Hardware: one CPU-only Windows laptop (6 cores, 30 GB RAM,
no GPU used). Every number in the manuscript is generated from files in `results/` by
`src/make_tables.py`; this file says which experiments those files come from.

## Completed vs. not completed

| Experiment | Status | Scale actually run | Output |
|---|---|---|---|
| Literature verification and novelty audit | done | 41 references opened on official pages | `NOVELTY_AUDIT.md`, `paper/references.bib` |
| Preregistration of RQs, hypotheses, tests | done (before any test outcome) | — | `PREREGISTRATION.md` |
| Data download and normalisation | done | RAGTruth, FaithBench, EVOUNA-NQ, HaluEval-QA | `results/dataset_stats.json` |
| FactBench | **not used** | — | reason in `data/README.md` |
| Counterfactual set | done | see `results/cf_stats.json` | `data/processed/counterfactuals.jsonl` (not redistributed) |
| Human audit of counterfactual labels | **not done** | 160-pair sheet prepared, unreviewed | `results/cf_audit_sheet.csv` |
| Lexical / structured checks | done | all splits | `results/scores_cheap.parquet` |
| Embedding cosine: MiniLM, BGE-small | done | all splits | `results/scores_cheap.parquet` |
| Embedding cosine: third model | **not run** | — | Deviation 1 |
| BERTScore | done | calib + test + counterfactuals | `results/scores_bertscore.parquet` |
| NLI (DeBERTa-v3-base) | done | calib + test + counterfactuals | `results/scores_nli.parquet` |
| Small LLM judge (llama3.2:3b) | done | **pilot subset** (sizes in `configs/experiment.yaml`) | `results/judge_small_*.jsonl` |
| Strong LLM judge (qwen2.5:14b) | done | **smaller pilot subset** | `results/judge_strong_*.jsonl` |
| Prompt sensitivity (small judge, 2 rubrics) | done | 10 sources per RAGTruth task | `results/judge_small_sensitivity_v2.jsonl` |
| Risk estimator, transfer matrix, cascade, ablations | done (2,000 bootstrap resamples) | full test sets; cascade system metrics on judged subsets | `results/*.json` |
| Clean latency benchmark | done | fixed RAGTruth sample | `results/cost.json` |
| PDF build | done | Tectonic 0.17.0, 25 pages; no undefined references or citations | `paper/main.pdf` |

## Run log

Completion times (local time, file timestamps), all on the same laptop, one job at a time:

| Stage | Finished | Notes |
|---|---|---|
| Data normalisation | 16 Nov 16:02 | `results/dataset_stats.json` |
| Counterfactual set built and validated | 16 Nov 17:04–17:58 | `results/cf_stats.json` |
| Lexical + embedding scores (re-run, Deviation 6) | 16 Nov 20:22 | |
| Small judge, natural subsets (1,635 calls) | 16 Nov 19:08 | 0 failed calls |
| BERTScore (re-run, Deviation 6) | 16 Nov 23:46 | |
| NLI, 103,900 pairs in float32 | 17 Nov 02:58 | ≈3.2 h |
| Small judge, counterfactual pairs (461 calls) | 17 Nov 03:43 | 0 failed calls |
| Small judge, reworded rubric (179 calls) | 17 Nov 04:03 | 0 failed calls |
| Strong judge, natural subsets (358 calls) | 17 Nov 05:30 | 0 failed calls; no fallback model needed |
| Strong judge, counterfactual pairs (118 calls) | 17 Nov 06:06 | 0 failed calls |
| Full analysis, 2,000 cluster-bootstrap resamples | 17 Nov 06:14 | `analysis_missing_columns.json`: none missing |
| Clean cost benchmark (incl. both judges) | 17 Nov 07:41 | second attempt, see Deviation 8 |
| Tables, figures, tests (15 passed), manuscript lint | 17 Nov | `src/make_tables.py`, `src/make_figures.py`, `src/lint_paper.py` |
| PDF compiled with Tectonic 0.17.0 | 17 Nov | `paper/main.pdf`; remaining overfull boxes are all under 11 pt |

## Deviations from `PREREGISTRATION.md`

1. **Third embedder dropped.** bge-base-en-v1.5 could not be downloaded (disk full); the
   substitute nomic-embed-text via Ollama ran at ≈6 texts/s, too slow for ≈160k texts.
   `configs/experiment.yaml` documents this; `src/score_cheap.py --skip-nomic` was used.
2. **Paraphraser** changed from gemma3:4b to phi4-mini:3.8b and from 450 to 240 bases
   (memory and time), before any paraphrase was scored.
3. **Sentence-splitter bug** (splitting after "Dr.", "U.S.") fixed after reading
   training-split counterfactuals; the counterfactual set was rebuilt before any evaluator
   ran on it.
4. **H2 test**: permutation of outcomes within base response (plus chi-square / Cramér's V).
5. **H3**: in addition to the preregistered CI criterion, a Bonferroni-adjusted one-sided
   bootstrap p-value over the 20 off-diagonal cells is reported so H3 can enter the Holm family.
6. **Crash recovery.** The first embedding run died during BGE-small encoding and the first
   BERTScore run ran out of memory, because the disk was full and Windows could not grow the
   page file while an LLM was loaded. Both were re-run one job at a time with no LLM loaded;
   `score_cheap.py --resume` keeps the lexical columns of the first run and recomputes all
   embedding columns; `run_bertscore.py` was changed to encode one evidence text at a time.
   Neither change alters any score definition.
7. **NLI precision.** The NLI checkpoint is stored in float16 and `transformers` kept that
   precision, which ran about 13× slower on this CPU (1.7 pairs/s). That run was stopped before
   it saved any score; `src/run_nli.py` and `src/cost_benchmark.py` now load the model in
   float32, and all 103,900 premise–hypothesis pairs were scored in float32.

8. **Cost benchmark re-run.** The first clean benchmark run measured every method but crashed
   while aggregating the judge rows (one judge call returned no verdict and the summary code
   assumed token counts on every call), so it saved no file. `src/cost_benchmark.py` now
   computes token and latency statistics over successful calls and records the number and error
   text of failed calls. The benchmark was re-run from scratch with nothing else running; in the
   reported run all 120 judge calls succeeded.
9. **Strong judge on H4.** The preregistration does not name the judge for H4. The confirmatory
   H4 test uses the small judge (the larger subset); the 14B-judge cascade is reported as
   exploratory.

## Known limitations of what was run

* LLM-judge results are a reduced-scale pilot on fixed source-clustered subsets.
* Counterfactual labels are rule-generated and rule-validated, not human-annotated.
* HaluEval-QA labels are synthetic.
* Bulk timing files (`results/cost_*_bulk.json`) were recorded while other jobs shared the
  CPU and are not used for cost claims; only `results/cost.json` (clean benchmark) is.
