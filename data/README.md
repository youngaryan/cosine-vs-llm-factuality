# Data

Nothing in `data/raw/` or `data/processed/` is redistributed with this project description;
the commands below re-create every file. All downloads are plain HTTPS GETs of public files.

## Sources

| Dataset | Used for | Files fetched | Licence (as stated by the source repository on 16 Nov 2026) | Labels |
|---|---|---|---|---|
| RAGTruth (Niu et al., ACL 2024) | evidence-grounded faithfulness; base for counterfactuals | `dataset/response.jsonl`, `dataset/source_info.jsonl` from `github.com/ParticleMedia/RAGTruth` (branch `main`) | MIT | Human span-level hallucination annotations with four types |
| FaithBench (Bao et al., NAACL 2025) | evidence-grounded faithfulness, out-of-domain target | `FaithBench.csv` from `github.com/vectara/FaithBench` (branch `main`) | CC BY-NC-SA 4.0 (repository `LICENSE` file) | Human annotations pooled per summary (`worst-label`, `best-label`) |
| EVOUNA-NQ (Wang et al., 2023) | reference-based factual correctness | `EVOUNA/NQ.json` from `github.com/wangcunxiang/QA-Eval` (branch `main`) | Apache-2.0 | Human correctness judgement per system answer |
| HaluEval-QA (Li et al., EMNLP 2023) | synthetic out-of-domain target only | `data/qa_data.json` from `github.com/RUCAIBox/HaluEval` (branch `main`) | MIT | **Synthetic**: each item pairs the gold answer with a ChatGPT-generated hallucinated answer |

FactBench was considered and not used: its public release was not checked for per-response
human labels with evidence within this study's time budget.

```bash
mkdir -p data/raw/{ragtruth,faithbench,evouna,halueval}
curl -L -o data/raw/ragtruth/response.jsonl    https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
curl -L -o data/raw/ragtruth/source_info.jsonl https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl
curl -L -o data/raw/faithbench/FaithBench.csv  https://raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv
curl -L -o data/raw/evouna/NQ.json             https://raw.githubusercontent.com/wangcunxiang/QA-Eval/main/EVOUNA/NQ.json
curl -L -o data/raw/halueval/qa_data.json      https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
```

The upstream repositories are not version-pinned by this project; record the commit hashes
if exact byte-level reproduction matters.

## Normalised schema (`data/processed/examples.jsonl`, written by `src/prepare_data.py`)

`id, dataset, domain, target, label_origin, group, split, question, evidence, reference, answer, label, generator, meta`

* `label = 1` means **unsupported** (hallucinated / incorrect); `0` means supported.
* `target` is `faithfulness` (judged against `evidence`) or `correctness` (judged against `reference`).
  The two are never pooled.
* `group` is the source document (RAGTruth, FaithBench), the question (EVOUNA) or the item
  (HaluEval). A group never crosses splits; `tests/test_core.py` checks this.
* `label_origin` is `human`, `synthetic` (HaluEval) or, in the counterfactual file, `rule` /
  `llm_paraphrase_rule_validated`.

## Label construction and exclusions

* **RAGTruth**: label 1 iff the response has at least one annotated span. Responses whose
  `quality` field is `incorrect_refusal` (144) or `truncated` (29) are excluded. Evidence is
  the passages (QA), the article (Summary), or the JSON record flattened to `key: value.` lines
  (Data2txt). The official `test` split is used in full; 200 + 100 sources per task are sampled
  from the official `train` split for router training and calibration.
* **FaithBench**: label 1 iff `worst-label` ∈ {Unwanted, Questionable}, matching the "worst"
  pooling option of the dataset's `scripts/binarize.py`. A sensitivity analysis uses `best-label`.
  30 sources → calibration, 50 → test. FaithBench deliberately contains only cases on which
  existing detectors disagreed, so absolute scores on it are low for every method.
* **EVOUNA-NQ**: 590 questions flagged `improper` and 6 answers with a missing answer or
  judgement are excluded. `golden_answer` is split on `/` into alternative references.
* **HaluEval-QA**: 1,500 of 10,000 items sampled (500 calibration, 1,000 test).

Counts per split are written to `results/dataset_stats.json`.

## Counterfactual set (`data/processed/counterfactuals.jsonl`, `src/build_counterfactuals.py`)

Built only from RAGTruth **test** responses that human annotators labelled hallucination-free.
One sentence is edited by a deterministic rule (negation, number, date/temporal, entity,
direction antonym, causal connective, role swap) or one unrelated sentence is appended
(addition). Validity filters, all automatic:

* the edited sentence shares ≥ 50 % of its content tokens with the evidence;
* number / date / entity edits replace a value that occurs in the evidence with one that does not;
* hedged sentences ("may", "reportedly", …) are skipped for negation, direction and causal edits;
* negation edits require an explicit subject and act on the root verb only;
* role swaps are restricted to two PERSON entities with a verb between them and no symmetric predicate;
* the edited sentence must not occur verbatim in the evidence.

Paraphrases of the target sentence are generated by `phi4-mini:3.8b` (not used as a judge) and
kept only if numbers, capitalised tokens and negation parity are preserved and the wording
actually changes. **None of these labels are human annotations.** `results/cf_audit_sheet.csv`
is a 160-pair sheet with empty verdict columns for a human audit that has not been carried out.
