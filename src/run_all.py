"""Run the whole study in order. Each step is an ordinary script that can also be run on its own.

  python src/run_all.py                 # everything (many hours on a CPU-only machine)
  python src/run_all.py --from analyze  # resume at a step
  python src/run_all.py --no-judges     # skip the LLM-judge steps (cheap evaluators + analysis only)
  python src/run_all.py --list

Needs: `pip install -r requirements.txt`, and for the judge / paraphrase steps a local Ollama server with
`ollama pull llama3.2:3b qwen2.5:14b phi4-mini:3.8b`. Judge steps are resume-safe: re-running continues where a
previous run stopped. Only one LLM is kept loaded at a time because the 14B judge needs most of the RAM.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = {
    "ragtruth/response.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
    "ragtruth/source_info.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
    "faithbench/FaithBench.csv": "https://raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv",
    "evouna/NQ.json": "https://raw.githubusercontent.com/wangcunxiang/QA-Eval/main/EVOUNA/NQ.json",
    "halueval/qa_data.json": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
}
STEPS = [
    ("download", None),
    ("prepare", ["src/prepare_data.py"]),
    ("counterfactuals", ["src/build_counterfactuals.py", "--split", "test"]),
    ("paraphrase", ["src/build_counterfactuals.py", "--split", "test", "--paraphrase"]),
    ("unload_paraphraser", ["!ollama", "stop", "phi4-mini:3.8b"]),
    ("cheap", ["src/score_cheap.py", "--skip-nomic"]),
    ("nli", ["src/run_nli.py"]),
    ("bertscore", ["src/run_bertscore.py"]),
    ("judge_select", ["src/run_judge.py", "--select"]),
    ("judge_small_natural", ["src/run_judge.py", "--judge", "small", "--part", "natural"]),
    ("judge_small_cf", ["src/run_judge.py", "--judge", "small", "--part", "cf"]),
    ("judge_small_sensitivity", ["src/run_judge.py", "--judge", "small", "--part", "sensitivity", "--prompt", "v2"]),
    ("unload_small", ["!ollama", "stop", "llama3.2:3b"]),
    ("judge_strong_natural", ["src/run_judge.py", "--judge", "strong", "--part", "natural"]),
    ("judge_strong_cf", ["src/run_judge.py", "--judge", "strong", "--part", "cf"]),
    ("unload_strong", ["!ollama", "stop", "qwen2.5:14b"]),
    ("analyze", ["src/analyze.py"]),
    ("cost", ["src/cost_benchmark.py", "--strong"]),
    ("tables", ["src/make_tables.py"]),
    ("figures", ["src/make_figures.py"]),
    ("tests", ["-m", "pytest", "-q", "tests"]),
    ("lint_paper", ["src/lint_paper.py"]),
]
JUDGE_STEPS = {"paraphrase", "unload_paraphraser", "judge_select", "judge_small_natural", "judge_small_cf",
               "judge_small_sensitivity", "unload_small", "judge_strong_natural", "judge_strong_cf", "unload_strong"}


def download():
    for rel, url in DOWNLOADS.items():
        dst = RAW / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--no-judges", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [n for n, _ in STEPS]
    if a.list:
        return print("\n".join(names))
    if a.start and a.start not in names:
        sys.exit(f"unknown step {a.start}; choose from {names}")
    started = a.start is None
    for name, cmd in STEPS:
        started = started or name == a.start
        if not started or (a.no_judges and name in JUDGE_STEPS):
            continue
        print(f"\n=== {name} ===", flush=True)
        if cmd is None:
            download()
            continue
        if cmd[0].startswith("!"):                                  # external command; failure is not fatal
            subprocess.run([cmd[0][1:]] + cmd[1:], cwd=ROOT)
            continue
        if name == "cost" and a.no_judges:
            continue
        subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
