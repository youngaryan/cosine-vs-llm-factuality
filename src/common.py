"""Shared paths, text utilities, metrics and bootstrap helpers."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

try:  # use the operating-system certificate store when present (needed behind TLS-inspecting proxies)
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - optional dependency
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
CFG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf8"))
SEED = CFG["seed"]


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- text
_ABBR = "".join(rf"(?<!\b{a}\.)" for a in ["Mr", "Mrs", "Ms", "Dr", "St", "vs", r"U\.S", r"e\.g", r"i\.e", "No", "Inc", "Jr", "Sr", "Sen", "Gov", "Rep", "Gen", "Prof", "Co", "Corp", "Ltd", r"a\.m", r"p\.m"])
_SENT = re.compile(r"(?<=[.!?])" + _ABBR + r"[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[.,'][0-9]+)*")
STOP = set(
    "a an the and or but if while of at by for with about against between into through during before after "
    "above below to from up down in out on off over under again further then once here there when where why "
    "how all any both each few more most other some such no nor not only own same so than too very s t can "
    "will just don should now is are was were be been being have has had having do does did doing i me my "
    "we our you your he him his she her it its they them their what which who whom this that these those am "
    "as also would could may might must shall".split()
)


def sentences(text: str) -> list[str]:
    out = [s.strip() for s in _SENT.split(text or "") if s and s.strip()]
    return [s for s in out if len(s) > 1]


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in STOP]


def chunk_evidence(text: str, max_words: int | None = None, window: int | None = None) -> list[str]:
    """Evidence -> overlapping chunks of `window` consecutive sentences; long sentences are windowed."""
    max_words = max_words or CFG["chunking"]["max_chunk_words"]
    window = window or CFG["chunking"]["window_sentences"]
    units = []
    for s in sentences(text):
        w = s.split()
        if len(w) <= max_words:
            units.append(s)
        else:
            step = max_words // 2
            for i in range(0, max(1, len(w) - step), step):
                units.append(" ".join(w[i : i + max_words]))
    if not units:
        return [text.strip() or "."]
    if len(units) <= window:
        return [" ".join(units)]
    return [" ".join(units[i : i + window]) for i in range(len(units) - window + 1)]


_NUM = re.compile(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?")
_NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000",
}


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def numbers(text: str, words: bool = True) -> list[str]:
    out = [norm_num(m.group()) for m in _NUM.finditer(text or "")]
    if words:
        out += [_NUMWORDS[w] for w in tokens(text) if w in _NUMWORDS and w != "one"]
    return [n for n in out if n]


_NEG = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot|n't|unable|fail(?:s|ed)?|den(?:y|ies|ied)|"
    r"lack(?:s|ed)?|refus(?:e|es|ed))\b|n't\b",
    re.I,
)


def neg_count(text: str) -> int:
    return len(_NEG.findall(text or ""))


_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]+(?:\s+(?:[A-Z][a-zA-Z]+|of|de|van))*\s*[A-Z]?[a-zA-Z]*\b")


def proper_tokens(text: str) -> list[str]:
    """Lower-cased tokens of capitalised (non sentence-initial) words: a cheap proxy for named entities."""
    out = []
    for s in sentences(text):
        words = s.split()
        for w in words[1:]:
            w2 = re.sub(r"[^A-Za-z]", "", w)
            if len(w2) > 2 and w2[0].isupper() and w2.lower() not in STOP:
                out.append(w2.lower())
    return out


# -------------------------------------------------------------------------- metrics
def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def binary_metrics(y, pred) -> dict:
    """Positive class = unsupported (1). `pred` = 1 means the evaluator rejects the answer."""
    tp, tn, fp, fn = confusion(y, pred)
    n = tp + tn + fp + fn
    rec = tp / (tp + fn) if tp + fn else float("nan")       # hallucination recall
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1p = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    f1n = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else float("nan")
    po = (tp + tn) / n if n else float("nan")
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n) if n else float("nan")
    return {
        "n": n,
        "accuracy": po,
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "f1_unsupported": f1p,
        "macro_f1": (f1p + f1n) / 2,
        "far": 1 - rec,              # false-acceptance rate: unsupported answers accepted
        "frr": 1 - spec,             # false-rejection rate: supported answers rejected
        "kappa": (po - pe) / (1 - pe) if n and pe < 1 else float("nan"),
    }


def auroc(y, risk) -> float:
    """AUROC of a risk score for the unsupported class (ties handled by average rank)."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    risk = np.asarray(risk, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(risk)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, risk) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, risk))


def ece_brier(y, p, bins: int = 10) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def thr_max_bacc(y, support) -> float:
    """Threshold on a support score (accept if score >= t) maximising balanced accuracy."""
    y = np.asarray(y).astype(int)
    s = np.asarray(support, dtype=float)
    cand = np.unique(s)
    cand = np.concatenate([cand, [cand.max() + 1e-9]])
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 401))
    best, bt = -1.0, cand[0]
    n1, n0 = max(1, y.sum()), max(1, (1 - y).sum())
    for t in cand:
        rej = s < t
        b = ((rej & (y == 1)).sum() / n1 + ((~rej) & (y == 0)).sum() / n0) / 2
        if b > best:
            best, bt = b, t
    return float(bt)


def thr_far(y, support, alpha: float) -> float:
    """Smallest threshold t (largest acceptance region) with calibration FAR <= alpha.

    FAR = share of unsupported answers with score >= t. Returns +inf if no finite threshold qualifies.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(support, dtype=float)
    pos = np.sort(s[y == 1])
    if len(pos) == 0:
        return float(s.min())
    k = int(math.floor(alpha * len(pos)))            # number of unsupported answers we may accept
    if k == 0:
        return float(np.nextafter(pos[-1], np.inf))
    t = pos[len(pos) - k]                            # accept the k highest-scoring unsupported answers
    # ties: move up until FAR <= alpha holds strictly
    while (pos >= t).sum() > k:
        bigger = pos[pos > t]
        if len(bigger) == 0:
            return float(np.nextafter(pos[-1], np.inf))
        t = bigger.min()
    return float(t)


def thr_frr(y, support, beta: float) -> float:
    """Largest threshold t such that at most beta of supported answers fall below t (are rejected)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(support, dtype=float)
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return float(s.min())
    k = int(math.floor(beta * len(neg)))
    return float(neg[k])                             # scores < neg[k] are at most k items (ties aside)


def cluster_bootstrap(groups, stat_fn, n_boot: int | None = None, seed: int = SEED):
    """Percentile CI for stat_fn(indices) resampling whole groups. Returns (point, lo, hi, samples)."""
    n_boot = n_boot or CFG["stats"]["bootstrap"]
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(len(groups)))
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = stat_fn(idx)
        samples.append(v)
    samples = np.asarray(samples, dtype=float)
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return point, lo, hi, samples


def fmt_ci(p, lo, hi, d: int = 3) -> str:
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
