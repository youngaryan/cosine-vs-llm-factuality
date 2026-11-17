"""Rule-based single-fact counterfactuals on human-labelled faithful RAGTruth responses.

For each base response (label 0) one sentence is edited so that exactly one fact changes.
Every edit must pass deterministic validity filters; nothing here is human-annotated.

  python src/build_counterfactuals.py --split train --limit 150   # rule development only
  python src/build_counterfactuals.py --split test                # final set
  python src/build_counterfactuals.py --split test --paraphrase   # add LLM paraphrases (Ollama)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import urllib.request
import zlib

import numpy as np
import spacy

from common import (CFG, PROC, RESULTS, SEED, content_tokens, neg_count, norm_num, numbers, proper_tokens,
                    read_jsonl, sentences, tokens, write_jsonl)

NLP = None
MONTHS = "January February March April May June July August September October November December".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
HEDGE = re.compile(r"\b(may|might|could|possibly|perhaps|unclear|unknown|not mentioned|not specified|"
                   r"no information|does not (?:mention|provide|specify)|likely|reportedly|allegedly)\b", re.I)
ENT_LABELS = {"PERSON", "ORG", "GPE"}
SYMMETRIC = {"meet", "marry", "join", "tie", "play", "face", "partner", "merge", "share", "resemble",
             "border", "date", "debate", "fight", "collaborate", "compete", "versus", "vs", "like", "equal"}

DIRECTION = [  # (a, b) swapped both ways
    ("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"),
    ("increasing", "decreasing"), ("rise", "fall"), ("rose", "fell"), ("rises", "falls"),
    ("rising", "falling"), ("higher", "lower"), ("highest", "lowest"), ("more than", "less than"),
    ("majority", "minority"), ("won", "lost"), ("win", "lose"), ("wins", "loses"), ("victory", "defeat"),
    ("positive", "negative"), ("accepted", "rejected"), ("approved", "rejected"), ("supported", "opposed"),
    ("supports", "opposes"), ("agreed", "refused"), ("success", "failure"), ("successful", "unsuccessful"),
    ("legal", "illegal"), ("guilty", "innocent"), ("largest", "smallest"), ("above", "below"),
    ("maximum", "minimum"), ("improved", "worsened"), ("allowed", "banned"), ("best", "worst"),
    ("gained", "lost"), ("open", "closed"), ("cheap", "expensive"), ("north", "south"), ("east", "west"),
    ("northern", "southern"), ("eastern", "western"), ("older", "younger"), ("oldest", "youngest"),
    ("male", "female"), ("alive", "dead"), ("safe", "dangerous"), ("strong", "weak"),
    ("praised", "criticized"), ("upheld", "overturned"), ("convicted", "acquitted"), ("arrived", "departed"),
]
TEMPORAL = [("before", "after"), ("earlier", "later"), ("first", "last"), ("prior to", "following"),
            ("began", "ended"), ("start", "end"), ("started", "ended"), ("early", "late")]
CAUSAL = [  # one-directional rewrites that reverse or cancel the stated causal relation
    (r"\bbecause of\b", "despite"), (r"\bbecause\b", "although"), (r"\bdue to\b", "despite"),
    (r"\bled to\b", "resulted from"), (r"\bleads to\b", "results from"), (r"\bleading to\b", "resulting from"),
    (r"\bresulted in\b", "resulted from"), (r"\bresults in\b", "results from"),
    (r"\bresulting in\b", "resulting from"), (r"\bas a result of\b", "in spite of"),
    (r"\bthanks to\b", "despite"), (r"\bcaused by\b", "the cause of"), (r"\bcausing\b", "caused by"),
    (r"\bcontributed to\b", "resulted from"), (r"\bcontributing to\b", "resulting from"),
    (r"\bprevented\b", "caused"), (r"\bprevents\b", "causes"), (r"\bprevent\b", "cause"),
    (r"\bin response to\b", "regardless of"), (r"\bas a result\b", "nevertheless"),
    (r"\btherefore\b", "nevertheless"), (r"\bprompting\b", "prompted by"), (r"\bdriven by\b", "the driver of"),
]
STOPWORDS_CAP = {"based", "according", "however", "overall", "additionally", "passage", "passages"}
NAME = re.compile(r"[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*)*")


def nlp():
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"] if False else [])
    return NLP


def keep_case(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() and not src.isupper() else dst.upper() if src.isupper() and len(src) > 1 else dst


def in_text(needle: str, hay_lower: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(needle.lower()) + r"(?![a-z0-9])", hay_lower) is not None


# ------------------------------------------------------------------ edit generators
def ed_number(sent, ev, ev_low, rng, ctx):
    ev_nums = set(numbers(ev))
    cands = []
    for m in re.finditer(r"(?<![\w:/\-(.])(\$?)(\d+(?:,\d{3})*)(\.\d+)?(%?)(?![\w:/\-)]|\.\d|,\d)", sent):
        raw = m.group(2) + (m.group(3) or "")
        n = norm_num(raw)
        try:
            val = float(n)
        except ValueError:
            continue
        if n not in ev_nums or (1800 <= val <= 2100 and not m.group(3) and "," not in raw) or val < 2:
            continue
        cands.append((m, val))
    rng.shuffle(cands)
    for m, val in cands:
        is_int = m.group(3) is None
        opts = [val + 1, val - 1, val * 2, val + 10, val * 10, val + 3, round(val * 1.5), val + 7] if is_int else \
               [val + 1, val + 0.5, val * 2, val + 0.3, val + 2]
        rng.shuffle(opts)
        for o in opts:
            if o <= 0 or o == val:
                continue
            if is_int:
                new = f"{int(o):,}" if "," in m.group(2) else str(int(o))
            else:
                new = f"{o:.{len(m.group(3)) - 1}f}"
            if norm_num(new) in ev_nums or (m.group(4) and o > 100):
                continue
            s2 = sent[: m.start(2)] + new + sent[m.end(3) if m.group(3) else m.end(2):]
            return s2, dict(old=raw, new=new)
    return None


def ed_date(sent, ev, ev_low, rng, ctx):
    cands = []
    for m in re.finditer(r"\b(1[89]\d\d|20\d\d)\b(?!,\d)", sent):
        if in_text(m.group(), ev_low):
            ks = [k for k in (-9, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 9) if not in_text(str(int(m.group()) + k), ev_low)]
            if ks:
                cands.append((m.start(), m.end(), str(int(m.group()) + int(rng.choice(ks)))))
    for names in (MONTHS, DAYS):
        for nm in names:
            for m in re.finditer(r"\b" + nm + r"\b", sent):
                if in_text(nm, ev_low) and not (nm == "May" and not re.match(r"\s+\d", sent[m.end():])):
                    alt = [x for x in names if x != nm and not in_text(x, ev_low)]
                    if alt:
                        cands.append((m.start(), m.end(), str(rng.choice(alt))))
    if not HEDGE.search(sent):
        for a, b in TEMPORAL:
            for x, y in ((a, b), (b, a)):
                for m in re.finditer(r"\b" + x + r"\b", sent, flags=re.I):
                    if in_text(x, ev_low):
                        cands.append((m.start(), m.end(), keep_case(m.group(), y)))
    if not cands:
        return None
    s, e, new = cands[int(rng.integers(len(cands)))]
    return sent[:s] + new + sent[e:], dict(old=sent[s:e], new=new)


def ed_direction(sent, ev, ev_low, rng, ctx):
    if HEDGE.search(sent):
        return None
    cands = []
    for a, b in DIRECTION:
        for x, y in ((a, b), (b, a)):
            for m in re.finditer(r"\b" + x + r"\b", sent, flags=re.I):
                if in_text(x, ev_low) and not in_text(y, ev_low):
                    cands.append((m.start(), m.end(), keep_case(m.group(), y)))
    if not cands:
        return None
    s, e, new = cands[int(rng.integers(len(cands)))]
    return sent[:s] + new + sent[e:], dict(old=sent[s:e], new=new)


def ed_causal(sent, ev, ev_low, rng, ctx):
    if HEDGE.search(sent):
        return None
    cands = []
    for pat, rep in CAUSAL:
        for m in re.finditer(pat, sent, flags=re.I):
            if m.start() > 0:
                cands.append((m.start(), m.end(), rep))
    if not cands:
        return None
    s, e, new = cands[int(rng.integers(len(cands)))]
    return sent[:s] + new + sent[e:], dict(old=sent[s:e], new=new)


def ed_negation(sent, ev, ev_low, rng, ctx):
    if HEDGE.search(sent) or sent.rstrip().endswith("?"):
        return None
    doc = nlp()(sent)
    roots = [t for t in doc if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")]
    if len(list(doc.sents)) != 1 or not roots:
        return None
    root = roots[0]
    if not any(c.dep_ in ("nsubj", "nsubjpass", "expl") and c.i < root.i and c.pos_ in ("NOUN", "PROPN", "PRON")
               for c in root.children):
        return None
    negs = [c for c in root.children if c.dep_ == "neg"]
    if negs:                                             # remove an existing negation
        t = negs[0]
        if t.text.lower() == "not":
            s2 = sent[: t.idx].rstrip() + " " + sent[t.idx + len(t.text):].lstrip()
            return re.sub(r"\s+", " ", s2), dict(old="not", new="")
        if t.text.lower() == "n't" and t.i > 0:
            aux = doc[t.i - 1]
            base = {"ca": "can", "wo": "will", "sha": "shall"}.get(aux.text.lower(), aux.text)
            return sent[: aux.idx] + keep_case(aux.text, base) + sent[t.idx + len(t.text):], dict(old=aux.text + "n't", new=base)
        return None
    if neg_count(sent):
        return None
    auxes = [c for c in root.children if c.dep_ in ("aux", "auxpass") and c.i < root.i]
    if auxes:
        a = auxes[0]
        if a.text.lower() in ("to", "may", "might", "could", "would", "should"):
            return None
        ins = a.idx + len(a.text)
        if a.text.lower() == "can":
            return sent[: a.idx] + keep_case(a.text, "cannot") + sent[ins:], dict(old="can", new="cannot")
        return sent[:ins] + " not" + sent[ins:], dict(old=a.text, new=a.text + " not")
    if root.lemma_ == "be" and root.pos_ == "AUX":
        ins = root.idx + len(root.text)
        return sent[:ins] + " not" + sent[ins:], dict(old=root.text, new=root.text + " not")
    if root.tag_ in ("VBD", "VBZ", "VBP") and root.lemma_.isalpha():
        do = {"VBD": "did", "VBZ": "does", "VBP": "do"}[root.tag_]
        new = f"{do} not {root.lemma_.lower()}"
        return sent[: root.idx] + new + sent[root.idx + len(root.text):], dict(old=root.text, new=new)
    return None


def _clean_ents(doc):
    """Entities that are whole names: alphabetic, capitalised, not glued to a neighbouring capitalised token."""
    out = []
    for e in doc.ents:
        if e.label_ not in ENT_LABELS or len(e.text) < 3 or not NAME.fullmatch(e.text):
            continue
        if e.text.split()[0] in ("The", "A", "An") or e.text.lower() in STOPWORDS_CAP:
            continue
        prev = doc[e.start - 1] if e.start > 0 else None
        nxt = doc[e.end] if e.end < len(doc) else None
        if prev is not None and (prev.text in ("&", "-", "#") or (prev.text[:1].isupper() and prev.i != e.sent.start)):
            continue
        if prev is not None and prev.i == e.sent.start and prev.text[:1].isupper() and prev.ent_type_:
            continue
        if nxt is not None and (nxt.text[:1].isupper() or nxt.text in ("'s", "’s", "&", "-")):
            continue
        out.append(e)
    return out


def _ents(sent):
    return _clean_ents(nlp()(sent))


def ed_entity(sent, ev, ev_low, rng, ctx):
    ents = [e for e in _ents(sent) if in_text(e.text, ev_low)]
    rng.shuffle(ents)
    for e in ents:
        ntok = len(e.text.split())
        if e.label_ != "PERSON" and e.text not in ctx["pool"][e.label_]:
            continue                                     # only swap entity strings the tagger found repeatedly
        pool = [p for p in ctx["pool"][e.label_] if len(p.split()) == ntok and not in_text(p, ev_low)
                and not set(tokens(p)) & set(tokens(e.text)) and all(not in_text(w, ev_low) for w in p.split() if len(w) > 3)]
        if not pool:
            continue
        new = pool[int(rng.integers(len(pool)))]
        s2 = sent[: e.start_char] + new + sent[e.end_char:]
        return s2, dict(old=e.text, new=new, label=e.label_)
    return None


def ed_role_swap(sent, ev, ev_low, rng, ctx):
    doc = nlp()(sent)
    ents = [e for e in _clean_ents(doc) if e.label_ == "PERSON"]
    pairs = []
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            if a.label_ != b.label_ or a.text.lower() == b.text.lower() or a.text in b.text or b.text in a.text:
                continue
            if sent.count(a.text) != 1 or sent.count(b.text) != 1:
                continue
            between = doc[a.end: b.start]
            if not any(t.pos_ in ("VERB", "AUX") for t in between):
                continue
            if any(t.lemma_.lower() in SYMMETRIC for t in doc) or b.root.dep_ == "conj" or a.root.dep_ == "conj":
                continue
            if not (in_text(a.text, ev_low) and in_text(b.text, ev_low)):
                continue
            pairs.append((a, b))
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(len(pairs)))]
    s2 = sent[: a.start_char] + b.text + sent[a.end_char: b.start_char] + a.text + sent[b.end_char:]
    return s2, dict(old=f"{a.text} | {b.text}", new=f"{b.text} | {a.text}", label=a.label_)


EDITORS = dict(negation=ed_negation, number=ed_number, date=ed_date, entity=ed_entity,
               direction=ed_direction, causal=ed_causal, role_swap=ed_role_swap)


def grounded(sent, ev_tok) -> float:
    ct = content_tokens(sent)
    return sum(t in ev_tok for t in ct) / len(ct) if ct else 0.0


def build(split: str, limit: int | None):
    rng = np.random.default_rng(SEED + 7)
    ex = [e for e in read_jsonl(PROC / "examples.jsonl")
          if e["dataset"] == "ragtruth" and e["split"] == split and e["label"] == 0]
    ex = [e for e in ex if 2 <= len(sentences(e["answer"])) <= 14]
    order = rng.permutation(len(ex))
    ex = [ex[i] for i in order]
    # entity pool and addition pool come from the same split, other groups
    pool = collections.defaultdict(collections.Counter)
    add_pool = collections.defaultdict(list)
    for e in ex[:700]:
        for s in sentences(e["answer"])[:6]:
            for en in _ents(s):
                pool[en.label_][en.text] += 1
            if 8 <= len(s.split()) <= 40 and (numbers(s, words=False) or proper_tokens(s)) and not s.endswith("?"):
                add_pool[e["domain"]].append((e["group"], s))
    ctx = dict(pool={k: sorted(t for t, c in v.items() if c >= 3) for k, v in pool.items()})
    for k in ENT_LABELS:
        ctx["pool"].setdefault(k, [])
    ctx["pool"]["PERSON"] = sorted(set(ctx["pool"]["PERSON"]) | {t for t in pool["PERSON"] if len(t.split()) == 2})
    ctx["pool"]["PERSON"] += sorted({t.split()[-1] for t in ctx["pool"]["PERSON"] if len(t.split()) == 2}
                                    - set(ctx["pool"]["PERSON"]))
    per_domain_cap = (limit // 3) if limit else 10**9
    taken = collections.Counter()
    bases, edits, rejects = [], [], collections.Counter()
    for e in ex:
        if taken[e["domain"]] >= per_domain_cap:
            continue
        ev, ev_low = e["evidence"], e["evidence"].lower()
        ev_tok = set(tokens(ev))
        sents = sentences(e["answer"])
        usable = [i for i, s in enumerate(sents)
                  if 6 <= len(s.split()) <= 60 and e["answer"].count(s) == 1 and grounded(s, ev_tok) >= 0.5]
        if not usable:
            rejects["no_grounded_sentence"] += 1
            continue
        found = collections.defaultdict(dict)          # sent idx -> type -> (edited sentence, info)
        for i in usable:
            for typ, fn in EDITORS.items():
                r = fn(sents[i], ev, ev_low, np.random.default_rng(zlib.crc32(f"{SEED}|{e['id']}|{i}|{typ}".encode())), ctx)
                if r is None:
                    continue
                s2, info = r
                s2 = s2.strip()
                if s2 == sents[i] or len(sentences(s2)) != len(sentences(sents[i])):
                    rejects[f"{typ}_identity_or_split"] += 1
                    continue
                if in_text(s2, ev_low):
                    rejects[f"{typ}_edited_sentence_verbatim_in_evidence"] += 1
                    continue
                found[i][typ] = (s2, info)
        if not found:
            rejects["no_applicable_edit"] += 1
            continue
        k_star = max(found, key=lambda i: (len(found[i]), -i))
        chosen = {}
        for typ in EDITORS:
            if typ in found[k_star]:
                chosen[typ] = (k_star, *found[k_star][typ])
            else:
                alt = [i for i in found if typ in found[i]]
                if alt:
                    i = alt[int(rng.integers(len(alt)))]
                    chosen[typ] = (i, *found[i][typ])
        base_id = e["id"]
        common = dict(dataset="counterfactual", domain=e["domain"], target="faithfulness", group=e["group"],
                      split=split, question=e["question"], evidence=ev, reference=[], generator=e["generator"])
        bases.append(dict(common, id=f"cf-{base_id}-orig", label_origin="human", answer=e["answer"], label=0,
                          meta=dict(base=base_id, type="original", k_star=k_star, target_sentence=sents[k_star],
                                    n_sentences=len(sents))))
        for typ, (i, s2, info) in chosen.items():
            edits.append(dict(common, id=f"cf-{base_id}-{typ}", label_origin="rule", label=1,
                              answer=e["answer"].replace(sents[i], s2, 1),
                              meta=dict(base=base_id, type=typ, sent_idx=i, matched_locus=bool(i == k_star),
                                        original_sentence=sents[i], edited_sentence=s2, edit=info,
                                        n_sentences=len(sents))))
        # unsupported addition: a sentence about a different source, appended
        cand = [s for g, s in add_pool[e["domain"]] if g != e["group"]]
        rng.shuffle(cand)
        for s in cand[:40]:
            pt = set(proper_tokens(s)) | set(numbers(s, words=False))
            if grounded(s, ev_tok) < 0.5 and pt and not any(in_text(p, ev_low) for p in pt):
                edits.append(dict(common, id=f"cf-{base_id}-addition", label_origin="rule", label=1,
                                  answer=e["answer"].rstrip() + " " + s,
                                  meta=dict(base=base_id, type="addition", sent_idx=len(sents), matched_locus=False,
                                            original_sentence="", edited_sentence=s, edit=dict(old="", new=s),
                                            n_sentences=len(sents) + 1)))
                break
        taken[e["domain"]] += 1
    return bases, edits, rejects


PARA_PROMPT = (
    "Rewrite the sentence below so that it has exactly the same meaning but different wording and sentence "
    "structure. Keep every name, number, date and quantity unchanged. Do not add or remove any information. "
    "Output only the rewritten sentence.\n\nSentence: {s}\nRewritten:"
)


def paraphrase(bases):
    model = CFG["counterfactual"]["paraphrase_model"]
    out, rej = [], collections.Counter()
    cap = CFG["counterfactual"]["paraphrase_bases"] // 3
    seen = collections.Counter()
    todo = []
    for b in bases:                                   # first `cap` bases per domain, in file order
        if seen[b["domain"]] < cap:
            seen[b["domain"]] += 1
            todo.append(b)
    bases = todo
    side = PROC / "paraphrase_attempts.jsonl"            # resume-safe log of every attempt
    prev = {r["base_id"]: r for r in read_jsonl(side)} if side.exists() else {}
    log = open(side, "a", encoding="utf8")
    for n, b in enumerate(bases):
        if b["id"] in prev:
            if prev[b["id"]]["row"]:
                out.append(prev[b["id"]]["row"])
            else:
                rej[prev[b["id"]]["why"]] += 1
            continue
        s = b["meta"]["target_sentence"]
        body = dict(model=model, prompt=PARA_PROMPT.format(s=s), stream=False, keep_alive="10m",
                    options=dict(temperature=0, seed=SEED, num_predict=120, num_ctx=1024))
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        p = json.load(urllib.request.urlopen(req, timeout=600))["response"].strip().strip('"').split("\n")[0].strip()
        why = None
        if not p or p == s:
            why = "empty_or_identical"
        elif sorted(numbers(p)) != sorted(numbers(s)):
            why = "numbers_changed"
        elif not set(proper_tokens(s)) <= set(tokens(p)):
            why = "entity_dropped"
        elif (neg_count(p) > 0) != (neg_count(s) > 0):
            why = "negation_parity"
        elif not 0.6 <= len(p.split()) / len(s.split()) <= 1.7 or len(sentences(p)) != 1:
            why = "length_or_split"
        elif len(set(tokens(p)) & set(tokens(s))) / len(set(tokens(p)) | set(tokens(s))) > 0.9:
            why = "too_similar"
        if why:
            rej[why] += 1
            log.write(json.dumps(dict(base_id=b["id"], why=why, text=p, row=None)) + chr(10))
            log.flush()
            continue
        m = dict(b["meta"], type="paraphrase", sent_idx=b["meta"]["k_star"], matched_locus=True,
                 original_sentence=s, edited_sentence=p, edit=dict(model=model))
        out.append(dict(b, id=b["id"].replace("-orig", "-paraphrase"), label_origin="llm_paraphrase_rule_validated",
                        answer=b["answer"].replace(s, p, 1), label=0, meta=m))
        log.write(json.dumps(dict(base_id=b["id"], why=None, text=p, row=out[-1])) + chr(10))
        log.flush()
        if n % 25 == 0:
            print("paraphrase", n, len(out), dict(rej), flush=True)
    return out, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--paraphrase", action="store_true")
    a = ap.parse_args()
    tag = "" if a.split == "test" else f"_{a.split}"
    path = PROC / f"counterfactuals{tag}.jsonl"
    if a.paraphrase:
        rows = read_jsonl(path)
        bases = [r for r in rows if r["meta"]["type"] == "original"]
        para, rej = paraphrase(bases)
        rows = [r for r in rows if r["meta"]["type"] != "paraphrase"] + para
        write_jsonl(path, rows)
        stats = json.loads((RESULTS / f"cf_stats{tag}.json").read_text())
        stats["paraphrase"] = dict(model=CFG["counterfactual"]["paraphrase_model"], eligible_bases=len(bases), attempted=len(para) + sum(rej.values()),
                                   kept=len(para), rejected=dict(rej))
        (RESULTS / f"cf_stats{tag}.json").write_text(json.dumps(stats, indent=1))
        print(stats["paraphrase"])
        return
    bases, edits, rejects = build(a.split, a.limit)
    write_jsonl(path, bases + edits)
    cnt = collections.Counter((e["meta"]["type"], e["domain"]) for e in edits)
    stats = dict(bases=len(bases), pairs=len(edits),
                 by_type={t: sum(v for (tt, _), v in cnt.items() if tt == t) for t in sorted({t for t, _ in cnt})},
                 by_type_domain={f"{t}|{d}": v for (t, d), v in sorted(cnt.items())},
                 matched_locus=sum(e["meta"]["matched_locus"] for e in edits), rejections=dict(rejects))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"cf_stats{tag}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    if a.split == "test":   # audit sheet for human reviewers (verdict columns intentionally empty)
        rng = np.random.default_rng(SEED + 11)
        pick = rng.permutation(len(edits))[:160]
        with open(RESULTS / "cf_audit_sheet.csv", "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "original_sentence", "edited_sentence", "evidence_excerpt",
                        "human_edit_is_unsupported(Y/N)", "human_edit_is_fluent(Y/N)", "notes"])
            for i in pick:
                e = edits[int(i)]
                w.writerow([e["id"], e["meta"]["type"], e["meta"]["original_sentence"], e["meta"]["edited_sentence"],
                            e["evidence"][:1500], "", "", ""])


if __name__ == "__main__":
    main()
