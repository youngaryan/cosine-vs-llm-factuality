"""Build an arXiv source package from paper/.

arXiv compiles the source itself, with pdfLaTeX and without BibTeX, and only sees the files in the upload.
This script therefore:
  * copies main.tex, sections/, generated/, the ACL style files and the figures that are actually used
    into arxiv/src/, rewriting ../figures/ paths to figures/;
  * compiles paper/ once with Tectonic to produce main.bbl and ships it with the source;
  * strips LaTeX comments (arXiv publishes the source, comments included), keeping a bare '%' so that
    line-end spacing is unchanged;
  * test-compiles the package in isolation and writes arxiv/cosine_factuality_arxiv.tar.gz.

Usage: python src/make_arxiv.py [--tectonic PATH]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, FIGS, OUT = ROOT / "paper", ROOT / "figures", ROOT / "arxiv"
DEFAULT_TECTONIC = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "tectonic" / "tectonic.exe"


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        if re.match(r"^\s*%", line):
            continue                                   # full-line comment: drop the line
        out.append(re.sub(r"(?<!\\)%.*$", "%", line))  # trailing comment: keep '%' (suppresses the line-end space)
    return "\n".join(out) + "\n"


def plain_abstract() -> str:
    """Abstract as plain text for the arXiv form: number macros and \\CodeURL expanded, no LaTeX left."""
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\n", (PAPER / "generated" / "numbers.tex").read_text(encoding="utf8")))
    url = re.search(r"\\newcommand\{\\CodeURL\}\{\\url\{([^}]+)\}\}", (PAPER / "main.tex").read_text(encoding="utf8"))
    if url:
        macros["CodeURL"] = url.group(1)
    t = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf8")
    t = re.sub(r"\\(begin|end)\{abstract\}", "", t)
    t = re.sub(r"\\([A-Za-z]+)(\\ )?", lambda m: macros[m.group(1)] + (" " if m.group(2) else ""), t)
    t = re.sub(r"\s+", " ", t.replace("\\%", "%").replace("~", " ")).strip()
    assert "\\" not in t and "{" not in t, t
    assert len(t) <= 1920, f"abstract is {len(t)} characters; arXiv allows 1,920"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=str(DEFAULT_TECTONIC if DEFAULT_TECTONIC.exists() else "tectonic"))
    a = ap.parse_args()

    # 1. fresh bibliography from the current source
    subprocess.run([a.tectonic, "-X", "compile", "main.tex", "--keep-intermediates", "--keep-logs"], cwd=PAPER, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bbl = PAPER / "main.bbl"
    assert bbl.exists(), "main.bbl was not produced"

    src = OUT / "src"
    if src.exists():
        shutil.rmtree(src)
    (src / "figures").mkdir(parents=True)

    # 2. TeX sources, comments stripped, figure paths rewritten
    tex_files = [PAPER / "main.tex"] + sorted((PAPER / "sections").glob("*.tex")) + sorted((PAPER / "generated").glob("*.tex"))
    used_figs = set()
    for f in tex_files:
        t = f.read_text(encoding="utf8")
        used_figs |= set(re.findall(r"\.\./figures/([\w.-]+)", t))
        t = t.replace("../figures/", "figures/")
        t = strip_comments(t)
        if f.name == "main.tex":
            assert "\\pdfoutput=1" in "\n".join(t.splitlines()[:5]), "\\pdfoutput=1 must be within the first 5 lines"
        dst = src / f.relative_to(PAPER)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(t, encoding="utf8", newline="\n")
    for name in sorted(used_figs):
        shutil.copy2(FIGS / name, src / "figures" / name)
    for name in ["acl.sty", "acl_natbib.bst", "references.bib"]:
        shutil.copy2(PAPER / name, src / name)
    shutil.copy2(bbl, src / "main.bbl")

    # 3. self-check: no path may leave the package, and every input/figure must exist inside it
    alltex = "\n".join(p.read_text(encoding="utf8") for p in src.rglob("*.tex"))
    assert "../" not in alltex, "a path still points outside the package"
    for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", alltex):
        assert (src / g).exists(), f"missing figure {g}"
    for g in re.findall(r"\\input\{([^}]+)\}", alltex):
        assert (src / (g if g.endswith(".tex") else g + ".tex")).exists(), f"missing input {g}"

    # 4. test-compile the package on its own copy (Tectonic, like pdfLaTeX, reads main.bbl)
    test = OUT / "test_build"
    if test.exists():
        shutil.rmtree(test)
    shutil.copytree(src, test)
    r = subprocess.run([a.tectonic, "-X", "compile", "main.tex", "--keep-logs"], cwd=test, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("test compile failed:\n" + r.stderr[-3000:])
    log = (test / "main.log").read_text(encoding="utf8", errors="replace")
    bad = re.findall(r"(LaTeX Warning: (?:Reference|Citation)[^\n]*undefined[^\n]*)", log)
    if bad:
        sys.exit("undefined references or citations:\n" + "\n".join(bad))
    shutil.copy2(test / "main.pdf", OUT / "preview.pdf")

    # 5. archive (files at the top level, as arXiv expects)
    tgz = OUT / "cosine_factuality_arxiv.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(src).as_posix())
    shutil.rmtree(test)
    ab = plain_abstract()
    (OUT / "abstract.txt").write_text(ab + "\n", encoding="utf8")
    print(f"abstract.txt: {len(ab)} characters")
    n = sum(1 for p in src.rglob("*") if p.is_file())
    print(f"{n} files, {tgz.stat().st_size / 1024:.0f} KiB -> {tgz}")
    print(f"preview of the package build -> {OUT / 'preview.pdf'}")


if __name__ == "__main__":
    main()
