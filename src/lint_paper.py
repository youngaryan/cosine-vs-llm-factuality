"""Static checks of the manuscript (no LaTeX compiler needed): undefined macros, dangling references and
citations, missing figure files, leftover TODO markers, unbalanced braces, and an approximate word count."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
KNOWN = set("""documentclass usepackage newcommand title author date begin end maketitle input bibliography appendix
section subsection paragraph label ref cite citep citet emph textbf texttt textsc small centering toprule midrule bottomrule
caption includegraphics textwidth columnwidth resizebox setlength tabcolsep itemsep item noindent quad le ge mid wedge
mathrm mathbb mathcal in tau alpha beta pi ast hat rightarrow ldots footnotesize scriptsize vspace hspace raggedright
ttfamily multicolumn cline yes checkmark makecell textquotedbl textbackslash textasciitilde textasciicircum
le leq geq times cdot sim approx neq infty sum frac log Pr top bottom bm boldsymbol url href thanks
emph textit underline sqrt min max arg exp prime dots vdots cdots mathbf text textrm rm bf it em
sffamily normalsize large Large LARGE huge Huge tiny smallskip medskip bigskip newline linewidth hline
displaystyle left right big Big bigl bigr Bigl Bigr cup cap subset subseteq setminus emptyset forall exists
hfill vfill clearpage newpage pagebreak linebreak nolinebreak label pageref eqref autoref nameref
tabularnewline arraystretch renewcommand def let relax protect footnote footnotetext
par parskip hangindent hangafter ifx fi undefined pdfoutput XeTeXrevision nolinkurl mathit ll gg varepsilon epsilon delta Delta sigma lambda mu rho theta phi psi omega gamma kappa eta""".split())


def main():
    files = [PAPER / "main.tex"] + sorted((PAPER / "sections").glob("*.tex"))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "generated" / "numbers.tex").read_text(encoding="utf8")))
    defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "main.tex").read_text(encoding="utf8")))
    gen_text = "".join(p.read_text(encoding="utf8") for p in (PAPER / "generated").glob("tab_*.tex"))
    text = {p.name: p.read_text(encoding="utf8") for p in files}
    allt = "\n".join(text.values()) + gen_text
    bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text(encoding="utf8")))
    labels = set(re.findall(r"\\label\{([^}]+)\}", allt))
    problems = []
    for name, t in text.items():
        body = re.sub(r"(?<!\\)%.*", "", t)
        for m in sorted(set(re.findall(r"(?<!\\)\\([A-Za-z]+)", body))):
            if m not in defined and m not in KNOWN and m not in ("acl", "times", "latexsym", "fontenc", "inputenc", "microtype",
                                                                  "amsmath", "amssymb", "booktabs", "graphicx", "tabular",
                                                                  "table", "figure", "itemize", "enumerate", "quote", "document",
                                                                  "center", "abstract", "setlength"):
                problems.append(f"{name}: undefined or unknown macro \\{m}")
        for r in re.findall(r"\\ref\{([^}]+)\}", body):
            if r not in labels:
                problems.append(f"{name}: dangling \\ref{{{r}}}")
        for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", body):
            for c in group.split(","):
                if c.strip() not in bib:
                    problems.append(f"{name}: unknown citation {c.strip()}")
        for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body):
            if not (PAPER / g).exists():
                problems.append(f"{name}: missing figure {g}")
        for g in re.findall(r"\\input\{([^}]+)\}", body):
            if not (PAPER / (g if g.endswith(".tex") else g + ".tex")).exists():
                problems.append(f"{name}: missing input {g}")
        if body.count("{") != body.count("}"):
            problems.append(f"{name}: unbalanced braces ({body.count('{')} vs {body.count('}')})")
        for m in re.findall(r"%\s*([A-Z][A-Z0-9-]{5,})", t):
            problems.append(f"{name}: placeholder marker {m}")
    cited = set(c.strip() for g in re.findall(r"\\cite[pt]?\{([^}]+)\}", allt) for c in g.split(","))
    print("uncited bib entries:", sorted(bib - cited))
    prose = "\n".join(t for n, t in text.items() if n not in ("main.tex", "appendix.tex"))
    prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", prose, flags=re.S)
    prose = re.sub(r"(?<!\\)%.*", "", prose)
    words = re.findall(r"[A-Za-z\\][A-Za-z\\-]*", re.sub(r"\\(label|ref|cite[pt]?|input)\{[^}]*\}", " ", prose))
    print("approx. words in main text (excl. tables/figures/appendix):", len(words))
    print("\n".join(problems) if problems else "no problems found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
