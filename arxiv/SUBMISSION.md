# arXiv submission

## Files

| File | Use |
|---|---|
| `cosine_factuality_arxiv.tar.gz` | **Upload this.** LaTeX source, figures, ACL style files and `main.bbl` (arXiv does not run BibTeX). Comments are stripped because arXiv publishes the source. |
| `preview.pdf` | The package compiled on its own. The text is identical to `paper/main.pdf` (25 pages). |
| `abstract.txt` | Plain-text abstract for the form (1,344 characters; the limit is 1,920). |
| `src/` | Unpacked contents of the archive. |

Rebuild after any change to the paper:

```bash
python src/make_arxiv.py
```

## Form fields

* **Title:** When Is Cosine Similarity Enough? Counterfactual Robustness and Cost-Aware Evaluation of LLM Factuality
* **Authors:** Aryan Golbaghi
* **Abstract:** paste `abstract.txt`.
* **Comments:** `25 pages, 5 figures, 19 tables`
* **Primary category:** cs.CL (Computation and Language)
* **Cross-lists (optional):** cs.LG, cs.IR
* **License:** your choice. CC BY 4.0 is the most permissive; the arXiv default licence lets arXiv distribute the paper without granting reuse rights.
* **Journal reference / DOI:** leave empty (not published elsewhere).

## Before you press submit

1. **Code link.** The abstract, the setup and the conclusion link to
   https://github.com/youngaryan/cosine-vs-llm-factuality. Check the repository is public before submitting.
2. **Endorsement.** A first-time submitter to cs.CL may need an endorsement from an existing arXiv author;
   arXiv says so during submission if it applies.
3. **Compiler.** The package was test-compiled with Tectonic (XeTeX engine). arXiv uses pdfLaTeX. The source uses
   only standard packages (ACL style, times, T1/utf8, amsmath, booktabs, graphicx, makecell) and PDF figures, so
   no difference is expected. Still, check arXiv's generated PDF on the preview page before confirming.
4. **Dataset licences.** Nothing from the datasets is in the upload, apart from short quoted examples in
   Appendix Table 19 (from RAGTruth, MIT licence).
