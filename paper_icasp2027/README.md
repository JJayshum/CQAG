# ICASSP 2027 Paper Draft

This draft follows the IEEE conference manuscript format used by the ICASSP author kit:

- `IEEEtran` conference class
- two-column layout
- letter paper
- numbered IEEE citations and bibliography
- figure and table captions in the IEEE style

The official call is [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/). It lists `Machine Learning and Generative AI` and `Speech and Language Processing` in the technical scope. The full-paper deadline listed there is September 16, 2026.

## Build

Install the IEEE author kit and a LaTeX distribution, then run:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The repository environment used to prepare this draft does not currently have `pdflatex`, `latexmk`, or another TeX engine, so PDF page-count and visual overflow checks must be run after installing one. Replace the anonymous author block in `main.tex` before submission.

The method figure is available as an editable `figures/cqag_pipeline.drawio`, clean PNG, embedded editable PNG, and SVG.
