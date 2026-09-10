# ICASSP 2027 Paper

This paper uses the official ICASSP 2027 Paper Kit:

- `spconf.sty` for the official two-column layout and typography
- `IEEEbib.bst` for the official bibliography style
- letter-size paper and 9-point text
- compact figures and tables within the kit's print area

The checked-in `spconf.sty` and `IEEEbib.bst` were verified byte-for-byte
against the files served by the [ICASSP 2027 Paper
Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php). The kit allows
four pages of technical content and an optional fifth page containing only
references, funding acknowledgments, and a Compliance with Ethical Standards
statement. This build has exactly that five-page split.

The official call is [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/). It lists `Machine Learning and Generative AI` and `Speech and Language Processing` in the technical scope. The full-paper deadline listed there is September 16, 2026.

## Build

The checked PDF was built with pdfLaTeX and BibTeX. From this directory, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The checked build produces a 5-page letter-size PDF: pages 1--4 are technical
content and page 5 contains only the permitted ancillary material. The paper
is non-anonymous and identifies Haowen Chen as corresponding author. Submit a
copy named `shen.pdf`, and make the portal author names and order match the PDF
exactly.

The method figure is available as an editable `figures/cqag_pipeline.drawio`, clean PNG, embedded editable PNG, and SVG.
