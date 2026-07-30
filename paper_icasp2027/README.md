# ICASSP Paper Draft (2026 Paper Kit)

This draft uses the official ICASSP 2026 Paper Kit, rather than the rejected
ICASSP 2025 template:

- `spconf.sty` for the official two-column layout and typography
- `IEEEbib.bst` for the official bibliography style
- letter-size paper and 9-point text
- compact figures and tables within the kit's print area

The downloaded kit archive and the exact style files are preserved in
`official_icassp2026/`. The authoritative instructions are the [ICASSP 2026
Paper Kit](https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php).
They allow four pages of technical content and an optional fifth page
containing only references, funding acknowledgements, and a Compliance with
Ethical Standards statement. This build has exactly that five-page split.

The official call is [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/). It lists `Machine Learning and Generative AI` and `Speech and Language Processing` in the technical scope. The full-paper deadline listed there is September 16, 2026.

## Build

Tectonic is the lightweight TeX engine used for the checked build. From this
directory, run:

```bash
tectonic -X compile main.tex
```

The checked build produces a 5-page letter-size PDF: pages 1--4 are technical
content and page 5 contains only the required ancillary material. Tectonic
runs BibTeX automatically. Replace the anonymous author line in `main.tex`
with the final author metadata before submission.

The method figure is available as an editable `figures/cqag_pipeline.drawio`, clean PNG, embedded editable PNG, and SVG.
