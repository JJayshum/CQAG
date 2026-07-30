# ICASSP 2027 Paper Draft

This draft follows the IEEE conference manuscript format used by the ICASSP author kit:

- `IEEEtran` conference class
- two-column layout
- letter paper
- numbered IEEE citations and bibliography
- figure and table captions in the IEEE style

The official call is [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/). It lists `Machine Learning and Generative AI` and `Speech and Language Processing` in the technical scope. The full-paper deadline listed there is September 16, 2026.

## Build

Tectonic is the lightweight TeX engine used for the checked build. From this
directory, run:

```bash
tectonic -X compile main.tex
```

The checked build produces a 4-page letter-size PDF and runs BibTeX
automatically. The ICASSP author kit should still be used for the final
submission build if the conference releases a newer template. Replace the
anonymous author block in `main.tex` before submission.

The method figure is available as an editable `figures/cqag_pipeline.drawio`, clean PNG, embedded editable PNG, and SVG.
