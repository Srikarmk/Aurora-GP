# Paper source

Build with any LaTeX distribution:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`figs/*.pdf` are generated from stored results by `../src/make_figures.py`; rerun that
after regenerating `../results/`. Build artefacts are gitignored; the source and the
figures are tracked so the PDF is reproducible from this directory alone.

For arXiv, upload `main.tex`, `refs.bib` and `figs/`. arXiv runs BibTeX itself, but it
is safer to also include the generated `main.bbl`.
