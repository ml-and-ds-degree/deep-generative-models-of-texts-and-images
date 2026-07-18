# Report sources

The academic briefing is available in two forms:

- `main.tex`: single-column, two-part academic LaTeX source;
- `briefing.md`: readable Markdown companion.

The bibliography is maintained in `references.bib`. Figures are generated or
copied into `figures/`. To regenerate the learning curve from the saved
Lightning metrics:

```bash
uv run --with reportlab==4.4.9 python report/scripts/plot_learning_curve.py
pdftoppm -png -singlefile -r 160 \
  report/figures/iwae_learning_curve.pdf \
  report/figures/iwae_learning_curve
```

To build the LaTeX paper on a machine with a standard TeX distribution:

```bash
cd report
latexmk -pdf main.tex
```

## Dev Container

For an isolated LaTeX environment, open this repository in VS Code and choose
**Dev Containers: Reopen in Container**. The container includes the project's
locked Python environment, `latexmk`, `pdftoppm`, and the TeX Live packages
required by `main.tex`; LaTeX Workshop writes generated files to
`report/build/`.

An equivalent manual build is:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source intentionally leaves the DReG results table as `TBD`. Fill it only
after running DReG under the same budget and validating its artifacts.
