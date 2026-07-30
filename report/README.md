# Report sources

The academic briefing is available in two forms:

- `main.tex`: single-column, four-part academic LaTeX source;

The bibliography is maintained in `references.bib`. Figures are generated or
copied into `figures/`. Each experiment shows its own training-versus-validation
curve and a 64-sample prior-predictive grid; the final section combines the
three sample grids side by side. Regenerate the source-backed architecture,
loss, schedule, and comparison figures from the saved artifacts:

```bash
uv run --with reportlab==4.4.9 python report/scripts/render_report_figures.py
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

All reported IWAE, DReG, and progressive-DReG results are backed by the saved
seed-236 artifacts and the shared evaluation protocol. KID uses the common
`outputs/classifier/checkpoints/epoch-04.ckpt` feature extractor, 10,000 real
and generated samples, 50 subsets of 1,000, and seed 236. The accelerated
result remains single-seed evidence until the replication gate in
`docs/training-acceleration.md` is completed.
