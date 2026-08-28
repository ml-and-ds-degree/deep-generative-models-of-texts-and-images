# MADE reproduction report

The report source and generated artifact live together in this directory.

From a LaTeX environment, build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The final PDF is written to `build/main.pdf`.

The figures are reproducible from the two Lightning runs and their sampled NPZ files:

```bash
uv run python make_figures.py \
  --baseline-metrics path/to/made/metrics.csv \
  --baseline-samples path/to/made_samples.npz \
  --improved-metrics path/to/pixelcnn/version_0/metrics.csv \
                     path/to/pixelcnn/version_1/metrics.csv \
  --improved-samples path/to/pixelcnn_samples.npz
```
