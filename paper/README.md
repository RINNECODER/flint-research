# Paper

`flint-report.pdf` is the technical report on this repo's work. Source: `main.tex` and `references.bib`.
Figures are built from `results/` by `figs/make_figs.py`.

## Build

From the repo root:

```bash
pip install matplotlib                 # only needed to rebuild the figures
python paper/figs/make_figs.py         # -> paper/figs/speed_path.pdf, paper/figs/pruning.pdf
cd paper
tectonic main.tex                      # or: latexmk -pdf main.tex   (brew install tectonic)
mv main.pdf flint-report.pdf
```

Tectonic fetches the LaTeX packages it needs on first run. Delete build files (`*.aux`, `*.log`, `*.bbl`, `*.blg`)
before committing.
