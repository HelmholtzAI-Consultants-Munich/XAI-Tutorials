# marimo port of the LIME tutorial — notes

Port of `xai-for-random-forest/Gen-2-Tutorial_LIME.ipynb`
(HelmholtzAI-Consultants-Munich/XAI-Tutorials) into a reactive marimo notebook,
as a small teaching prototype.

## What was built

- **`Gen-2-Tutorial_LIME.py`** — reactive marimo version of the notebook, in this folder.
- The 5 matplotlib LIME plots are replaced with **Vega-Altair** charts (`mo.ui.altair_chart`).
- **Interactive controls in both the housing and wine sections:** instance index,
  `kernel_width`, `num_samples`, and **surrogate model (LinearRegression vs Ridge)**.
- All original narrative is kept: explanations, the Q1–Q3 answers, the Wine task,
  the conclusion, and the "LIME step by step" walkthrough.
- Outputs reproduce the original at default settings: housing R² 0.84 / 0.72;
  wine accuracy 100% / 97.22%; wine surrogate scores 0.55 / 0.68 / 0.75 for
  type_1 / type_2 / type_3.
- **The notebook itself keeps the original tutorial's text and code comments and
  carries no commentary about the conversion** — all marimo/transition specifics
  are documented here in NOTES.md, so the teaching material stays clean.

## Caching / isolation of data + model loading

Achieved via marimo's **dependency graph** rather than an explicit cache:

- The two `pickle.load` cells depend only on `MODELS_DIR` — never on a control —
  so marimo runs them exactly once. Moving a slider never reloads the model.
- Fine granularity: the LIME explainer rebuilds **only** when `kernel_width`
  changes (it's a constructor argument). Changing instance / num_samples /
  surrogate skips the explainer and re-runs only `explain_instance` + the chart.
- This is the idiomatic marimo approach. `mo.cache` / `mo.persistent_cache` are
  available if heavier caching is ever needed, but loading here is ~0.1 s, so
  isolation alone is enough.

## What works well

- **LIME is a great fit for reactive teaching.** Dragging `kernel_width`,
  `num_samples`, or the instance and watching the explanation shift makes the
  "local, sample-based surrogate" idea tangible — it turns the original's
  "feel free to play around with different values" line into something you do.
- The **surrogate dropdown (Linear vs Ridge)** surfaces the fidelity-vs-stability
  trade-off in seconds.
- Recomputation is **automatic and fine-grained** — you write normal Python and
  reference `slider.value`; no callbacks or manual wiring.
- The `.py` format is **diff-friendly** and runs as a plain module; HTML export
  works for a static snapshot.

## What was difficult / friction points

- **Single-definition rule.** The original reuses variable names (`X_train`,
  `explainer`, `explanation`, `inst_idx`, …) across the housing / wine / step
  sections. marimo requires each global to be defined once, so I renamed per
  section (`X_train_h` vs `X_train_w`, etc.). This is the main porting cost and
  it scales with notebook size.
- **Order-by-dependency, not top-to-bottom.** The original sets a global
  `np.random.seed` once at the top; that ordering isn't guaranteed in marimo, so
  randomness is pinned locally (`random_state=seed` everywhere, plus
  `np.random.default_rng` in the manual section).
- **UI `.value` must be read in a *different* cell** than the one that creates the
  widget, otherwise it isn't reactive — easy to trip on at first.
- **Per-interaction cost.** Each slider move re-runs LIME (~0.5–2 s for these
  small models). Fine here; for heavier models you'd lower `num_samples` or gate
  recompute behind a `mo.ui.run_button`. marimo cancels stale runs mid-drag,
  which helps.
- **Environment (unrelated to marimo).** The model pickles need pandas 3.x /
  Python 3.11 to unpickle (newer `StringDtype`); the system Python 3.9 can't.
- **One deliberate narrative change.** The three hardcoded wine instances became a
  single slider-driven explorer (less duplication, more reactive). The conclusion
  still discusses instances 4 / 1 / 101, with a tip to select them.

## Is marimo suitable for this tutorial?

**Yes — for an interactive teaching prototype it's arguably a better fit than
Jupyter.** The entire pedagogical point (how do explanations change with
neighborhood size, sample count, and surrogate choice?) is exactly what marimo's
reactivity is built for, and students avoid the out-of-order / stale-state
confusion that bites in Jupyter.

Trade-offs to weigh: the rename cost when porting existing notebooks; recompute
latency for larger models; and students must run it via `marimo edit` / `marimo
run` (a PDF or HTML handout won't be interactive). For a self-contained,
reproducible, hands-on lesson, I'd recommend it.

## How to run

```
# from the repository root:
cd xai-for-random-forest/marimo
.venv/bin/marimo edit Gen-2-Tutorial_LIME.py     # interactive editor — drag the sliders
.venv/bin/marimo run  Gen-2-Tutorial_LIME.py     # clean app view, no code shown
```

Dependencies are pinned in `requirements-marimo.txt` (Python 3.11 recommended).
The static HTML export shows the controls but is **not** interactive — the live
recomputation needs the running marimo engine.
