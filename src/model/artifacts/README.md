# Model artifacts (runtime assets)

These files are **not committed to git**. They are declared as `assets` in
[`schema/model.json`](../../../schema/model.json) and provided to the tool at
run time by the LMDS platform:

1. An admin uploads a file for each declared asset path under the version's
   **Manage → Versions → Assets** page (the upload UI only accepts paths that
   are declared in the schema).
2. When a run starts, the runner downloads each asset from S3 to its declared
   path under the container WORKDIR (`./artifacts/…`), before the handler runs.

The model reads them from `./artifacts/` (or the package's `artifacts/` dir for
a local source checkout; `MODEL_ARTIFACTS_DIR` overrides both). It fails fast if
the required file is missing.

| Asset path (declared in model.json) | ~Size | What it is |
|--------------------------------------|------|------------|
| `artifacts/model.pk` | 15 MB | Random-forest regressor (100 trees, 133 features), the original `pickle_jar/model.pk` — upload unchanged |

`model.pk` is a scikit-learn **0.22.2** pickle. It is read by
`src/model/forest.py`, which rebuilds the trees without importing
scikit-learn, so no old scikit-learn pin is needed (see the main README).

`featurizer.pk` (the original `pickle_jar/featurizer.pk`, 3 MB) is **not** a
runtime asset. The small amount of fitted state the tool needs from it
(feature order, RobustScaler parameters, training means) was exported once to
`src/model/featurizer_state.json`, which ships in the package. Keep a copy here
only if you need to re-run `training/export_featurizer_state.py`.

**Local development:** place `model.pk` in this directory
(`src/model/artifacts/`); running from source resolves it here by default.
