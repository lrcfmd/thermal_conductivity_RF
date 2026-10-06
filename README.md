# Thermal Conductivity RF

Random-forest prediction of **thermal conductivity** (κ, W m⁻¹ K⁻¹) from a
chemical composition, packaged for the LMDS (Local Model Deployment Service)
platform.

Each composition is featurised with the 132 Magpie elemental-property
statistics (matminer `ElementProperty`), plus a `computational_label` flag. The
features are robust-scaled with the training-time parameters and passed to a
random forest trained on ~2000 measured and computed conductivities.

This is a conversion of the original standalone scripts to the LMDS
model-template contract (`schema/model.json` + a single `process` handler).

## Quick Start

```bash
# Predict one composition:
scripts/run_instance.sh Bi2Te3      # {"thermal_conductivity": 1.9396, "imputed_features": 0}

# Batch-predict a CSV with a 'composition' column:
scripts/run_dataset.sh data/input.csv   # -> data/output/predictions.csv + predictions_hist.png
```

## Modes

| Mode | Response | Input | Output |
|------|----------|-------|--------|
| `instance` | immediate | `composition` (string, e.g. `"Bi2Te3"`, `"Ag9 Tl1 Te5"`) | `thermal_conductivity` (W m⁻¹ K⁻¹), `imputed_features` |
| `dataset` | deferred | `file` (CSV with a `composition` column) | `predictions.csv` (input rows + `kappa_pred`, `imputed_features`, `error`), `predictions_hist.png` |

`imputed_features` counts the features that are undefined for the composition
because Magpie has no data for an element, e.g. a noble gas's electronegativity.
Those features are replaced with the training-set mean. A non-zero count means
the prediction is less reliable. In `dataset` mode, a row that cannot be parsed
gets an `error` message and an empty `kappa_pred`, and the rest of the batch
still runs.

## Model artifacts (runtime assets)

One file is **not committed to git** and **not baked into the image**. It is
declared as an `asset` in `schema/model.json` and provided by the LMDS platform
at run time:

| Asset path | Upload this file | Size |
|------------|------------------|------|
| `artifacts/model.pk` | the original `pickle_jar/model.pk` (unchanged) | 15 MB |

1. An admin uploads the file for the declared asset path under the version's
   **Manage → Versions → Assets** page (only schema-declared paths are accepted).
2. On each run, the runner downloads it from S3 to `./artifacts/model.pk` under
   the container WORKDIR before the handler executes.

The model reads it from `./artifacts/` (the package's `artifacts/` dir is the
fallback for a local source checkout; `MODEL_ARTIFACTS_DIR` overrides both), and
fails fast if it is missing. See
[`src/model/artifacts/README.md`](src/model/artifacts/README.md).

**Local development:** put `model.pk` in `src/model/artifacts/`.

## How the original 2020-era model runs on a modern stack

The original tool needs Python ≤3.8 with automatminer 1.0.3, matminer 0.6.2,
pymatgen 2020.1.28 and scikit-learn 0.22.2 (automatminer pins these exactly).
The LMDS runner needs Python ≥3.10, so the conversion does not unpickle
anything with that stack at run time:

- **Random forest:** `src/model/forest.py` reads the original `model.pk` with
  an allow-listed unpickler. It maps the scikit-learn classes onto inert
  stand-ins and never imports or executes scikit-learn code. It then evaluates
  the trees in numpy with scikit-learn 0.22.2's semantics: float32 inputs,
  `x <= threshold` goes left, and the forest output is the mean of the trees.
- **Featurizer:** `featurizer.pk` (automatminer `AutoFeaturizer` +
  `DataCleaner` + sklearn `RobustScaler`) is replaced by modern matminer
  Magpie features plus `src/model/featurizer_state.json`. That file holds the
  feature order, scaler parameters, training means and target scale, and was
  exported once from `featurizer.pk` by `training/export_featurizer_state.py`
  (run under the legacy stack; instructions are in the script). Only the Magpie
  features survive the original DataCleaner (the oxidation-state/ionic features
  were dropped), so this is the complete fitted state.
  `ElementProperty(..., impute_nan=False)` keeps matminer 0.6.2's behaviour of
  returning NaN for missing elemental data. Modern matminer imputes by default.

**Verified:** `tests/data/golden_predictions.csv` holds predictions from the
original pipeline (Python 3.8 env, original pickles) for 317 compositions, 45
of which exercise NaN imputation. The tool reproduces all of them to ≤4e-15,
and `tests/test_model.py` checks this on every run.

### Behaviour differences from the original `predict.py`

1. **No `exp()` on the output.** `predict.py` treats the model output as
   log κ and returns `exp(pred) - 1e-6`. But the committed pickles were saved on
   2022-11-07, when training used *linear* κ. The 2023-01-09 commit switched
   `train_and_save.py` to a log target with 329 trees and added the `exp()`, but
   the pickles were never re-saved. The pickled forest has 100 default trees,
   and its stored training targets are linear (0.08–49.6 W m⁻¹ K⁻¹). It fits
   linear κ with R² 0.94. The original `predict.py` therefore returned
   nonsense such as Al₂O₃ ≈ 2×10⁸ W m⁻¹ K⁻¹. This tool returns the forest
   output directly (Al₂O₃ ≈ 19.3). `featurizer_state.json` records
   `target_transform` (detected from the stored targets), so a model retrained
   with `train_and_save.py` and re-exported would get `exp()` applied
   automatically.
2. **Row-independent NaN imputation.** For a batch, automatminer's DataCleaner
   forward/back-filled a missing feature from the *neighbouring rows* of the
   input CSV, so a prediction depended on row order. Here every row uses the
   rule automatminer applied to a single composition: impute with the
   training mean. Results are identical for single compositions and for any
   composition with full Magpie data.
3. `computational_label` is fixed at 0 (treated as an experimental measurement),
   as in `predict.py`.

As a random forest, predictions are bounded by the training targets (≈0.1–50
W m⁻¹ K⁻¹), so very high conductors (e.g. diamond, C ≈ 16) are under-predicted.

## Project Structure

```
├── schema/
│   ├── model.json              # Contract (instance + dataset modes, model.pk asset)
│   └── model.schema.json       # Shared meta-schema (same as the other converted tools)
├── src/
│   ├── cmds/                   # Entrypoints — copied from template, unmodified
│   ├── infrastructure/         # Runner/handler/S3/logging — copied, unmodified
│   └── model/
│       ├── __init__.py         # handler = ThermalConductivityModel()
│       ├── model.py            # Magpie featurisation + scaling + prediction
│       ├── forest.py           # sklearn-0.22.2 forest reader/evaluator (no sklearn)
│       ├── featurizer_state.json  # Fitted featurizer state (from featurizer.pk)
│       ├── types.py            # Auto-generated from model.json
│       └── artifacts/          # model.pk — provided at runtime, NOT in git
├── training/                   # Original training/CV scripts + data (not packaged)
│   └── export_featurizer_state.py
├── data/                       # Sample inputs (output/ ignored)
├── scripts/                    # run_instance.sh, run_dataset.sh
├── tests/                      # test_model.py + golden predictions
├── Dockerfile
└── pyproject.toml
```

`training/` keeps the original `train_and_save.py`, `cross_validate.py`,
`locallib.py`, `data.csv` and `post_train_sanity_check.csv` for provenance.
They run only under the legacy stack.

## Local development

```bash
uv sync --extra dev
uv run validate-model schema/model.json
uv run model-types schema/model.json -o src/model/types.py
uv run pytest tests/ -v
```

## Dependencies

Runs on the template default **Python 3.12**. `matminer` + `pymatgen` provide
composition parsing and Magpie features. `numpy`/`pandas` handle the forest
and CSV I/O, and `matplotlib` draws the histogram. scikit-learn is only a
transitive dependency of matminer and is not used to load the model. Docker
installs the exact `uv.lock` set (`uv sync --frozen`).

## Docker

```bash
docker build -t thermal-conductivity-rf:1.0.0 .

# On the platform the runner downloads the declared asset to ./artifacts before
# running. To run the image by hand, mount the model where the runner would put
# it (/app/artifacts):
docker run --rm \
  -v /path/to/artifacts:/app/artifacts:ro \
  thermal-conductivity-rf:1.0.0 --local \
  --parameters '{"mode":"instance","inputs":{"composition":{"value":"Bi2Te3"}},"parameters":{}}'
```

## Environment Variables

Copy `.env.example` to `.env` and adjust for remote (S3) runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |
