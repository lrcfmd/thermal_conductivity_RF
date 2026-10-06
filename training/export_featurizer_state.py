"""Export the fitted featurizer state from featurizer.pk to a small JSON file.

`featurizer.pk` is a pickled `locallib.automatminer_featurizer_sklearn_pipeline`
(automatminer AutoFeaturizer + DataCleaner + sklearn RobustScaler). Unpickling it
needs the original 2020 stack (Python <=3.8, automatminer 1.0.3.20200727,
matminer 0.6.2, pymatgen 2020.1.28, scikit-learn 0.22.2, pandas<1.1), which
cannot coexist with the LMDS runner (Python >=3.10). The fitted state the tool
actually needs is tiny, so it is exported once to
`src/model/featurizer_state.json` and shipped inside the package:

- `columns`     final feature order fed to the random forest
- `center`, `scale`  RobustScaler parameters (same order)
- `fill_values` training-set column means, used by the DataCleaner to impute
                features that are NaN for a sample

Run in a legacy env (uv resolves it as of the model's training date):

    uv venv --python 3.8 /tmp/legacy38
    VIRTUAL_ENV=/tmp/legacy38 uv pip install --exclude-newer 2020-08-01 \
        "automatminer==1.0.3.20200727" "numpy==1.18.5" "cython<3" wheel setuptools
    # pymatgen's Cython extensions must be built against the pinned numpy:
    VIRTUAL_ENV=/tmp/legacy38 uv pip install --exclude-newer 2020-08-01 \
        --reinstall --no-deps --no-build-isolation "pymatgen==2020.1.28"
    /tmp/legacy38/bin/python training/export_featurizer_state.py \
        src/model/artifacts/featurizer.pk src/model/featurizer_state.json
"""

import hashlib
import json
import os
import pickle
import sys

import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for locallib


def main(src, dst):
    with open(src, "rb") as f:
        raw = f.read()
    feat = pickle.loads(raw)

    af, dc, scaler = feat.pipe, feat.DataCleaner, feat.RobustScalar
    target = dc.fitted_target
    columns = [c for c in dc.fitted_df.columns if c != target]
    assert len(columns) == len(scaler.center_) == len(scaler.scale_)
    fill = dc.fitted_df[columns].mean()

    # The DataCleaner keeps the training targets, which tells us whether the
    # model was trained on log(kappa + 1e-6) (current train_and_save.py) or on
    # linear kappa (the version that produced the committed 2022 pickles).
    # Measured kappa is strictly positive and the log transform maps
    # kappa < 1 to negative values, so a negative target means "log".
    y = dc.fitted_df[target]
    target_transform = "log" if (y < 0).any() else "none"

    state = {
        "source": {
            "file": os.path.basename(src),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sklearn_version": sklearn.__version__,
        },
        "featurizer": {
            "preset": af.preset,
            # Featurizers applied to the composition column. Only the Magpie
            # ElementProperty features survive the DataCleaner (the rest were
            # all-NaN on the training set and dropped), so they are recorded
            # for provenance only.
            "composition_featurizers": [
                type(f).__name__ for f in af.featurizers["composition"]
            ],
            "dropped_features": sorted(dc.dropped_features),
        },
        "target": target,
        # "log": model predicts log(kappa + 1e-6) -> kappa = exp(pred) - 1e-6
        # "none": model predicts kappa directly
        "target_transform": target_transform,
        "target_range": [float(y.min()), float(y.max())],
        "columns": columns,
        "center": [float(v) for v in scaler.center_],
        "scale": [float(v) for v in scaler.scale_],
        "fill_values": [float(fill[c]) for c in columns],
    }
    with open(dst, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    print("wrote", dst, "with", len(columns), "features; target_transform =",
          target_transform)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
