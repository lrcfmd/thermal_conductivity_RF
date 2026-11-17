import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matminer.featurizers.composition import ElementProperty  # noqa: E402
from pymatgen.core import Composition, Element  # noqa: E402

from infrastructure.handler import InputError  # noqa: E402
from infrastructure.logging import newLogger  # noqa: E402
from model.forest import Forest  # noqa: E402

# Fitted featurizer state (feature order, RobustScaler parameters, training
# means for NaN imputation, target scale), exported once from the original
# featurizer.pk by training/export_featurizer_state.py. Small, so it ships in
# the package.
STATE_PATH = Path(__file__).resolve().parent / "featurizer_state.json"

MODEL_FILE = "model.pk"

# The forest takes a "computational_label" feature: 1 if the training target
# was a computed lattice conductivity, 0 if it was measured. The original
# predict.py always predicts with 0 (i.e. as an experimental measurement).
COMPUTATIONAL_LABEL = 0

EPS = 1e-6  # offset used by the log target transform in train_and_save.py


# The model file is NOT committed to git; it is declared as an `asset` in
# schema/model.json and provided at runtime. On the platform the runner
# downloads it to its declared path under the container WORKDIR (./artifacts/…),
# so that is the primary location. For a local source checkout the file lives
# in the package's own artifacts/ dir. MODEL_ARTIFACTS_DIR overrides both. See
# artifacts/README.md.
def _artifacts_dir() -> Path:
    env = os.environ.get("MODEL_ARTIFACTS_DIR")
    if env:
        return Path(env)
    workdir_assets = Path("artifacts")  # where the runner downloads assets
    if workdir_assets.is_dir():
        return workdir_assets
    return Path(__file__).resolve().parent / "artifacts"


FORMULA_HINT = (
    "Use element symbols with amounts, e.g. Bi2Te3 or Ag9 Tl1 Te5 "
    "(symbols are case-sensitive)."
)


def parse_composition(composition) -> Composition:
    """A pymatgen Composition, or an InputError saying what is wrong with it."""
    if composition is None or (
        isinstance(composition, float) and np.isnan(composition)
    ) or (isinstance(composition, str) and not composition.strip()):
        raise InputError(f"The composition is empty. {FORMULA_HINT}")
    if not isinstance(composition, str):
        raise InputError(
            f"{composition!r} is not a valid chemical formula. {FORMULA_HINT}"
        )
    text = composition.strip()
    try:
        comp = Composition(text)
    except KeyError as e:  # well-formed token that is not an element, e.g. "Xx"
        raise InputError(
            f"{text!r} is not a valid chemical formula: {e.args[0]!r} is not an "
            f"element symbol. {FORMULA_HINT}"
        ) from None
    except ValueError as e:
        if "negative" in str(e):
            raise InputError(
                f"{text!r} has a negative amount; element amounts must be above 0."
            ) from None
        raise InputError(
            f"{text!r} is not a valid chemical formula. {FORMULA_HINT}"
        ) from None
    except Exception:  # pymatgen raises assorted errors for odd strings
        raise InputError(
            f"{text!r} is not a valid chemical formula. {FORMULA_HINT}"
        ) from None
    unknown = [str(el.symbol) for el in comp.elements if not isinstance(el, Element)]
    if unknown:  # pymatgen reads symbols such as "Xx" as placeholder species
        raise InputError(
            f"{text!r} is not a valid chemical formula: {unknown[0]!r} is not an "
            f"element symbol. {FORMULA_HINT}"
        )
    amounts = list(comp.values())
    if not all(np.isfinite(a) for a in amounts):
        raise InputError(f"{text!r} has an element amount that is too large.")
    if comp.num_atoms <= 0:
        raise InputError(
            f"{text!r} has no atoms; element amounts must be above 0."
        )
    return comp


class ThermalConductivityModel:
    """Predict thermal conductivity (W m^-1 K^-1) from chemical composition.

    Ported from the LMDS thermal_conductivity_RF tool. Each composition is
    featurised with the 132 Magpie ElementProperty statistics (matminer) plus
    the computational_label flag, robust-scaled with the training-time
    parameters, and passed to the original random forest. Two Modes share one
    prediction path:
      ``instance`` — one composition, prediction returned inline.
      ``dataset``  — a CSV with a ``composition`` column, predictions CSV +
                     histogram out.
    """

    def __init__(self):
        self.logger = newLogger("model")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        state = json.loads(STATE_PATH.read_text())
        self.columns = state["columns"]
        self.center = np.asarray(state["center"], dtype=np.float64)
        self.scale = np.asarray(state["scale"], dtype=np.float64)
        self.fill_values = np.asarray(state["fill_values"], dtype=np.float64)
        self.target_transform = state["target_transform"]
        if self.target_transform not in ("none", "log"):
            raise ValueError(f"Unknown target_transform {self.target_transform!r}")

        artifacts = _artifacts_dir()
        model_path = artifacts / MODEL_FILE
        self.logger.info("loading artifacts", dir=str(artifacts))
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact missing from {artifacts}: {MODEL_FILE}. "
                "This file is provided to the application at runtime and is "
                "not committed to git — see src/model/artifacts/README.md. Set "
                "MODEL_ARTIFACTS_DIR to the directory that holds it, or mount "
                "it into the default location."
            )
        self.forest = Forest(model_path)
        if self.forest.n_features != len(self.columns):
            raise ValueError(
                f"{MODEL_FILE} expects {self.forest.n_features} features but "
                f"featurizer_state.json defines {len(self.columns)}; they must "
                "come from the same training run."
            )

        # impute_nan=False: matminer >= 0.8 fills missing elemental data (e.g.
        # noble-gas electronegativity) with a per-property element mean by
        # default; matminer 0.6.2, which the model was trained with, returned
        # NaN, which the DataCleaner then filled with the training mean.
        self.element_property = ElementProperty.from_preset("magpie", impute_nan=False)
        labels = self.element_property.feature_labels()
        index = {label: i for i, label in enumerate(labels)}
        # Position of each model column in the Magpie feature vector
        # (None for computational_label, which is not a Magpie feature).
        self._magpie_index = [index.get(c) for c in self.columns]
        missing = [
            c for c, i in zip(self.columns, self._magpie_index)
            if i is None and c != "computational_label"
        ]
        if missing:
            raise ValueError(f"matminer does not produce features: {missing}")
        self._loaded = True

    def featurize(self, composition: str) -> tuple[np.ndarray, int]:
        """Scaled feature vector for one composition, and how many raw
        features were NaN and had to be imputed with the training mean."""
        self._load()
        comp = parse_composition(composition)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            magpie = self.element_property.featurize(comp)
        raw = np.array(
            [
                COMPUTATIONAL_LABEL if i is None else magpie[i]
                for i in self._magpie_index
            ],
            dtype=np.float64,
        )
        # automatminer's DataCleaner: inf -> NaN, then a feature that is NaN
        # for the sample is imputed with its training-set mean. (For a batch,
        # automatminer instead forward/back-filled from neighbouring rows; the
        # per-sample rule here makes predictions independent of row order and
        # is identical for single compositions.)
        raw[~np.isfinite(raw)] = np.nan
        nan = np.isnan(raw)
        raw[nan] = self.fill_values[nan]
        return (raw - self.center) / self.scale, int(nan.sum())

    def _to_kappa(self, pred: np.ndarray) -> np.ndarray:
        if self.target_transform == "log":
            return np.exp(pred) - EPS
        return pred

    def predict(self, composition: str) -> dict:
        x, n_imputed = self.featurize(composition)
        kappa = float(self._to_kappa(self.forest.predict(x[None, :]))[0])
        return {"thermal_conductivity": kappa, "imputed_features": n_imputed}

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        if mode == "instance":
            return self._process_instance(values, logger)
        elif mode == "dataset":
            return self._process_dataset(files, output_dir, logger)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _process_instance(self, values: dict, logger) -> dict:
        composition = values.get("composition")
        if composition is None or (
            isinstance(composition, str) and not composition.strip()
        ):
            raise InputError(f"Please enter a composition. {FORMULA_HINT}")
        parse_composition(composition)  # report a bad formula before loading the model
        result = self.predict(composition)
        logger.info("predicted", composition=composition, **result)
        if result["imputed_features"]:
            logger.warning(
                "features imputed with training means",
                composition=composition, n=result["imputed_features"],
            )
        return result

    def _process_dataset(
        self, files: dict[str, tuple[Path, str]], output_dir: Path, logger
    ) -> dict:
        if "file" not in files:
            raise InputError(
                "No CSV file was given. Upload a CSV with a 'composition' column "
                "of chemical formulas."
            )
        csv_path, _ = files["file"]
        data = self._read_csv(csv_path)

        self._load()
        X = np.zeros((len(data), len(self.columns)))
        ok = np.zeros(len(data), dtype=bool)
        imputed = [""] * len(data)
        errors = [""] * len(data)
        for i, comp in enumerate(data["composition"]):
            try:
                X[i], imputed[i] = self.featurize(comp)
                ok[i] = True
            except Exception as e:  # a bad composition shouldn't abort the batch
                logger.warning("failed to featurize", row=i, composition=str(comp), error=str(e))
                errors[i] = str(e)

        if not ok.any():
            raise InputError(
                f"None of the {len(data)} compositions could be read. The first "
                f"one ({str(data['composition'].iloc[0])!r}) failed: {errors[0]}"
            )

        kappa = np.full(len(data), np.nan)
        kappa[ok] = self._to_kappa(self.forest.predict(X[ok]))

        # Same layout as the original predict.py: the input columns plus
        # kappa_pred, with a status column per row.
        data["kappa_pred"] = kappa
        data["imputed_features"] = imputed
        data["error"] = errors
        logger.info("predicted dataset", rows=len(data), failed=int((~ok).sum()))

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.csv"
        data.to_csv(output_path, index=False)
        logger.info("saved asset", path=str(output_path))

        hist_path = output_dir / "predictions_hist.png"
        self._plot_histogram(kappa[ok], hist_path)
        logger.info("saved asset", path=str(hist_path))
        return {}

    @staticmethod
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        """Read the dataset CSV, turning read problems into InputErrors."""
        try:
            data = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise InputError(
                "The CSV file is empty. It needs a header row with a "
                "'composition' column and one chemical formula per row."
            ) from None
        except UnicodeDecodeError:
            raise InputError(
                "The file is not a plain-text CSV file. Save it as CSV (UTF-8) "
                "with a 'composition' column."
            ) from None
        except pd.errors.ParserError as e:
            raise InputError(
                "Could not read the CSV file: some rows have more values than "
                f"the header row ({_parser_detail(e)}). Check for unquoted commas "
                "in a cell."
            ) from None
        except Exception as e:
            raise InputError(f"Could not read the CSV file ({e}).") from None

        columns = [str(c) for c in data.columns]
        if "composition" not in columns:
            close = [c for c in columns if c.strip().lower() == "composition"]
            hint = (
                f" Rename the column {close[0]!r} to 'composition'."
                if close
                else ""
            )
            raise InputError(
                "The CSV file needs a 'composition' column (found: "
                f"{', '.join(columns)}).{hint}"
            )
        if data.empty:
            raise InputError(
                "The CSV file has a header row but no compositions; add one "
                "chemical formula per row."
            )
        return data

    @staticmethod
    def _plot_histogram(kappa: np.ndarray, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if kappa.size:
            # "fd" (as in predict.py) needs spread in the data.
            bins = "fd" if kappa.size > 1 and np.ptp(kappa) > 0 else 10
            ax.hist(kappa, bins=bins, color="#4c72b0", edgecolor="white")
        ax.set_xlabel(r"predicted $\kappa$ (W m$^{-1}$ K$^{-1}$)")
        ax.set_ylabel("count")
        ax.set_title(f"Predicted thermal conductivity (n = {kappa.size})")
        fig.tight_layout()
        fig.savefig(path, format="png", dpi=100)
        plt.close(fig)


def _parser_detail(e: Exception) -> str:
    """'Expected 1 fields in line 3, saw 2' from a pandas ParserError."""
    msg = str(e)
    return msg.split("C error: ", 1)[-1].strip().rstrip(".")
