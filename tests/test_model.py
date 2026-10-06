import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from infrastructure.logging import newLogger
from model.forest import Forest
from model.model import ThermalConductivityModel, _artifacts_dir

LOG = newLogger("test")
ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "input.csv"
# Predictions from the ORIGINAL pipeline (Python 3.8, automatminer 1.0.3,
# matminer 0.6.2, pymatgen 2020.1.28, scikit-learn 0.22.2), one composition at
# a time, for 317 compositions — 45 of which have missing elemental data and
# exercise the NaN-imputation path.
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_predictions.csv"


@pytest.fixture(scope="module")
def model() -> ThermalConductivityModel:
    return ThermalConductivityModel()


def _golden() -> list[tuple[str, float]]:
    with GOLDEN.open() as f:
        return [(r["composition"], float(r["kappa_pred"])) for r in csv.DictReader(f)]


def test_matches_original_pipeline(model: ThermalConductivityModel) -> None:
    golden = _golden()
    assert len(golden) == 317
    got = np.array([model.predict(c)["thermal_conductivity"] for c, _ in golden])
    want = np.array([k for _, k in golden])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_imputation_reported(model: ThermalConductivityModel) -> None:
    # He has no Magpie melting point / electronegativity etc.
    assert model.predict("HeLi")["imputed_features"] == 12
    assert model.predict("Bi2Te3")["imputed_features"] == 0


def test_instance(model: ThermalConductivityModel) -> None:
    result = model.process(
        "instance", {"composition": "Bi2Te3"}, {}, Path("/tmp/tc_out"), {}, LOG
    )
    assert set(result) == {"thermal_conductivity", "imputed_features"}
    assert result["thermal_conductivity"] == pytest.approx(1.939591024380001, abs=1e-12)
    assert result["imputed_features"] == 0


def test_composition_formats_agree(model: ThermalConductivityModel) -> None:
    # The training data writes formulas space-separated with explicit 1s.
    a = model.predict("Ag9 Tl1 Te5")["thermal_conductivity"]
    b = model.predict("Ag9TlTe5")["thermal_conductivity"]
    assert a == b


def test_instance_missing_composition(model: ThermalConductivityModel) -> None:
    with pytest.raises(ValueError, match="No composition"):
        model.process("instance", {}, {}, Path("/tmp/tc_out"), {}, LOG)


@pytest.mark.parametrize("bad", ["", "   ", "NotAnElement", 42])
def test_instance_invalid_composition(model: ThermalConductivityModel, bad) -> None:
    with pytest.raises(Exception):
        model.process("instance", {"composition": bad}, {}, Path("/tmp/tc_out"), {}, LOG)


def test_dataset(model: ThermalConductivityModel, tmp_path: Path) -> None:
    out = tmp_path / "output"
    result = model.process(
        "dataset", {}, {"file": (INPUT_CSV, "text/csv")}, out, {}, LOG
    )
    assert result == {}
    assert (out / "predictions_hist.png").stat().st_size > 0

    with (out / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    with INPUT_CSV.open() as f:
        inputs = list(csv.DictReader(f))
    assert [r["composition"] for r in rows] == [r["composition"] for r in inputs]
    assert all(r["error"] == "" for r in rows)
    # Dataset predictions are row-independent and equal instance predictions.
    for r in rows:
        assert float(r["kappa_pred"]) == pytest.approx(
            model.predict(r["composition"])["thermal_conductivity"], abs=1e-12
        )


def test_dataset_bad_rows_do_not_abort(model: ThermalConductivityModel, tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("id,composition\n1,PbTe\n2,NotAnElement\n3,\n4,Si\n")
    out = tmp_path / "output"
    model.process("dataset", {}, {"file": (src, "text/csv")}, out, {}, LOG)
    with (out / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["1", "2", "3", "4"]  # pass-through column
    assert rows[0]["error"] == "" and rows[0]["kappa_pred"] != ""
    assert rows[1]["error"] != "" and rows[1]["kappa_pred"] == ""
    assert rows[2]["error"] != "" and rows[2]["kappa_pred"] == ""
    assert rows[3]["error"] == "" and rows[3]["kappa_pred"] != ""


def test_dataset_missing_column(model: ThermalConductivityModel, tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("formula\nPbTe\n")
    with pytest.raises(ValueError, match="'composition' column"):
        model.process("dataset", {}, {"file": (src, "text/csv")}, tmp_path / "o", {}, LOG)


def test_dataset_no_file(model: ThermalConductivityModel) -> None:
    with pytest.raises(ValueError, match="No input file"):
        model.process("dataset", {}, {}, Path("/tmp/tc_out"), {}, LOG)


def test_unknown_mode(model: ThermalConductivityModel) -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        model.process("bogus", {}, {}, Path("/tmp/tc_out"), {}, LOG)


def test_forest_refuses_unexpected_globals(tmp_path: Path) -> None:
    # The forest unpickler only resolves scikit-learn tree classes and numpy
    # array reconstruction; anything else is rejected rather than executed.
    evil = tmp_path / "evil.pk"
    evil.write_bytes(pickle.dumps(print))
    with pytest.raises(pickle.UnpicklingError, match="Refusing"):
        Forest(evil)


def test_forest_metadata() -> None:
    forest = Forest(_artifacts_dir() / "model.pk")
    assert forest.n_estimators == 100
    assert forest.n_features == 133
    assert forest.sklearn_version == "0.22.2"
