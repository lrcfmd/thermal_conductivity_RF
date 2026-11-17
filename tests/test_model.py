import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from infrastructure.handler import InputError
from infrastructure.logging import newLogger
from model.forest import Forest
from model.model import ThermalConductivityModel, _artifacts_dir, parse_composition

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
    with pytest.raises(InputError, match="Please enter a composition"):
        model.process("instance", {}, {}, Path("/tmp/tc_out"), {}, LOG)


@pytest.mark.parametrize("bad", ["", "   ", "NotAnElement", 42])
def test_instance_invalid_composition(model: ThermalConductivityModel, bad) -> None:
    with pytest.raises(InputError):
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
    with pytest.raises(InputError, match="'composition' column"):
        model.process("dataset", {}, {"file": (src, "text/csv")}, tmp_path / "o", {}, LOG)


def test_dataset_no_file(model: ThermalConductivityModel) -> None:
    with pytest.raises(InputError, match="No CSV file was given"):
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


# --- input errors ----------------------------------------------------------

SCHEMA = ROOT / "schema" / "model.json"


@pytest.mark.parametrize(
    "bad, match",
    [
        ("", "composition is empty"),
        (None, "composition is empty"),
        (42, "not a valid chemical formula"),
        (float("nan"), "composition is empty"),
        ("Xx2O", "'Xx' is not an element symbol"),
        ("NotAnElement", "not a valid chemical formula"),
        ("bi2te3", "not a valid chemical formula.*case-sensitive"),
        ("Bi2Te3)", "not a valid chemical formula"),
        ("Fe2O3, Al2O3", "not a valid chemical formula"),
        ("1234", "not a valid chemical formula"),
        ("H-2O", "negative amount"),
        ("Bi0", "has no atoms"),
        ("Fe1e400", "too large"),
    ],
)
def test_parse_composition_errors(bad, match) -> None:
    with pytest.raises(InputError, match=match):
        parse_composition(bad)


@pytest.mark.parametrize("good", ["Bi2Te3", "Ag9 Tl1 Te5", "Fe(OH)3", " PbTe \n", "Fe2.5O3"])
def test_parse_composition_accepts(good) -> None:
    assert parse_composition(good).num_atoms > 0


@pytest.mark.parametrize("value", [None, "", "  "])
def test_instance_blank_composition(model: ThermalConductivityModel, value) -> None:
    with pytest.raises(InputError, match="Please enter a composition"):
        model.process("instance", {"composition": value}, {}, Path("/tmp/tc_out"), {}, LOG)


def test_instance_unknown_element_message(model: ThermalConductivityModel) -> None:
    with pytest.raises(InputError, match="'Xx' is not an element symbol"):
        model.process("instance", {"composition": "Xx2O"}, {}, Path("/tmp/tc_out"), {}, LOG)


def _dataset(model, tmp_path, content: bytes | str):
    src = tmp_path / "in.csv"
    if isinstance(content, str):
        src.write_text(content)
    else:
        src.write_bytes(content)
    out = tmp_path / "output"
    model.process("dataset", {}, {"file": (src, "text/csv")}, out, {}, LOG)
    return out


@pytest.mark.parametrize(
    "content, match",
    [
        ("", "CSV file is empty"),
        ("composition\n", "header row but no compositions"),
        ("Composition\nPbTe\n", "Rename the column 'Composition' to 'composition'"),
        ("formula,name\nPbTe,x\n", "found: formula, name"),
        ("composition\nPbTe\nFe2O3, Al2O3\n", "Expected 1 fields in line 3, saw 2"),
        (b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x00", "not a plain-text CSV"),
    ],
)
def test_dataset_bad_file(model: ThermalConductivityModel, tmp_path: Path, content, match) -> None:
    with pytest.raises(InputError, match=match):
        _dataset(model, tmp_path, content)


def test_dataset_all_rows_bad(model: ThermalConductivityModel, tmp_path: Path) -> None:
    with pytest.raises(InputError, match=r"None of the 2 compositions.*'Xx2O'.*'Xx' is not an element"):
        _dataset(model, tmp_path, "composition\nXx2O\nbi2te3\n")


def test_dataset_row_errors_are_plain(model: ThermalConductivityModel, tmp_path: Path) -> None:
    out = _dataset(model, tmp_path, "composition\nPbTe\nXx2O\n\"\"\n")
    with (out / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["error"] == ""
    assert "'Xx' is not an element symbol" in rows[1]["error"]
    assert "composition is empty" in rows[2]["error"]


def test_schema_example_parses_and_runs(model: ThermalConductivityModel, tmp_path: Path) -> None:
    spec = json.loads(SCHEMA.read_text())["execution"]["modes"]["dataset"]["inputs"]["file"]
    assert spec["header"] is True
    assert [c["name"] for c in spec["columns"]] == ["composition"]
    out = _dataset(model, tmp_path, spec["example"])
    with (out / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == spec["example"].count("\n") - 1
    assert all(r["error"] == "" and r["kappa_pred"] != "" for r in rows)


def test_runner_reports_input_error_without_traceback(tmp_path: Path, capsys) -> None:
    from infrastructure.runner import run

    config = {"mode": "instance", "inputs": {"composition": {"value": "Xx2O"}}}
    with pytest.raises(SystemExit) as exc:
        run(ThermalConductivityModel(), ["--local", "--parameters", json.dumps(config)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.strip().splitlines()[-1].startswith("LMDS_ERROR: 'Xx2O' is not a valid")
