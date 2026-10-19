
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, low_memory=False)


def test_source_files_exist():
    required_files = [
        "patients.csv",
        "encounters.csv",
        "observations.csv",
        "conditions.csv",
    ]

    for name in required_files:
        assert (DATA_DIR / name).is_file(), f"Missing file: {name}"


def test_patient_ids_are_present_and_unique():
    patients = load_csv("patients.csv")

    assert patients["Id"].notna().all()
    assert patients["Id"].is_unique


def test_encounter_ids_are_present_and_unique():
    encounters = load_csv("encounters.csv")

    assert encounters["Id"].notna().all()
    assert encounters["Id"].is_unique


def test_child_tables_reference_known_patients():
    patients = load_csv("patients.csv")
    known_ids = set(patients["Id"].dropna())

    for name in ["encounters.csv", "observations.csv", "conditions.csv"]:
        df = load_csv(name)

        assert df["PATIENT"].notna().all(), (
            f"{name} contains missing patient references"
        )

        unmatched = ~df["PATIENT"].isin(known_ids)

        assert not unmatched.any(), (
            f"{name} contains {int(unmatched.sum())} unknown patient references"
        )


def test_required_columns_exist():
    expected = {
        "patients.csv": {"Id", "BIRTHDATE"},
        "encounters.csv": {"Id", "PATIENT", "START", "STOP"},
        "observations.csv": {
            "DATE", "PATIENT", "ENCOUNTER",
            "CODE", "DESCRIPTION", "VALUE", "UNITS",
        },
        "conditions.csv": {
            "START", "STOP", "PATIENT", "ENCOUNTER",
            "CODE", "DESCRIPTION",
        },
    }

    for name, required_columns in expected.items():
        df = load_csv(name)
        missing = required_columns - set(df.columns)

        assert not missing, f"{name} is missing columns: {sorted(missing)}"


def test_encounter_references_are_valid():
    encounters = load_csv("encounters.csv")
    encounter_ids = set(encounters["Id"].dropna())

    for name in ["observations.csv", "conditions.csv"]:
        df = load_csv(name)

        # Missing encounter references may be valid; investigate separately.
        present = df["ENCOUNTER"].dropna()
        unmatched = ~present.isin(encounter_ids)

        assert not unmatched.any(), (
            f"{name} contains {int(unmatched.sum())} "
            "encounter references not found in encounters.csv"
        )


def test_encounter_dates_are_parseable():
    encounters = load_csv("encounters.csv")

    for column in ["START", "STOP"]:
        parsed = pd.to_datetime(
            encounters[column], errors="coerce", utc=True
        )

        invalid = encounters[column].notna() & parsed.isna()

        assert not invalid.any(), (
            f"encounters.{column} has invalid dates"
        )

