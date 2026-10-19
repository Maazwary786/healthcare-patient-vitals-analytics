
"""Quality tests for the generated healthcare analytics tables."""

import pandas as pd
import pytest

from ingestion.build_analytics import main


@pytest.fixture(scope="module")
def analytics_tables():
    """Build outputs once, then load them for the tests."""
    main()

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"

    return {
        "patients": pd.read_csv(processed / "dim_patient.csv"),
        "encounters": pd.read_csv(processed / "fact_encounters.csv"),
        "vitals": pd.read_csv(processed / "fact_vitals.csv"),
    }


def test_patient_dimension_schema(analytics_tables):
    df = analytics_tables["patients"]

    expected = {
        "patient_key", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"
    }
    assert expected.issubset(df.columns)


def test_patient_keys_are_present_and_unique(analytics_tables):
    df = analytics_tables["patients"]

    assert df["patient_key"].notna().all()
    assert df["patient_key"].is_unique


def test_encounter_fact_has_unique_ids(analytics_tables):
    df = analytics_tables["encounters"]

    assert "source_encounter_id" in df.columns
    assert df["source_encounter_id"].notna().all()
    assert df["source_encounter_id"].is_unique


def test_vitals_have_valid_timestamps_and_numeric_values(analytics_tables):
    df = analytics_tables["vitals"]

    timestamps = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    values = pd.to_numeric(df["numeric_value"], errors="coerce")

    assert timestamps.notna().all()
    assert values.notna().all()
    assert values.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all()


def test_fact_patient_keys_match_patient_dimension(analytics_tables):
    patients = analytics_tables["patients"]
    encounters = analytics_tables["encounters"]
    vitals = analytics_tables["vitals"]

    known_keys = set(patients["patient_key"])

    assert encounters["patient_key"].notna().all()
    assert vitals["patient_key"].notna().all()
    assert set(encounters["patient_key"]).issubset(known_keys)
    assert set(vitals["patient_key"]).issubset(known_keys)


def test_vitals_measurement_values_are_supported(analytics_tables):
    df = analytics_tables["vitals"]

    expected = {
        "heart_rate",
        "respiratory_rate",
        "systolic_bp",
        "diastolic_bp",
    }

    assert not df.empty
    assert set(df["measurement"]).issubset(expected)
    assert df["measurement"].notna().all()
