
"""Build validated healthcare analytics tables and publish a versioned snapshot."""

import hashlib
import logging
from pathlib import Path

import pandas as pd

from ingestion.publish import publish_tables
from ingestion.quality import DataQualityError, load_validated_csv


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


PATIENT_COLUMNS = {
    "Id",
    "BIRTHDATE",
    "GENDER",
    "RACE",
    "ETHNICITY",
}

ENCOUNTER_COLUMNS = {
    "Id",
    "PATIENT",
    "START",
    "STOP",
    "ENCOUNTERCLASS",
    "DESCRIPTION",
    "BASE_ENCOUNTER_COST",
    "TOTAL_CLAIM_COST",
}

OBSERVATION_COLUMNS = {
    "DATE",
    "PATIENT",
    "ENCOUNTER",
    "DESCRIPTION",
    "VALUE",
    "UNITS",
}

VITALS = {
    "heart rate": "heart_rate",
    "respiratory rate": "respiratory_rate",
    "systolic blood pressure": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
}


def patient_key(value):
    """Create a deterministic pseudonymous patient key.

    This hash is pseudonymization, not guaranteed anonymization.
    """
    if pd.isna(value):
        return None

    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()


def validate_patient_references(
    df: pd.DataFrame,
    known_patient_ids: set,
    table_name: str,
) -> None:
    """Reject non-null patient references that do not exist."""
    unmatched = (
        df["PATIENT"].notna()
        & ~df["PATIENT"].isin(known_patient_ids)
    )

    if unmatched.any():
        count = int(unmatched.sum())
        raise DataQualityError(
            f"{table_name} contains {count} unknown patient references"
        )


def build_patient_dimension(
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Build the patient dimension with a limited set of attributes."""
    patient_dim = patients[
        ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]
    ].copy()

    patient_dim["patient_key"] = patient_dim["Id"].map(patient_key)
    patient_dim = patient_dim.drop(columns=["Id"])

    if patient_dim["patient_key"].isna().any():
        raise DataQualityError(
            "Patient dimension contains missing patient keys"
        )

    if patient_dim["patient_key"].duplicated().any():
        raise DataQualityError(
            "Patient dimension contains duplicate patient keys"
        )

    return patient_dim


def build_vitals_fact(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize supported numeric vital-sign observations."""
    obs = observations.copy()

    obs["measurement"] = (
        obs["DESCRIPTION"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(VITALS)
    )

    obs["numeric_value"] = pd.to_numeric(
        obs["VALUE"],
        errors="coerce",
    )

    obs["event_time"] = pd.to_datetime(
        obs["DATE"],
        errors="coerce",
        utc=True,
    )

    obs["patient_key"] = obs["PATIENT"].map(patient_key)

    vitals = obs.loc[
        obs["measurement"].notna()
        & obs["numeric_value"].notna()
        & obs["event_time"].notna()
        & obs["patient_key"].notna(),
        [
            "event_time",
            "patient_key",
            "ENCOUNTER",
            "measurement",
            "numeric_value",
            "UNITS",
        ],
    ].copy()

    vitals = vitals.rename(
        columns={
            "ENCOUNTER": "source_encounter_id",
            "UNITS": "unit",
        }
    )

    if vitals.empty:
        raise DataQualityError(
            "No valid supported vital measurements were found"
        )

    if not vitals["numeric_value"].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise DataQualityError(
            "Vitals fact contains non-finite numeric values"
        )

    return vitals


def build_encounters_fact(
    encounters: pd.DataFrame,
) -> pd.DataFrame:
    """Build the encounter analytics fact table."""
    df = encounters.copy()

    df["patient_key"] = df["PATIENT"].map(patient_key)

    df["START"] = pd.to_datetime(
        df["START"],
        errors="coerce",
        utc=True,
    )

    df["STOP"] = pd.to_datetime(
        df["STOP"],
        errors="coerce",
        utc=True,
    )

    df["duration_minutes"] = (
        df["STOP"] - df["START"]
    ).dt.total_seconds() / 60

    result = df[
        [
            "Id",
            "patient_key",
            "START",
            "STOP",
            "ENCOUNTERCLASS",
            "DESCRIPTION",
            "duration_minutes",
            "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
        ]
    ].rename(
        columns={
            "Id": "source_encounter_id",
            "START": "start_time",
            "STOP": "end_time",
            "ENCOUNTERCLASS": "encounter_class",
            "DESCRIPTION": "encounter_description",
            "BASE_ENCOUNTER_COST": "base_encounter_cost",
            "TOTAL_CLAIM_COST": "total_claim_cost",
        }
    )

    if result["source_encounter_id"].isna().any():
        raise DataQualityError(
            "Encounter fact contains missing encounter IDs"
        )

    if result["source_encounter_id"].duplicated().any():
        raise DataQualityError(
            "Encounter fact contains duplicate encounter IDs"
        )

    # Missing STOP timestamps can be valid for ongoing encounters.
    # Negative durations, however, indicate inconsistent timestamps.
    negative_duration = result["duration_minutes"] < 0

    if negative_duration.fillna(False).any():
        raise DataQualityError(
            "Encounter fact contains negative encounter durations"
        )

    return result


def main() -> None:
    """Load, validate, transform, and publish analytics tables."""
    logger.info("Starting healthcare analytics build")

    try:
        patients = load_validated_csv(
            RAW / "patients.csv",
            required_columns=PATIENT_COLUMNS,
            unique_key="Id",
        )

        encounters = load_validated_csv(
            RAW / "encounters.csv",
            required_columns=ENCOUNTER_COLUMNS,
            unique_key="Id",
        )

        observations = load_validated_csv(
            RAW / "observations.csv",
            required_columns=OBSERVATION_COLUMNS,
        )

        known_patient_ids = set(patients["Id"])

        validate_patient_references(
            encounters,
            known_patient_ids,
            "encounters",
        )

        validate_patient_references(
            observations,
            known_patient_ids,
            "observations",
        )

        patient_dim = build_patient_dimension(patients)
        vitals = build_vitals_fact(observations)
        encounter_fact = build_encounters_fact(encounters)

        # Validate foreign keys before publishing.
        known_patient_keys = set(patient_dim["patient_key"])

        for name, df in {
            "fact_vitals": vitals,
            "fact_encounters": encounter_fact,
        }.items():
            if df["patient_key"].isna().any():
                raise DataQualityError(
                    f"{name} contains missing patient keys"
                )

            unmatched = ~df["patient_key"].isin(known_patient_keys)

            if unmatched.any():
                raise DataQualityError(
                    f"{name} contains {int(unmatched.sum())} "
                    "patient keys absent from dim_patient"
                )

        # Publish all tables as one versioned snapshot.
        snapshot_dir = publish_tables(
            {
                "dim_patient.csv": patient_dim,
                "fact_vitals.csv": vitals,
                "fact_encounters.csv": encounter_fact,
            },
            PROCESSED,
        )

        logger.info(
            "Analytics build successful: patients=%d, vitals=%d, encounters=%d",
            len(patient_dim),
            len(vitals),
            len(encounter_fact),
        )

        logger.info("Published snapshot: %s", snapshot_dir)
        logger.warning(
            "Outputs use synthetic data. Patient keys are pseudonymous, "
            "not guaranteed anonymous. Encounter source IDs are retained "
            "in this development version."
        )

        logger.info(
            "The dashboard must read the completed snapshot referenced "
            "by data/processed/manifest.json."
        )

    except (FileNotFoundError, DataQualityError):
        logger.exception("Analytics build failed data-quality validation")
        raise

    except Exception:
        logger.exception("Unexpected analytics build failure")
        raise


if __name__ == "__main__":
    main()
