
"""Build analytics tables from validated synthetic healthcare data."""

import hashlib
import logging
from pathlib import Path

import pandas as pd

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
    "Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"
}
ENCOUNTER_COLUMNS = {
    "Id", "PATIENT", "START", "STOP", "ENCOUNTERCLASS",
    "DESCRIPTION", "BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST",
}
OBSERVATION_COLUMNS = {
    "DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION",
    "VALUE", "UNITS",
}

VITALS = {
    "heart rate": "heart_rate",
    "respiratory rate": "respiratory_rate",
    "systolic blood pressure": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
}


def patient_key(value):
    """Create a deterministic pseudonymous key, not guaranteed anonymization."""
    if pd.isna(value):
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def main():
    logger.info("Starting healthcare analytics build")

    PROCESSED.mkdir(parents=True, exist_ok=True)

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

        # Referential integrity: every non-null patient reference must exist.
        known_patient_ids = set(patients["Id"])
        for name, df in {
            "encounters": encounters,
            "observations": observations,
        }.items():
            unmatched = (
                df["PATIENT"].notna()
                & ~df["PATIENT"].isin(known_patient_ids)
            )
            if unmatched.any():
                raise DataQualityError(
                    f"{name} has {int(unmatched.sum())} unknown patient references"
                )

        # Build patient dimension with limited fields.
        patient_dim = patients[
            ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]
        ].copy()
        patient_dim["patient_key"] = patient_dim["Id"].map(patient_key)
        patient_dim = patient_dim.drop(columns=["Id"])

        # Normalize selected vital measurements.
        obs = observations.copy()
        obs["measurement"] = (
            obs["DESCRIPTION"].astype("string").str.strip().str.lower().map(VITALS)
        )
        obs["numeric_value"] = pd.to_numeric(obs["VALUE"], errors="coerce")
        obs["event_time"] = pd.to_datetime(
            obs["DATE"], errors="coerce", utc=True
        )
        obs["patient_key"] = obs["PATIENT"].map(patient_key)

        vitals = obs.loc[
            obs["measurement"].notna()
            & obs["numeric_value"].notna()
            & obs["event_time"].notna(),
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

        # Build encounter fact table.
        encounter_analytics = encounters.copy()
        encounter_analytics["patient_key"] = (
            encounter_analytics["PATIENT"].map(patient_key)
        )
        encounter_analytics["START"] = pd.to_datetime(
            encounter_analytics["START"], errors="coerce", utc=True
        )
        encounter_analytics["STOP"] = pd.to_datetime(
            encounter_analytics["STOP"], errors="coerce", utc=True
        )
        encounter_analytics["duration_minutes"] = (
            encounter_analytics["STOP"] - encounter_analytics["START"]
        ).dt.total_seconds() / 60

        encounter_analytics = encounter_analytics[
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

        # Validate generated tables before writing.
        if patient_dim["patient_key"].isna().any():
            raise DataQualityError("Patient dimension contains missing patient keys")

        if vitals["event_time"].isna().any():
            raise DataQualityError("Vitals fact contains invalid event timestamps")

        if encounter_analytics["source_encounter_id"].duplicated().any():
            raise DataQualityError("Encounter fact contains duplicate encounter IDs")

        # Write only after source checks and transformations complete.
        patient_dim.to_csv(PROCESSED / "dim_patient.csv", index=False)
        vitals.to_csv(PROCESSED / "fact_vitals.csv", index=False)
        encounter_analytics.to_csv(
            PROCESSED / "fact_encounters.csv", index=False
        )

        logger.info(
            "Analytics build successful: patients=%d, vitals=%d, encounters=%d",
            len(patient_dim),
            len(vitals),
            len(encounter_analytics),
        )
        logger.info("Output directory: %s", PROCESSED)
        logger.warning(
            "Outputs use synthetic data. Patient IDs are pseudonymized, "
            "not guaranteed anonymized."
        )

    except (FileNotFoundError, DataQualityError):
        logger.exception("Analytics build failed data-quality validation")
        raise
    except Exception:
        logger.exception("Unexpected analytics build failure")
        raise


if __name__ == "__main__":
    main()
