
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Keep only the fields needed for this analytics module.
SAFE_PATIENT_COLUMNS = [
    "Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"
]

VITALS = {
    "heart rate": "heart_rate",
    "respiratory rate": "respiratory_rate",
    "systolic blood pressure": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
}


def patient_key(value):
    if pd.isna(value):
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(RAW / "patients.csv", low_memory=False)
    encounters = pd.read_csv(RAW / "encounters.csv", low_memory=False)
    observations = pd.read_csv(RAW / "observations.csv", low_memory=False)

    # Pseudonymous identifier; this is not guaranteed anonymization.
    patient_dim = patients[SAFE_PATIENT_COLUMNS].copy()
    patient_dim["patient_key"] = patient_dim["Id"].map(patient_key)
    patient_dim = patient_dim.drop(columns=["Id"])

    # Do not export names, SSNs, addresses, passport numbers or raw IDs.
    patient_dim.to_csv(PROCESSED / "dim_patient.csv", index=False)

    # Normalize selected measurements by description and numeric value.
    obs = observations.copy()
    obs["measurement"] = (
        obs["DESCRIPTION"].str.strip().str.lower().map(VITALS)
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
            "event_time", "patient_key", "ENCOUNTER",
            "measurement", "numeric_value", "UNITS"
        ],
    ].copy()

    vitals = vitals.rename(columns={
        "ENCOUNTER": "source_encounter_id",
        "UNITS": "unit",
    })

    vitals.to_csv(PROCESSED / "fact_vitals.csv", index=False)

    # A simple encounter analytics table. Do not export patient raw IDs.
    encounters["patient_key"] = encounters["PATIENT"].map(patient_key)
    encounters["START"] = pd.to_datetime(
        encounters["START"], errors="coerce", utc=True
    )
    encounters["STOP"] = pd.to_datetime(
        encounters["STOP"], errors="coerce", utc=True
    )
    encounters["duration_minutes"] = (
        encounters["STOP"] - encounters["START"]
    ).dt.total_seconds() / 60

    encounter_analytics = encounters[
        [
            "Id", "patient_key", "START", "STOP",
            "ENCOUNTERCLASS", "DESCRIPTION",
            "duration_minutes", "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
        ]
    ].rename(columns={
        "Id": "source_encounter_id",
        "START": "start_time",
        "STOP": "end_time",
        "ENCOUNTERCLASS": "encounter_class",
        "DESCRIPTION": "encounter_description",
        "BASE_ENCOUNTER_COST": "base_encounter_cost",
        "TOTAL_CLAIM_COST": "total_claim_cost",
    })

    encounter_analytics.to_csv(
        PROCESSED / "fact_encounters.csv", index=False
    )

    print("Analytics files created in:", PROCESSED)
    print("Patient dimension rows:", len(patient_dim))
    print("Vitals rows:", len(vitals))
    print("Encounter rows:", len(encounter_analytics))
    print("\nVitals by measurement:")
    print(vitals["measurement"].value_counts().to_string())
    print("\nThese outputs are derived from synthetic data.")


if __name__ == "__main__":
    main()
