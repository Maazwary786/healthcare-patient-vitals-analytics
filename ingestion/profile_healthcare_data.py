
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_csv(name):
    return pd.read_csv(DATA_DIR / name, low_memory=False)


def main():
    patients = load_csv("patients.csv")
    encounters = load_csv("encounters.csv")
    observations = load_csv("observations.csv")
    conditions = load_csv("conditions.csv")

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Patients:     {len(patients):,}")
    print(f"Encounters:   {len(encounters):,}")
    print(f"Observations: {len(observations):,}")
    print(f"Conditions:   {len(conditions):,}")

    print("\n========== ENCOUNTER TYPES ==========")
    print(encounters["ENCOUNTERCLASS"].value_counts(dropna=False).to_string())

    print("\n========== TOP OBSERVATION CATEGORIES ==========")
    print(observations["CATEGORY"].value_counts(dropna=False).to_string())

    print("\n========== MOST COMMON OBSERVATIONS ==========")
    print(
        observations["DESCRIPTION"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\n========== OBSERVATION VALUE TYPES ==========")
    print(observations["TYPE"].value_counts(dropna=False).to_string())

    print("\n========== POTENTIAL VITAL SIGN RECORDS ==========")
    keywords = (
        "oxygen saturation",
        "heart rate",
        "heart rate",
        "respiratory rate",
        "body temperature",
        "blood pressure",
    )

    description = observations["DESCRIPTION"].fillna("").str.lower()
    mask = description.apply(
        lambda value: any(keyword in value for keyword in keywords)
    )

    vital_candidates = observations.loc[
        mask, ["DATE", "PATIENT", "DESCRIPTION", "VALUE", "UNITS", "TYPE"]
    ]

    print(f"Candidate records: {len(vital_candidates):,}")
    print(vital_candidates.head(20).to_string(index=False))

    print("\n========== CONDITION SUMMARY ==========")
    print(
        conditions["DESCRIPTION"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nNote: these are synthetic dataset statistics, not real hospital KPIs.")


if __name__ == "__main__":
    main()
