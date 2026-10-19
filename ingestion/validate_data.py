
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    return pd.read_csv(path, low_memory=False)


def report_duplicates(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        print(f"[ERROR] {label}: key column '{key}' not found")
        return

    missing = int(df[key].isna().sum())
    duplicates = int(df[key].duplicated().sum())

    print(f"{label}.{key}: missing={missing}, duplicates={duplicates}")


def main() -> None:
    patients = load_csv("patients.csv")
    encounters = load_csv("encounters.csv")
    observations = load_csv("observations.csv")
    conditions = load_csv("conditions.csv")

    tables = {
        "patients": patients,
        "encounters": encounters,
        "observations": observations,
        "conditions": conditions,
    }

    print("\n=== SOURCE RECORD COUNTS ===")
    for name, df in tables.items():
        print(f"{name}: {len(df):,} rows, {len(df.columns)} columns")

    print("\n=== KEY QUALITY CHECKS ===")
    report_duplicates(patients, "Id", "patients")
    report_duplicates(encounters, "Id", "encounters")

    for name, df in {
        "encounters": encounters,
        "observations": observations,
        "conditions": conditions,
    }.items():
        report_duplicates(df, "PATIENT", f"{name} patient reference")

    print("\n=== PATIENT REFERENCES ===")
    if "Id" in patients.columns:
        known_patients = set(patients["Id"].dropna())

        for name, df in {
            "encounters": encounters,
            "observations": observations,
            "conditions": conditions,
        }.items():
            if "PATIENT" not in df.columns:
                print(f"{name}: PATIENT column missing")
                continue

            refs = df["PATIENT"]
            invalid = refs.notna() & ~refs.isin(known_patients)

            print(
                f"{name}: missing patient references={int(refs.isna().sum())}, "
                f"unmatched patient references={int(invalid.sum())}"
            )

    print("\n=== DATE PARSING CHECKS ===")
    date_columns = {
        "encounters": (encounters, ["START", "STOP"]),
        "observations": (observations, ["DATE"]),
        "conditions": (conditions, ["START", "STOP"]),
    }

    for name, (df, columns) in date_columns.items():
        for column in columns:
            if column not in df.columns:
                print(f"{name}.{column}: column missing")
                continue

            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
            invalid = df[column].notna() & parsed.isna()

            print(
                f"{name}.{column}: missing={int(df[column].isna().sum())}, "
                f"invalid dates={int(invalid.sum())}"
            )

    print("\nValidation checks completed. Source files were not modified.")


if __name__ == "__main__":
    main()
