
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def inspect_csv(file_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    try:
        df = pd.read_csv(file_path, nrows=100_000, low_memory=False)

        print(f"Rows inspected: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumn names and data types:")
        print(df.dtypes.to_string())

        print("\nFirst 3 records:")
        print(df.head(3).to_string())

        print("\nMissing values:")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        print(missing.to_string() if not missing.empty else "No missing values found")

        print("\nDuplicate rows in inspected sample:")
        print(df.duplicated().sum())

        print("\nPotential ID columns:")
        id_columns = [
            column for column in df.columns
            if "id" in column.lower()
        ]
        print(id_columns if id_columns else "No obvious ID columns found")

    except Exception as exc:
        print(f"ERROR inspecting {file_path.name}: {exc}")

def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        return

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print(f"No CSV files found in: {DATA_DIR}")
        return

    for file_path in files:
        inspect_csv(file_path)

if __name__ == "__main__":
    main()
