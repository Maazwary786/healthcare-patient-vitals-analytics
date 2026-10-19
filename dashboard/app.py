
"""Healthcare Patient Analytics dashboard using published snapshots."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED / "manifest.json"

EXPECTED_TABLES = {
    "dim_patient.csv",
    "fact_vitals.csv",
    "fact_encounters.csv",
}


@st.cache_data
def load_snapshot(manifest_path: str, processed_dir: str):
    """Load all dashboard tables from one completed snapshot."""
    manifest_file = Path(manifest_path)
    processed_root = Path(processed_dir)

    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Published manifest not found: {manifest_file}"
        )

    with manifest_file.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("status") != "complete":
        raise ValueError("The published snapshot is not marked complete.")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Manifest does not contain a valid run_id.")

    # Prevent path traversal through an unexpected manifest value.
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Manifest contains an invalid run_id.")

    tables_metadata = manifest.get("tables", {})
    if not EXPECTED_TABLES.issubset(tables_metadata):
        raise ValueError("Manifest is missing one or more required tables.")

    snapshot_dir = processed_root / "versions" / run_id

    loaded = {}
    for filename in EXPECTED_TABLES:
        metadata = tables_metadata[filename]
        file_path = snapshot_dir / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Snapshot {run_id} is missing {filename}"
            )

        df = pd.read_csv(file_path, low_memory=False)

        expected_rows = metadata.get("rows")
        if expected_rows is not None and len(df) != expected_rows:
            raise ValueError(
                f"Row count mismatch for {filename}: "
                f"expected {expected_rows}, got {len(df)}"
            )

        expected_columns = metadata.get("columns")
        if expected_columns is not None and list(df.columns) != expected_columns:
            raise ValueError(f"Column mismatch for {filename}")

        loaded[filename] = df

    return run_id, loaded


st.set_page_config(
    page_title="Healthcare Patient Analytics",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare Patient Analytics")
st.caption(
    "Synthetic Synthea data | Dataset analytics demo | "
    "Not for clinical decision-making"
)

try:
    run_id, tables = load_snapshot(
        str(MANIFEST_PATH),
        str(PROCESSED),
    )
except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load a valid published snapshot: {exc}")
    st.info(
        "Run `python -m ingestion.build_analytics` to publish "
        "a validated analytics snapshot."
    )
    st.stop()

patients = tables["dim_patient.csv"]
vitals = tables["fact_vitals.csv"]
encounters = tables["fact_encounters.csv"]

# Validate required fields used by the dashboard.
required_columns = {
    "patients": {"patient_key"},
    "vitals": {
        "event_time", "patient_key", "measurement",
        "numeric_value", "unit",
    },
    "encounters": {
        "patient_key", "start_time", "end_time",
        "encounter_class", "duration_minutes",
    },
}

for label, df in {
    "patients": patients,
    "vitals": vitals,
    "encounters": encounters,
}.items():
    missing = required_columns[label] - set(df.columns)
    if missing:
        st.error(f"{label} snapshot is missing columns: {sorted(missing)}")
        st.stop()

st.caption(f"Published snapshot: `{run_id}`")

# Normalize data types for plotting.
vitals["event_time"] = pd.to_datetime(
    vitals["event_time"], errors="coerce", utc=True
)
vitals["numeric_value"] = pd.to_numeric(
    vitals["numeric_value"], errors="coerce"
)
encounters["start_time"] = pd.to_datetime(
    encounters["start_time"], errors="coerce", utc=True
)
encounters["end_time"] = pd.to_datetime(
    encounters["end_time"], errors="coerce", utc=True
)
encounters["duration_minutes"] = pd.to_numeric(
    encounters["duration_minutes"], errors="coerce"
)

# Executive overview.
col1, col2, col3, col4 = st.columns(4)

col1.metric("Patients", f"{len(patients):,}")
col2.metric("Encounters", f"{len(encounters):,}")
col3.metric("Vital measurements", f"{len(vitals):,}")
col4.metric(
    "Patients with recorded vitals",
    f"{vitals['patient_key'].nunique():,}",
)

st.divider()

# Vitals analytics.
st.header("Vitals Analytics")

measurements = sorted(
    vitals["measurement"].dropna().unique().tolist()
)

if measurements:
    selected = st.selectbox(
        "Choose a measurement",
        measurements,
        index=measurements.index("heart_rate")
        if "heart_rate" in measurements
        else 0,
    )

    selected_vitals = vitals.loc[
        vitals["measurement"] == selected
    ].copy()

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Records", f"{len(selected_vitals):,}")
    metric2.metric(
        "Mean recorded value",
        f"{selected_vitals['numeric_value'].mean():.2f}",
    )
    metric3.metric(
        "Distinct patients",
        f"{selected_vitals['patient_key'].nunique():,}",
    )

    st.caption(
        "These are historical synthetic records, not live monitoring "
        "and not a clinical assessment."
    )

    selected_vitals = selected_vitals.dropna(
        subset=["event_time", "numeric_value"]
    )

    if not selected_vitals.empty:
        date_values = selected_vitals["event_time"].dt.date
        min_date = min(date_values)
        max_date = max(date_values)

        start_date, end_date = st.date_input(
            "Filter date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        selected_vitals = selected_vitals.loc[
            selected_vitals["event_time"].dt.date.between(
                start_date, end_date
            )
        ]

        granularity = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
        )

        if not selected_vitals.empty:
            frequency = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "MS",
            }[granularity]

            trend = (
                selected_vitals.set_index("event_time")
                .resample(frequency)["numeric_value"]
                .mean()
                .reset_index()
            )

            fig = px.line(
                trend,
                x="event_time",
                y="numeric_value",
                markers=True,
                title=f"{granularity} average: {selected}",
                labels={
                    "event_time": "Date",
                    "numeric_value": "Average recorded value",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No records found for the selected date range.")
else:
    st.warning("No supported vital measurements are available.")

st.divider()

# Encounter analytics.
st.header("Encounter Analytics")

if not encounters.empty:
    encounter_counts = (
        encounters["encounter_class"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Encounter type")
        .reset_index(name="Records")
    )

    fig = px.bar(
        encounter_counts,
        x="Encounter type",
        y="Records",
        title="Encounters by type",
    )
    st.plotly_chart(fig, use_container_width=True)

    valid_durations = encounters.loc[
        encounters["duration_minutes"].notna()
        & (encounters["duration_minutes"] >= 0),
        "duration_minutes",
    ]

    if not valid_durations.empty:
        st.metric(
            "Mean recorded encounter duration",
            f"{valid_durations.mean():,.1f} minutes",
        )

    st.caption(
        "Recorded encounter duration is not automatically equivalent "
        "to hospital length of stay."
    )

st.divider()

# Data quality snapshot.
st.header("Data Quality Snapshot")

quality_snapshot = pd.DataFrame(
    [
        {
            "Dataset": "Patient dimension",
            "Rows": len(patients),
            "Columns": len(patients.columns),
        },
        {
            "Dataset": "Vitals fact table",
            "Rows": len(vitals),
            "Columns": len(vitals.columns),
        },
        {
            "Dataset": "Encounter fact table",
            "Rows": len(encounters),
            "Columns": len(encounters.columns),
        },
    ]
)

st.dataframe(quality_snapshot, hide_index=True, use_container_width=True)

st.caption(
    "Dashboard data comes from one completed versioned snapshot. "
    "All source records are synthetic and must not be used for "
    "clinical decisions."
)
