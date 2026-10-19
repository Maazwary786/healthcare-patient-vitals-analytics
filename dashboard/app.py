
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare Patient Analytics")
st.caption(
    "Synthetic Synthea data | Dataset analytics demo | "
    "Not for clinical decision-making"
)


@st.cache_data
def load_data():
    required = [
        DATA_DIR / "dim_patient.csv",
        DATA_DIR / "fact_vitals.csv",
        DATA_DIR / "fact_encounters.csv",
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed files not found. Run "
            "'python ingestion/build_analytics.py' first.\n"
            + "\n".join(missing)
        )

    patients = pd.read_csv(required[0])
    vitals = pd.read_csv(required[1])
    encounters = pd.read_csv(required[2])

    vitals["event_time"] = pd.to_datetime(
        vitals["event_time"], errors="coerce", utc=True
    )
    encounters["start_time"] = pd.to_datetime(
        encounters["start_time"], errors="coerce", utc=True
    )

    vitals["numeric_value"] = pd.to_numeric(
        vitals["numeric_value"], errors="coerce"
    )
    encounters["duration_minutes"] = pd.to_numeric(
        encounters["duration_minutes"], errors="coerce"
    )

    return patients, vitals, encounters


try:
    patients, vitals, encounters = load_data()
except (FileNotFoundError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

# Top-level metrics come from the processed files.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(patients):,}")
c2.metric("Encounters", f"{len(encounters):,}")
c3.metric("Vital measurements", f"{len(vitals):,}")
c4.metric(
    "Patients with recorded vitals",
    f"{vitals['patient_key'].nunique():,}",
)

st.divider()

st.subheader("Vitals Analytics")

if vitals.empty:
    st.warning("No valid vital measurements are available.")
else:
    measurements = sorted(vitals["measurement"].dropna().unique())
    selected = st.selectbox(
        "Choose a measurement",
        measurements,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    selected_data = vitals.loc[
        (vitals["measurement"] == selected)
        & vitals["event_time"].notna()
        & vitals["numeric_value"].notna()
    ].copy()

    if selected_data.empty:
        st.info("No valid records for this measurement.")
    else:
        # Daily averages are descriptive summaries, not clinical thresholds.
        selected_data["date"] = (
            selected_data["event_time"].dt.floor("D")
        )

        daily = (
            selected_data.groupby("date", as_index=False)
            .agg(
                average_value=("numeric_value", "mean"),
                record_count=("numeric_value", "size"),
            )
        )

        fig = px.line(
            daily,
            x="date",
            y="average_value",
            markers=True,
            title=f"Daily average: {selected.replace('_', ' ').title()}",
            labels={
                "date": "Observation date",
                "average_value": "Average recorded value",
            },
        )
        st.plotly_chart(fig, width="stretch")

        a, b, c = st.columns(3)
        a.metric("Records", f"{len(selected_data):,}")
        b.metric(
            "Mean recorded value",
            f"{selected_data['numeric_value'].mean():.2f}",
        )
        c.metric(
            "Distinct patients",
            f"{selected_data['patient_key'].nunique():,}",
        )

        st.caption(
            "Measurements are historical synthetic records. "
            "The chart is not a live monitor and does not establish "
            "a patient's current condition."
        )

st.divider()

st.subheader("Encounter Analytics")

if encounters.empty:
    st.warning("No encounter records are available.")
else:
    class_counts = (
        encounters["encounter_class"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Encounter type")
        .reset_index(name="Records")
    )

    fig = px.bar(
        class_counts,
        x="Encounter type",
        y="Records",
        title="Encounters by type",
        labels={"Records": "Number of encounters"},
    )
    st.plotly_chart(fig, width="stretch")

    valid_duration = encounters["duration_minutes"].dropna()
    if not valid_duration.empty:
        st.metric(
            "Mean recorded encounter duration",
            f"{valid_duration.mean():.1f} minutes",
        )

    st.caption(
        "Encounter duration is calculated from the recorded start "
        "and stop timestamps. It is not automatically equivalent "
        "to hospital length of stay."
    )

st.divider()

st.subheader("Data Quality Snapshot")

quality = pd.DataFrame(
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

st.dataframe(quality, hide_index=True, width="stretch")

st.caption(
    "Dashboard metrics are calculated from local processed CSV files. "
    "They refresh when the app reruns and the files are reloaded."
)
