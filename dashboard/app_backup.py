
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Healthcare Patient Analytics",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Healthcare Patient Vitals Analytics")
st.caption(
    "Capstone Project | Real-Time Monitoring & Data Engineering"
)

# Demo data only — Kafka/Spark se connected nahi hai abhi
rng = np.random.default_rng(42)

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Admissions", "1,248", "+8.2%")
c2.metric("Bed Occupancy", "76.4%", "+2.1%")
c3.metric("Active Alerts", "7", "-2")
c4.metric("Pipeline Status", "Demo Mode", "Not connected")

st.divider()

# Simulated vital-sign data
times = pd.date_range(
    end=pd.Timestamp.now(),
    periods=60,
    freq="min",
)

spo2 = np.clip(
    96 + rng.normal(0, 1.2, len(times)),
    85,
    100,
)

heart_rate = np.clip(
    75 + rng.normal(0, 8, len(times)),
    45,
    150,
)

df = pd.DataFrame({
    "timestamp": times,
    "spo2": spo2,
    "heart_rate": heart_rate,
})

left, right = st.columns(2)

with left:
    st.subheader("SpO₂ Trend")

    fig = px.line(
        df,
        x="timestamp",
        y="spo2",
        title="Oxygen Saturation (%)",
        labels={
            "timestamp": "Time",
            "spo2": "SpO₂ (%)",
        },
    )

    fig.add_hline(
        y=90,
        line_dash="dash",
        annotation_text="90% reference threshold",
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Heart Rate Trend")

    fig = px.line(
        df,
        x="timestamp",
        y="heart_rate",
        title="Heart Rate (bpm)",
        labels={
            "timestamp": "Time",
            "heart_rate": "Heart Rate (bpm)",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Latest Vital Sign Readings")
st.dataframe(
    df.tail(10).sort_values(
        "timestamp",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Prototype dashboard: readings and KPIs are simulated. "
    "Kafka, Spark, hospital datasets, and alert APIs "
    "have not been connected yet."
)
