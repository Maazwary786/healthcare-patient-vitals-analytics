
"""End-to-end tests for the healthcare analytics publishing pipeline."""

import json

import pandas as pd
import pytest

import ingestion.build_analytics as builder


@pytest.fixture
def published_snapshot(tmp_path, monkeypatch):
    """Run the real pipeline and publish into an isolated test directory."""
    output_dir = tmp_path / "processed"
    monkeypatch.setattr(builder, "PROCESSED", output_dir)

    builder.main()

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_dir = output_dir / "versions" / manifest["run_id"]

    return manifest, snapshot_dir


def test_pipeline_publishes_complete_snapshot(published_snapshot):
    manifest, snapshot_dir = published_snapshot

    assert manifest["status"] == "complete"
    assert snapshot_dir.is_dir()

    expected_files = {
        "dim_patient.csv",
        "fact_vitals.csv",
        "fact_encounters.csv",
    }
    assert expected_files.issubset(manifest["tables"])

    for filename in expected_files:
        assert (snapshot_dir / filename).is_file()


def test_manifest_row_counts_match_published_files(published_snapshot):
    manifest, snapshot_dir = published_snapshot

    for filename, metadata in manifest["tables"].items():
        df = pd.read_csv(snapshot_dir / filename, low_memory=False)
        assert len(df) == metadata["rows"]
        assert list(df.columns) == metadata["columns"]


def test_fact_patient_keys_match_dimension(published_snapshot):
    _, snapshot_dir = published_snapshot

    patients = pd.read_csv(snapshot_dir / "dim_patient.csv")
    encounters = pd.read_csv(
        snapshot_dir / "fact_encounters.csv",
        low_memory=False,
    )
    vitals = pd.read_csv(
        snapshot_dir / "fact_vitals.csv",
        low_memory=False,
    )

    known_keys = set(patients["patient_key"])

    assert patients["patient_key"].notna().all()
    assert patients["patient_key"].is_unique
    assert encounters["patient_key"].notna().all()
    assert vitals["patient_key"].notna().all()
    assert set(encounters["patient_key"]).issubset(known_keys)
    assert set(vitals["patient_key"]).issubset(known_keys)


def test_pipeline_does_not_publish_empty_tables(published_snapshot):
    manifest, snapshot_dir = published_snapshot

    for filename in manifest["tables"]:
        df = pd.read_csv(snapshot_dir / filename, low_memory=False)
        assert not df.empty
