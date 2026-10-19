
import json

import pandas as pd
import pytest

from ingestion.publish import publish_tables


def test_publish_creates_snapshot_and_manifest(tmp_path):
    tables = {
        "dim_patient.csv": pd.DataFrame(
            {"patient_key": ["p1", "p2"], "gender": ["F", "M"]}
        ),
        "fact_vitals.csv": pd.DataFrame(
            {"patient_key": ["p1"], "value": [72]}
        ),
    }

    snapshot = publish_tables(tables, tmp_path)

    assert snapshot.is_dir()
    assert (snapshot / "dim_patient.csv").is_file()
    assert (snapshot / "fact_vitals.csv").is_file()

    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == snapshot.name
    assert manifest["tables"]["dim_patient.csv"]["rows"] == 2


def test_publish_rejects_empty_table(tmp_path):
    tables = {"fact_vitals.csv": pd.DataFrame(columns=["value"])}

    with pytest.raises(ValueError, match="empty table"):
        publish_tables(tables, tmp_path)

    assert not (tmp_path / "manifest.json").exists()


def test_publish_rejects_invalid_output_name(tmp_path):
    tables = {"fact_vitals.txt": pd.DataFrame({"value": [72]})}

    with pytest.raises(ValueError, match="must end with .csv"):
        publish_tables(tables, tmp_path)


def test_failed_publish_does_not_replace_previous_manifest(tmp_path):
    publish_tables(
        {"fact_vitals.csv": pd.DataFrame({"value": [72]})},
        tmp_path,
    )
    previous_manifest = (tmp_path / "manifest.json").read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="empty table"):
        publish_tables(
            {"fact_vitals.csv": pd.DataFrame(columns=["value"])},
            tmp_path,
        )

    assert (tmp_path / "manifest.json").read_text(
        encoding="utf-8"
    ) == previous_manifest
