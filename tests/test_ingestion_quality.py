
import logging

import pandas as pd
import pytest

from ingestion.quality import DataQualityError, load_validated_csv


def test_load_valid_csv(tmp_path):
    path = tmp_path / "patients.csv"
    pd.DataFrame(
        {"Id": ["p1", "p2"], "BIRTHDATE": ["2000-01-01", "1995-05-10"]}
    ).to_csv(path, index=False)

    result = load_validated_csv(
        path,
        required_columns={"Id", "BIRTHDATE"},
        unique_key="Id",
    )

    assert len(result) == 2
    assert result["Id"].is_unique


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_validated_csv(
            tmp_path / "missing.csv",
            required_columns={"Id"},
        )


def test_missing_required_column_fails(tmp_path):
    path = tmp_path / "patients.csv"
    pd.DataFrame({"Id": ["p1"]}).to_csv(path, index=False)

    with pytest.raises(DataQualityError, match="missing required columns"):
        load_validated_csv(
            path,
            required_columns={"Id", "BIRTHDATE"},
        )


def test_duplicate_key_fails(tmp_path):
    path = tmp_path / "patients.csv"
    pd.DataFrame({"Id": ["p1", "p1"]}).to_csv(path, index=False)

    with pytest.raises(DataQualityError, match="duplicate"):
        load_validated_csv(
            path,
            required_columns={"Id"},
            unique_key="Id",
        )


def test_null_key_fails(tmp_path):
    path = tmp_path / "patients.csv"
    pd.DataFrame({"Id": ["p1", None]}).to_csv(path, index=False)

    with pytest.raises(DataQualityError, match="null values"):
        load_validated_csv(
            path,
            required_columns={"Id"},
            unique_key="Id",
        )


def test_empty_csv_fails(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Id,BIRTHDATE\n", encoding="utf-8")

    with pytest.raises(DataQualityError, match="empty"):
        load_validated_csv(
            path,
            required_columns={"Id", "BIRTHDATE"},
        )


def test_successful_load_is_logged(tmp_path, caplog):
    path = tmp_path / "patients.csv"
    pd.DataFrame({"Id": ["p1"]}).to_csv(path, index=False)

    with caplog.at_level(logging.INFO):
        load_validated_csv(path, required_columns={"Id"})

    assert "Loaded patients.csv successfully" in caplog.text
