
"""Reusable CSV ingestion and data-quality utilities."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityError(ValueError):
    """Raised when source data fails a quality check."""


def load_validated_csv(
    path: Path,
    required_columns: set[str],
    unique_key: str | None = None,
) -> pd.DataFrame:
    """Load a CSV and validate its schema and optional unique key."""

    path = Path(path)

    if not path.is_file():
        logger.error("Source file not found: %s", path)
        raise FileNotFoundError(f"Source file not found: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Unable to parse source CSV: %s", path)
        raise DataQualityError(f"Invalid CSV format: {path.name}") from exc

    if df.empty:
        raise DataQualityError(f"Source CSV is empty: {path.name}")

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = sorted(missing_columns)
        logger.error("%s is missing required columns: %s", path.name, missing)
        raise DataQualityError(
            f"{path.name} is missing required columns: {missing}"
        )

    if unique_key is not None:
        if unique_key not in df.columns:
            raise DataQualityError(
                f"{path.name} does not contain unique key '{unique_key}'"
            )

        if df[unique_key].isna().any():
            raise DataQualityError(
                f"{path.name} contains null values in key '{unique_key}'"
            )

        if df[unique_key].duplicated().any():
            raise DataQualityError(
                f"{path.name} contains duplicate values in key '{unique_key}'"
            )

    logger.info(
        "Loaded %s successfully: rows=%d, columns=%d",
        path.name,
        len(df),
        len(df.columns),
    )

    return df

