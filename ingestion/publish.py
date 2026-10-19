
"""Publish validated analytics tables as versioned snapshots."""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


def publish_tables(
    tables: dict[str, pd.DataFrame],
    output_root: Path,
) -> Path:
    """Publish non-empty DataFrames as one versioned snapshot.

    The manifest points to the latest complete snapshot. Existing snapshots
    are retained so readers can continue using a previous version.
    """

    output_root = Path(output_root)
    versions_dir = output_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    if not tables:
        raise ValueError("No tables supplied for publishing")

    for name, df in tables.items():
        if not name.endswith(".csv"):
            raise ValueError(f"Output name must end with .csv: {name}")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} is not a pandas DataFrame")
        if df.empty:
            raise ValueError(f"Refusing to publish empty table: {name}")
        if df.columns.duplicated().any():
            raise ValueError(f"Duplicate column names in table: {name}")

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:8]
    )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=versions_dir)
    )
    final_dir = versions_dir / run_id

    try:
        table_metadata = {}

        for name, df in tables.items():
            path = staging_dir / name
            df.to_csv(path, index=False)

            # Read back to confirm the CSV is parseable and has expected rows.
            check = pd.read_csv(path, low_memory=False)
            if len(check) != len(df):
                raise ValueError(
                    f"Row-count mismatch for {name}: "
                    f"expected {len(df)}, got {len(check)}"
                )
            if list(check.columns) != list(df.columns):
                raise ValueError(f"Column mismatch after writing {name}")

            table_metadata[name] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "file": name,
            }

        # Staging and final directory are on the same filesystem.
        # The directory rename publishes the completed snapshot.
        os.replace(staging_dir, final_dir)

        manifest = {
            "run_id": run_id,
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "tables": table_metadata,
        }

        output_root.mkdir(parents=True, exist_ok=True)
        fd, temp_manifest_name = tempfile.mkstemp(
            prefix=".manifest-", suffix=".tmp", dir=output_root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)

            # Replace the manifest only after the snapshot is complete.
            os.replace(temp_manifest_name, output_root / "manifest.json")
        finally:
            if os.path.exists(temp_manifest_name):
                os.unlink(temp_manifest_name)

        logger.info(
            "Published snapshot %s with %d tables",
            run_id,
            len(tables),
        )
        return final_dir

    except Exception:
        logger.exception("Snapshot publishing failed")
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        # A completed version directory may remain if manifest publishing
        # fails. It is not the current snapshot unless the manifest points to it.
        raise
