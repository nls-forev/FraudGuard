"""Detect feature drift between training-time data and live traffic.

Reference window: the champion model's held-out split, exported to S3 by the
model pusher at promotion time. Current window: recent live predictions that
the flush job landed in MongoDB. Runs Evidently's DataDriftPreset over the
schema feature columns; when the share of drifted columns crosses the
threshold it writes a ``drift.flag`` file, which the detect_drift workflow
uses to dispatch a retraining run.

Label drift is not measured: confirmed fraud labels lag live traffic by
weeks (chargebacks), so input-distribution drift is the retraining signal.

Environment overrides:
    DRIFT_WINDOW_DAYS      current-window lookback (default 7)
    DRIFT_MIN_SAMPLES      minimum current rows to attempt detection (default 100)
    DRIFT_SHARE_THRESHOLD  drifted-column share that flags drift (default 0.3)
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.configuration.mongo_connection import MongoDBClient
from src.constants import (
    BUCKET_NAME,
    DATABASE_NAME,
    DRIFT_REPORT_S3_PREFIX,
    PREDICTIONS_COLLECTION_NAME,
    SCHEMA_FILE_PATH,
)
from src.data_access.aws_sagemaker import BucketOperations
from src.logger import logging
from src.utils.main_utils import read_yaml_file

DRIFT_FLAG_PATH = Path("drift.flag")
DRIFT_SUMMARY_PATH = Path("drift_summary.json")


def feature_columns(schema: dict) -> list[str]:
    return (
        schema["numeric_columns"]
        + schema["binary_columns"]
        + schema["categorical_columns"]
    )


def load_current_window(window_days: int) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    collection = MongoDBClient(DATABASE_NAME).db[PREDICTIONS_COLLECTION_NAME]
    cursor = collection.find({"predicted_at": {"$gte": since.isoformat()}}, {"_id": 0})
    return pd.DataFrame(list(cursor))


def run_drift_check(bucket_ops: BucketOperations | None = None) -> dict:
    bucket_ops = bucket_ops or BucketOperations()

    window_days = int(os.getenv("DRIFT_WINDOW_DAYS", "7"))
    min_samples = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))
    share_threshold = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.3"))

    schema = read_yaml_file(SCHEMA_FILE_PATH.as_posix())
    columns = feature_columns(schema)

    current = load_current_window(window_days)
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "current_samples": len(current),
        "drift_detected": False,
        "drifted_columns_share": None,
        "drifted_columns_count": None,
        "share_threshold": share_threshold,
    }

    if len(current) < min_samples:
        logging.info(
            f"Only {len(current)} live predictions in the last {window_days} day(s) "
            f"(minimum {min_samples}). Skipping drift detection."
        )
        return summary

    reference = pd.read_csv(bucket_ops.download_reference_data())

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference[columns], current_data=current[columns]
    )

    drift_metric = snapshot.dict()["metrics"][0]["value"]
    share = float(drift_metric["share"])
    summary["drifted_columns_share"] = share
    summary["drifted_columns_count"] = int(drift_metric["count"])
    summary["drift_detected"] = share >= share_threshold

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_key = f"{DRIFT_REPORT_S3_PREFIX}/drift_report_{timestamp}.html"
    report_path = Path(f"drift_report_{timestamp}.html")
    snapshot.save_html(report_path.as_posix())
    bucket_ops.s3_client.upload_file(report_path.as_posix(), BUCKET_NAME, report_key)
    summary["report_s3_uri"] = f"s3://{BUCKET_NAME}/{report_key}"

    logging.info(
        f"Drift check: {summary['drifted_columns_count']} of {len(columns)} columns "
        f"drifted (share={share:.2f}, threshold={share_threshold}). "
        f"Report: {summary['report_s3_uri']}"
    )
    return summary


def main() -> None:
    summary = run_drift_check()

    DRIFT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if summary["drift_detected"]:
        DRIFT_FLAG_PATH.touch()
        logging.info("Drift detected — drift.flag written.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Drift detection failed.")
        sys.exit(1)
