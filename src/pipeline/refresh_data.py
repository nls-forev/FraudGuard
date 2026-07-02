import os

from src.constants import COLLECTION_NAME, RAW_DATA_PATH
from src.data_access.fraudguard_data import FraudGuardData
from src.logger import logging


def main() -> None:
    df = FraudGuardData().export_collection_as_dataframe(COLLECTION_NAME)

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_parquet(RAW_DATA_PATH, index=False)

    logging.info(f"Snapshot written: {RAW_DATA_PATH} ({df.shape[0]:,} rows)")
    print(
        f"Snapshot written: {RAW_DATA_PATH} ({df.shape[0]:,} rows)\n"
        "Next: dvc add data/raw.parquet && git add data/raw.parquet.dvc && dvc push"
    )


if __name__ == "__main__":
    main()
