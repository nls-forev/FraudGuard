import os
import pandera.pandas as pa

from dataclasses import dataclass
from datetime import datetime

from pandera.typing import Series


from src.constants import (
    PIPELINE_NAME,
    ARTIFACT_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    FILE_NAME,
    DATA_INGESTION_INGESTED_DIR,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    DATA_VALIDATION_DIR,
    DATA_VALIDATION_REPORT,
    DATA_TRANSFORMATION_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_DIR,
    DATA_TRANSFORMATION_PREPROCESSOR_DIR,
    PREPROCESSOR_FILE_NAME,
)

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR_NAME,
    )
    feature_store_file: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_FEATURE_STORE_DIR,
        FILE_NAME,
    )
    training_file: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_INGESTED_DIR,
        TRAIN_FILE_NAME,
    )
    testing_file: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_INGESTED_DIR,
        TEST_FILE_NAME,
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR,
    )
    data_validation_report: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_REPORT,
    )


class BAFSchema(pa.DataFrameModel):
    # Target
    fraud_bool: Series[int] = pa.Field(isin=[0, 1])

    # Continuous / Numeric
    income: Series[float] = pa.Field(ge=0.1, le=0.9)
    name_email_similarity: Series[float] = pa.Field(ge=0.0, le=1.0)

    prev_address_months_count: Series[int] = pa.Field(ge=-1, le=383)
    current_address_months_count: Series[int] = pa.Field(ge=-1, le=428)

    customer_age: Series[int] = pa.Field(isin=[10, 20, 30, 40, 50, 60, 70, 80, 90])

    intended_balcon_amount: Series[float] = pa.Field(
        ge=-15.530554840076814,
        le=112.9569276953714,
    )

    zip_count_4w: Series[int] = pa.Field(ge=1, le=6700)

    velocity_6h: Series[float] = pa.Field(ge=-170.60307235124628, le=16715.565404174275)
    velocity_24h: Series[float] = pa.Field(ge=1300.3073144849477, le=9506.896596111665)
    velocity_4w: Series[float] = pa.Field(ge=2825.748405284728, le=6994.764200834217)

    bank_branch_count_8w: Series[int] = pa.Field(ge=0, le=2385)
    date_of_birth_distinct_emails_4w: Series[int] = pa.Field(ge=0, le=39)

    credit_risk_score: Series[int] = pa.Field(ge=-170, le=389)

    bank_months_count: Series[int] = pa.Field(ge=-1, le=32)

    proposed_credit_limit: Series[float] = pa.Field(ge=190.0, le=2100.0)

    session_length_in_minutes: Series[float] = pa.Field(ge=-1.0, le=85.89914319274027)

    device_distinct_emails_8w: Series[int] = pa.Field(ge=-1, le=2)

    month: Series[int] = pa.Field(ge=0, le=7)

    # Binary
    email_is_free: Series[int] = pa.Field(isin=[0, 1])
    phone_home_valid: Series[int] = pa.Field(isin=[0, 1])
    phone_mobile_valid: Series[int] = pa.Field(isin=[0, 1])
    has_other_cards: Series[int] = pa.Field(isin=[0, 1])
    foreign_request: Series[int] = pa.Field(isin=[0, 1])
    keep_alive_session: Series[int] = pa.Field(isin=[0, 1])

    # Categorical
    payment_type: Series[str] = pa.Field(isin=["AA", "AB", "AC", "AD", "AE"])

    employment_status: Series[str] = pa.Field(
        isin=["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
    )

    housing_status: Series[str] = pa.Field(
        isin=["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    )

    source: Series[str] = pa.Field(isin=["INTERNET", "TELEAPP"])

    device_os: Series[str] = pa.Field(
        isin=["windows", "macintosh", "linux", "x11", "other"]
    )


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
    )

    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy"),
    )

    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DIR,
        TEST_FILE_NAME.replace("csv", "npy"),
    )

    preprocessor_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_PREPROCESSOR_DIR,
        PREPROCESSOR_FILE_NAME,
    )
