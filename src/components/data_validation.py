import os
import pandera.pandas as pa
import pandas as pd

from src.logger import logging
from src.utils.main_utils import read_yaml_file

from src.entity.config_entity import DataValidationConfig, BAFSchema
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact,
)

from src.constants import (
    SCHEMA_FILE_PATH,
)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._scheme_config = read_yaml_file(SCHEMA_FILE_PATH.as_posix())

    @staticmethod
    def read_data(file_path: str):
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise e

    def validate_existence_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            expected = set(df.columns)
            actual = set(self._scheme_config["columns"])

            missing = expected - actual
            extra = actual - expected

            logging.debug(f"Missing: {missing}")
            logging.debug(f"Extra: {extra}")

            if missing or extra:
                return False

            return True

        except Exception as e:
            raise e

    def validation_schema(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating against the schema...")
            BAFSchema.validate(
                df,
                lazy=True,
            )
            logging.info("Schema validation passed")

            return True

        except pa.errors.SchemaErrors as err:
            logging.error(err)

            report = err.failure_cases

            dir_name = os.path.dirname(
                self.data_validation_config.data_validation_report
            )
            os.makedirs(dir_name, exist_ok=True)

            report.to_csv(
                self.data_validation_config.data_validation_report, index=False
            )

            logging.error(
                f"Error Validation report saved to {self.data_validation_config.data_validation_report}",
            )

            raise

    def init_data_validation(self) -> DataValidationArtifact:
        validation_msg = ""

        logging.info("Initiating Data Validation...")

        train_df = DataValidation.read_data(
            self.data_ingestion_artifact.train_file_path
        )
        test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

        logging.info("Successfully loadded train and test dataframes")
        logging.debug(
            f"Train and test dataframe shape: {train_df.shape}, {test_df.shape}"
        )

        logging.info(
            "Initiating validation of all column's existence in both the dataframes, "
        )

        train_status, test_status = (
            self.validate_existence_of_columns(train_df),
            self.validate_existence_of_columns(test_df),
        )

        if not train_status:
            validation_msg += "Columns missing in training dataframe"

        else:
            logging.info(
                "Successfully passed validation of all columns present in training dataframe"
            )

        if not test_status:
            validation_msg += "Columns missing in testing dataframe, "

        else:
            logging.info(
                "Successfully passed validation of all columns present in testing dataframe"
            )

        logging.info("Initiating schema validation of training dataframe.")
        train_status = self.validation_schema(train_df)

        logging.info("Successfully finished schema validation of training dataframe.")

        logging.info("Initiating schema validation of testing dataframe.")
        test_status = self.validation_schema(train_df)

        logging.info("Successfully finished schema validation of testing dataframe.")

        validation_status = len(validation_msg) == 0
        logging.debug(f"Validation status: {validation_status}")

        data_validation_artifact = DataValidationArtifact(
            validation_report_path=self.data_validation_config.data_validation_report,
            validation_msg=validation_msg,
            validation_status=validation_status,
        )

        logging.info("Finished Data Validation.")

        return data_validation_artifact
