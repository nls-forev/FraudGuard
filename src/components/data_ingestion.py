import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.fraudguard_data import FraudGuardData
from src.constants import (
    COLLECTION_NAME,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
)


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        self.data_ingestion_config = data_ingestion_config

    def export_data_as_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting dataframe from mongoDB")
            df = FraudGuardData().export_collection_as_dataframe(COLLECTION_NAME)

            logging.info(f"Shape of dataframe: {df.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.infof(f"Saved Exported data: {feature_store_file_path}")

            return df

        except Exception as e:
            raise e

    def split_train_test_split(self, df: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                df,
                test_size=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
                random_state=RANDOM_STATE,
            )
            logging.info("Train and test split!")

            dir_name = os.path.dirname(self.data_ingestion_config.training_file)
            os.makedirs(dir_name, exist_ok=True)

            logging.info("Exporting train and test set")
            train_set.to_csv(
                self.data_ingestion_config.training_file, index=False, Header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file, index=False, Header=True
            )
            logging.info(f"Exported train and test set to {dir_name}/")

        except Exception as e:
            raise e

    def init_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Initiating Data ingestion.")

        df = self.export_data_as_feature_store()
        self.split_train_test_split(df)

        logging.info("Exiting Data ingestion step.")

        data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=self.data_ingestion_config.training_file,
            test_file_path=self.data_ingestion_config.testing_file,
        )

        logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
        return data_ingestion_artifact
