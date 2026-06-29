import numpy as np
import pandas as pd

from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.entity.config_entity import DataTransformationConfig

from src.constants import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file, save_object, save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema = read_yaml_file(SCHEMA_FILE_PATH.as_posix())

            self.SENTINEL_COLS = self._schema["sentinel_columns"]
            self.LOG1P_COLS = self._schema["log1p_columns"]
            self.TARGET = self._schema["target"]
            self.BINARY_COLS = self._schema["binary_columns"]
            self.CATEGORICAL_COLS = self._schema["categorical_columns"]
            self.NUMERIC_COLS = self._schema["numeric_columns"]

        except Exception as e:
            raise e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def clean_baf_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        logging.info("Initializing cleaning and feature engineering.")

        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                if col == "device_os":
                    df[col] = df[col].astype(str).str.lower().str.strip()
                else:
                    df[col] = df[col].astype(str).str.upper().str.strip()

        if self.TARGET in df.columns:
            df[self.TARGET] = df[self.TARGET].astype(int)

        for col in self.BINARY_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in self.SENTINEL_COLS:
            if col in df.columns:
                df[f"{col}_missing"] = (df[col] == -1).astype("Int64")

        for col in self.SENTINEL_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] == -1, col] = np.nan

        for col in self.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.LOG1P_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        logging.info("Done cleaning and feature engineering.")
        return df

    def build_preprocessor(self):
        logging.info("Creating preprocessing pipeline.")

        try:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,  # for older sklearn versions  # pyright: ignore[reportCallIssue]
            )

        numeric_features = self.NUMERIC_COLS + [
            f"{c}_missing" for c in self.SENTINEL_COLS
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("bin", binary_pipeline, self.BINARY_COLS),
                ("cat", categorical_pipeline, self.CATEGORICAL_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        logging.info("Created preprocessing pipeline.")
        return preprocessor

    def init_data_transformation(self) -> DataTransformationArtifact:
        if not self.data_validation_artifact.validation_status:
            raise Exception(self.data_validation_artifact.validation_msg)

        train_df = DataTransformation.read_data(
            self.data_ingestion_artifact.train_file_path
        )
        test_df = DataTransformation.read_data(
            self.data_ingestion_artifact.test_file_path
        )
        logging.info("Train and test dataframes loaded.")

        train_df = self.clean_baf_dataframe(train_df)
        test_df = self.clean_baf_dataframe(test_df)
        logging.info("Train and test dataframes cleaned.")

        X_train = train_df.drop(columns=[self.TARGET])
        y_train = train_df[self.TARGET]

        X_test = test_df.drop(columns=[self.TARGET])
        y_test = test_df[self.TARGET]

        logging.info("Separated X and y for train and test dataframes.")

        preprocessor = self.build_preprocessor()
        logging.info("Created preprocessor object.")

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        logging.info("Transformed train and test feature matrices.")

        train_arr = np.c_[X_train_transformed, np.array(y_train)]
        test_arr = np.c_[X_test_transformed, np.array(y_test)]
        logging.info("Concatenated transformed X and y arrays.")

        save_object(
            self.data_transformation_config.preprocessor_file_path, preprocessor
        )
        save_numpy_array_data(
            self.data_transformation_config.transformed_train_file_path, array=train_arr
        )
        save_numpy_array_data(
            self.data_transformation_config.transformed_test_file_path, array=test_arr
        )
        logging.info("Saved transformation object and transformed arrays.")

        return DataTransformationArtifact(
            transformed_preprocessor_file_path=self.data_transformation_config.preprocessor_file_path,
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
        )
