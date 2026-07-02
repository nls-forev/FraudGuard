import os
import numpy as np

from xgboost import XGBClassifier

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from src.logger import logging

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig

from src.utils.main_utils import read_yaml_file, load_numpy_array_data, convert_to_tar

from src.constants import HYPERPARAMS_FILE_PATH


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self._hyperparams = read_yaml_file(HYPERPARAMS_FILE_PATH.as_posix())

    def model_train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> XGBClassifier:
        try:
            logging.info("Beginning XGBoost model training.")

            model = XGBClassifier(**self._hyperparams["params"])

            model.fit(x_train, y_train)

            logging.info("Successfully trained XGBoost model.")

            return model

        except Exception as e:
            raise e

    def save_onnx_model(
        self,
        model: XGBClassifier,
        x_train: np.ndarray,
    ) -> None:
        try:
            logging.info("Converting XGBoost model to ONNX.")

            initial_types = [
                (
                    "input",
                    FloatTensorType([None, x_train.shape[1]]),
                )
            ]

            os.makedirs(
                os.path.dirname(self.model_trainer_config.model_trainer_file_path),
                exist_ok=True,
            )

            onnx_model = convert_xgboost(
                model,
                initial_types=initial_types,
            )

            with open(
                self.model_trainer_config.model_trainer_file_path,
                "wb",
            ) as f:
                f.write(onnx_model.SerializeToString())

            logging.info("Saved ONNX model successfully.")

        except Exception as e:
            raise e

    def init_model_training(self):
        x_train = load_numpy_array_data(
            self.data_transformation_artifact.transformed_x_train_file_path,
        )

        y_train = load_numpy_array_data(
            self.data_transformation_artifact.transformed_y_train_file_path,
        )

        logging.info("Loaded X_train and y_train arrays.")

        model = self.model_train(
            x_train=x_train,
            y_train=y_train,
        )

        logging.info("Model successfully trained.")

        self.save_onnx_model(
            model=model,
            x_train=x_train,
        )

        logging.info("Saved trained model in ONNX format.")

        convert_to_tar(
            self.model_trainer_config.model_trainer_file_path,
            self.model_trainer_config.model_trainer_tar_file_path,
        )

        logging.info("Saved onnx model in tar format.")

        model_trainer_artifact = ModelTrainerArtifact(
            model_trainer_file_path=self.model_trainer_config.model_trainer_file_path,
            model_trainer_tar_file_path=self.model_trainer_config.model_trainer_tar_file_path,
            model_trainer_preprocessor_file_path=self.model_trainer_config.model_trainer_preprocessor_file_path,
        )

        return model_trainer_artifact
