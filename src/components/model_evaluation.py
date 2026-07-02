import os
import numpy as np
import json
import onnxruntime as ort

from src.logger import logging

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

from src.utils.main_utils import load_numpy_array_data


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_artifact: ModelTrainerArtifact,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_artifact = model_training_artifact
        self.model_evaluation_config = model_evaluation_config

    def evaluate_model(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Beginning model evaluation")

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            metrics = {
                "Accuracy": accuracy,
                "f1 score": f1,
                "Precision": precision,
                "Recall": recall,
            }

            logging.debug(metrics)

            os.makedirs(
                os.path.dirname(
                    self.model_evaluation_config.model_evaluation_metrics_file_path
                ),
                exist_ok=True,
            )

            with open(
                self.model_evaluation_config.model_evaluation_metrics_file_path, "w"
            ) as f:
                json.dump(metrics, f)

            logging.debug(
                f"Metrics successfully dumped at path: {self.model_evaluation_config.model_evaluation_metrics_file_path}"
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                accuracy=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                model_evaluation_metrics_file_path=self.model_evaluation_config.model_evaluation_metrics_file_path,
            )

            return model_evaluation_artifact

        except Exception as e:
            raise e

    def load_onnx_model(self) -> ort.InferenceSession:
        try:
            model = ort.InferenceSession(
                self.model_training_artifact.model_trainer_file_path,
            )

            return model

        except Exception as e:
            raise e

    def generate_y_pred(
        self, model: ort.InferenceSession, x_test: np.ndarray
    ) -> np.ndarray:
        try:
            y_pred = model.run(
                None,
                {"input": x_test.astype("float32")},
            )[0]

            y_pred = np.asarray(y_pred)
            if np.issubdtype(y_pred.dtype, np.floating):
                y_pred = (y_pred >= 0.5).astype(int)

            return y_pred

        except Exception as e:
            raise e

    def init_model_evaluation(self) -> ModelEvaluationArtifact:
        x_test = load_numpy_array_data(
            self.data_transformation_artifact.transformed_x_test_file_path,
        )
        y_test = load_numpy_array_data(
            self.data_transformation_artifact.transformed_y_test_file_path,
        )
        logging.info("Loaded X test and Y test")

        model = self.load_onnx_model()
        y_pred = self.generate_y_pred(model, x_test)
        logging.info("Generated model predictions")

        model_evaluation_artifact = self.evaluate_model(y_test, y_pred)
        logging.info("Generated model evaluation artifact.")

        return model_evaluation_artifact
