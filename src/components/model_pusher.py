import json
import os
from datetime import datetime

from src.data_access.aws_sagemaker import BucketOperations

from src.constants import (
    CHAMPION_METRIC_PATH,
    CHAMPION_MODEL_PATH,
    CHAMPION_PREPROCESSOR_PATH,
    SAGEMAKER_MODEL_NAME_PREFIX,
)
from src.entity.artifact_entity import (
    ModelCompareArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelPusherConfig
from src.logger import logging
from src.utils.main_utils import convert_to_tar


class ModelPusher:
    def __init__(
        self,
        model_compare_artifact: ModelCompareArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
        bucket_ops: BucketOperations = BucketOperations(),
    ):
        self.model_compare_artifact = model_compare_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.bucket_ops = bucket_ops
        self.decision_file_path = (
            model_compare_artifact.model_compare_decision_file_path
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.s3_model_key = CHAMPION_MODEL_PATH.format(timestamp=self.timestamp)
        self.s3_metric_key = CHAMPION_METRIC_PATH.format(timestamp=self.timestamp)
        self.s3_preprocessor_key = CHAMPION_PREPROCESSOR_PATH.format(
            timestamp=self.timestamp
        )
        self.tar_output_path = model_trainer_artifact.model_trainer_tar_file_path
        self.preprocessor_path = (
            model_trainer_artifact.model_trainer_preprocessor_file_path
        )

    def compress_model(self):
        try:
            if os.path.exists(self.tar_output_path):
                logging.info(f"Model tar already exists at: {self.tar_output_path}")
                return

            logging.info("Packaging model artifact into tar.gz...")
            convert_to_tar(
                self.model_trainer_artifact.model_trainer_file_path,
                self.tar_output_path,
            )

        except Exception as e:
            raise e

    def fetch_challenger_metrics(self) -> dict:
        try:
            with open(
                self.model_evaluation_artifact.model_evaluation_metrics_file_path,
                "r",
            ) as f:
                return json.load(f)

        except Exception as e:
            raise e

    def initiate_pusher(self) -> ModelPusherArtifact:
        try:
            if not os.path.exists(self.decision_file_path):
                raise FileNotFoundError(
                    f"Comparison result missing at {self.decision_file_path}"
                )

            with open(self.decision_file_path, "r") as f:
                decision = json.load(f)

            if not decision.get("push_to_production", False):
                raise RuntimeError(
                    "Challenger model performance insufficient. Push aborted."
                )

            self.compress_model()

            challenger_metrics = self.fetch_challenger_metrics()

            model_url = self.bucket_ops.upload_model_artifact(
                self.tar_output_path,
                self.s3_model_key,
            )

            preprocessor_url = self.bucket_ops.upload_preprocessor_artifact(
                self.preprocessor_path,
                self.s3_preprocessor_key,
            )

            self.bucket_ops.upload_metrics_artifact(
                challenger_metrics,
                self.s3_metric_key,
            )
            self.bucket_ops.promote_champion_artifacts(
                self.tar_output_path,
                challenger_metrics,
            )

            os.makedirs(self.model_pusher_config.model_pusher_dir, exist_ok=True)
            with open(
                os.path.join(
                    self.model_pusher_config.model_pusher_dir, "push_manifest.json"
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "timestamp": self.timestamp,
                        "s3_model_key": self.s3_model_key,
                        "s3_metric_key": self.s3_metric_key,
                        "model_url": model_url,
                        "preprocessor_url": preprocessor_url,
                    },
                    f,
                    indent=4,
                )

            serving_image_uri = self.bucket_ops.get_serving_image_uri()
            # SageMaker model names only allow [a-zA-Z0-9-]
            model_name = (
                f"{SAGEMAKER_MODEL_NAME_PREFIX}-{self.timestamp.replace('_', '-')}"
            )

            core_model = self.bucket_ops.create_sagemaker_model(
                model_name=model_name,
                model_url=model_url,
                serving_image_uri=serving_image_uri,
            )
            model_package = self.bucket_ops.register_model_package(
                model_url=model_url,
                serving_image_uri=serving_image_uri,
                build_id=self.timestamp,
            )

            logging.info(
                f"Model pusher executed successfully. Core Version ARN: {model_package.model_package_arn}"
            )

            return ModelPusherArtifact(
                model_package_arn=str(model_package.model_package_arn),
                model_name=str(core_model.model_name),
                s3_model_uri=model_url,
                s3_metric_key=self.s3_metric_key,
            )

        except Exception as e:
            logging.error(f"Pusher operation fatal exception: {str(e)}")
            raise e

    def init_model_pusher(self) -> ModelPusherArtifact:
        return self.initiate_pusher()
