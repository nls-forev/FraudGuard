import os
import json

from src.data_access.aws_sagemaker import BucketOperations

from src.entity.artifact_entity import ModelEvaluationArtifact, ModelCompareArtifact
from src.entity.config_entity import ModelCompareConfig

from src.logger import logging


class ModelCompare:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_compare_config: ModelCompareConfig = ModelCompareConfig(),
        bucket_ops: BucketOperations = BucketOperations(),
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_compare_config = model_compare_config
        self.bucket_ops = bucket_ops

    def fetch_challenger_metrics(self) -> dict:
        try:
            with open(
                self.model_evaluation_artifact.model_evaluation_metrics_file_path, "r"
            ) as f:
                metrics = json.load(f)

            return metrics

        except Exception as e:
            raise e

    def compare_metrics(
        self, challenger_metrics: dict, champion_metrics: dict
    ) -> ModelCompareArtifact:
        try:
            os.makedirs(
                os.path.dirname(
                    self.model_compare_config.model_compare_decision_file_path
                ),
                exist_ok=True,
            )

            approved = False

            challenger_f1 = challenger_metrics["f1 score"]
            champion_f1 = champion_metrics["f1 score"]

            logging.debug(
                f"Fetched challenger and champion f1 score: {challenger_f1}, {champion_f1}"
            )

            cond = challenger_f1 >= champion_f1 + 0.01

            if cond:
                approved = True
                logging.info("Challenger is approved!")

            decision_data = {
                "push_to_production": approved,
            }

            with open(
                self.model_compare_config.model_compare_decision_file_path, "w"
            ) as f:
                json.dump(
                    decision_data,
                    f,
                    indent=4,
                )

            model_compare_artifact = ModelCompareArtifact(
                push_to_production=approved,
                model_compare_decision_file_path=self.model_compare_config.model_compare_decision_file_path,
            )

            return model_compare_artifact

        except Exception as e:
            raise e

    def init_model_compare(self):
        champion_metrics = self.bucket_ops.get_champion_metrics()
        challenger_metrics = self.fetch_challenger_metrics()

        logging.debug(
            f"Champion Metrics: {champion_metrics}, Challenger Metrics: {challenger_metrics}"
        )

        model_compare_artifact = self.compare_metrics(
            challenger_metrics, champion_metrics
        )

        return model_compare_artifact
