import argparse
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_compare import ModelCompare
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from src.logger import logging


def ingestion_artifact() -> DataIngestionArtifact:
    config = DataIngestionConfig()
    return DataIngestionArtifact(
        train_file_path=config.training_file,
        test_file_path=config.testing_file,
    )


def validation_artifact() -> DataValidationArtifact:
    config = DataValidationConfig()
    return DataValidationArtifact(
        validation_report_path=config.data_validation_report,
        validation_msg="",
        validation_status=True,
    )


def transformation_artifact() -> DataTransformationArtifact:
    config = DataTransformationConfig()
    return DataTransformationArtifact(
        transformed_preprocessor_file_path=config.preprocessor_file_path,
        transformed_x_train_file_path=config.transformed_x_train_file_path,
        transformed_y_train_file_path=config.transformed_y_train_file_path,
        transformed_x_test_file_path=config.transformed_x_test_file_path,
        transformed_y_test_file_path=config.transformed_y_test_file_path,
    )


def trainer_artifact() -> ModelTrainerArtifact:
    config = ModelTrainerConfig()
    return ModelTrainerArtifact(
        model_trainer_file_path=config.model_trainer_file_path,
        model_trainer_tar_file_path=config.model_trainer_tar_file_path,
    )


def evaluation_artifact() -> ModelEvaluationArtifact:
    # Downstream stages read metric values from the metrics file, not from
    # the scalar fields, so zeros are safe placeholders here.
    config = ModelEvaluationConfig()
    return ModelEvaluationArtifact(
        accuracy=0.0,
        f1_score=0.0,
        precision_score=0.0,
        recall_score=0.0,
        model_evaluation_metrics_file_path=config.model_evaluation_metrics_file_path,
    )


def run_ingest() -> None:
    DataIngestion().init_data_ingestion()


def run_validate() -> None:
    artifact = DataValidation(ingestion_artifact()).init_data_validation()

    if not artifact.validation_status:
        logging.error(f"Data validation failed: {artifact.validation_msg}")
        sys.exit(1)


def run_transform() -> None:
    DataTransformation(
        data_ingestion_artifact=ingestion_artifact(),
        data_validation_artifact=validation_artifact(),
    ).init_data_transformation()


def run_train() -> None:
    ModelTrainer(transformation_artifact()).init_model_training()


def run_evaluate() -> None:
    ModelEvaluation(
        data_transformation_artifact=transformation_artifact(),
        model_training_artifact=trainer_artifact(),
    ).init_model_evaluation()


def run_compare() -> None:
    ModelCompare(evaluation_artifact()).init_model_compare()


def run_push() -> None:
    compare_artifact = ModelCompare(evaluation_artifact()).init_model_compare()

    ModelPusher(
        model_compare_artifact=compare_artifact,
        model_evaluation_artifact=evaluation_artifact(),
        model_trainer_artifact=trainer_artifact(),
    ).init_model_pusher()


STAGES = {
    "ingest": run_ingest,
    "validate": run_validate,
    "transform": run_transform,
    "train": run_train,
    "evaluate": run_evaluate,
    "compare": run_compare,
    "push": run_push,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single pipeline stage.")
    parser.add_argument("stage", choices=STAGES.keys(), help="Pipeline stage to run")
    args = parser.parse_args()

    logging.info(f"Running pipeline stage: {args.stage}")
    STAGES[args.stage]()
    logging.info(f"Finished pipeline stage: {args.stage}")


if __name__ == "__main__":
    main()
