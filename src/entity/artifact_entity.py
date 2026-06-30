from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_report_path: str
    validation_msg: str
    validation_status: bool


@dataclass
class DataTransformationArtifact:
    transformed_preprocessor_file_path: str
    transformed_x_train_file_path: str
    transformed_y_train_file_path: str
    transformed_x_test_file_path: str
    transformed_y_test_file_path: str


@dataclass
class ModelTrainerArtifact:
    model_trainer_file_path: str
    model_trainer_tar_file_path: str


@dataclass
class ModelEvaluationArtifact:
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float
    model_evaluation_metrics_file_path: str
