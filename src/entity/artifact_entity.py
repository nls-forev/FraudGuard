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
    transformed_train_file_path: str
    transformed_test_file_path: str
