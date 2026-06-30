from from_root import from_root

# Global Constants
RANDOM_STATE = 42

# AWS Config
BUCKET_PATH: str = "s3://fraudguard-dataset-klasta/fraudguard/model"

# Database Config
DATABASE_NAME = "FraudGuard"
COLLECTION_NAME = "fraudguard-data"

# Training Data Config
PIPELINE_NAME = "fraudguard"
ARTIFACT_DIR = "artifact"

# Files
FILE_NAME = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
MODEL_FILE_NAME: str = "model.onnx"
MODEL_TAR_FILE_NAME: str = "model.tar.gz"
SCHEMA_FILE_PATH = from_root() / "config" / "schema.yaml"
HYPERPARAMS_FILE_PATH = from_root() / "config" / "hyperparams.yaml"

# Data Ingestion Config
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# Data Validation Config
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALIDATION_REPORT: str = "data_validation_report.csv"

# Data Transformation Config
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
DATA_TRANSFORMATION_PREPROCESSOR_DIR: str = "preprocessor"

# Model trainer config
MODEL_TRAINER_DIR: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_METRICS_DIR: str = "metrics"
MODEL_TRAINER_METRICS_FILE_PATH: str = "metrics.json"

# Model Evaluation config
MODEL_EVALUATION_DIR = "model_evaluation"
