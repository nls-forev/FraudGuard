# Global Constants
RANDOM_STATE = 42

# Training Data Config
PIPELINE_NAME = "fraudguard"
ARTIFACT_DIR = "artifact"

# Files
FILE_NAME = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Data Ingestion Config
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# Database Config
DATABASE_NAME = "FraudGuard"
COLLECTION_NAME = "fraudguard-data"
