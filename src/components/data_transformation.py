from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        pass
