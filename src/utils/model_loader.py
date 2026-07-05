import os
import json

from src.logger import logging

from src.data_access.aws_sagemaker import BucketOperations

from src.entity.artifact_entity import ChampionBundle

from src.utils.main_utils import load_object, extract_tar

from onnxruntime import InferenceSession

from src.constants import MODEL_FILE_NAME


def load_champion_bundle() -> ChampionBundle:
    try:
        bucket_ops = BucketOperations()

        local_path_model, local_path_preprocessor, local_path_metrics = (
            bucket_ops.download_champion_artifacts()
        )

        extract_dir = os.path.dirname(local_path_model)
        model_path = os.path.join(extract_dir, MODEL_FILE_NAME)

        extract_tar(local_path_model.as_posix(), extract_dir)
        logging.info(f"Extracted model at path: {extract_dir}")

        preprocessor = load_object(local_path_preprocessor.as_posix())
        logging.info(
            f"loaded preprocessor from path: {local_path_preprocessor.as_posix()}"
        )

        with open(local_path_metrics.as_posix(), "r") as f:
            metrics = json.load(f)

        logging.info(f"loaded metrics from path: {local_path_metrics.as_posix()}")

        return ChampionBundle(
            model=InferenceSession(model_path, providers=["CPUExecutionProvider"]),
            preprocessor=preprocessor,
            metrics=metrics,
        )

    except Exception as e:
        raise e
