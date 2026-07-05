import json
import tempfile
import boto3

from src.logger import logging

from pathlib import Path

from botocore.exceptions import ClientError


from src.constants import (
    BUCKET_NAME,
    CHAMPION_LATEST_METRIC_PATH,
    CHAMPION_LATEST_MODEL_PATH,
    CHAMPION_LATEST_PREPROCESSOR_PATH,
    MODEL_PACKAGE_GROUP_NAME,
    SAGEMAKER_INSTANCE_TYPE,
    SAGEMAKER_ROLE_ARN,
    SAGEMAKER_TRITON_VERSION,
    MODEL_TAR_FILE_NAME,
    PREPROCESSOR_FILE_NAME,
    MODEL_TRAINER_METRICS_FILE_PATH,
)


class BucketOperations:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.boto_session = boto3.Session()

        logging.info("Connected to S3 client.")

    def get_champion_metrics(self) -> dict:
        """Fetch the metrics of the model currently in production

        Returns:
            dict: The metrics of the model
        """

        try:
            response = self.s3_client.get_object(
                Bucket=BUCKET_NAME,
                Key=CHAMPION_LATEST_METRIC_PATH,
            )

            return json.loads(
                response["Body"].read().decode("utf-8"),
            )

        except self.s3_client.exceptions.NoSuchKey:
            logging.info("No champion model found in S3. This is the first deployment.")
            return {
                "Accuracy": 0.0,
                "f1 score": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
            }

        except Exception as e:
            raise e

    def download_champion_artifacts(self):
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="fraudguard_champion_"))
            local_path_model = temp_dir / MODEL_TAR_FILE_NAME
            local_path_preprocessor = temp_dir / PREPROCESSOR_FILE_NAME
            local_path_metrics = temp_dir / MODEL_TRAINER_METRICS_FILE_PATH

            # Download model
            self.s3_client.download_file(
                Bucket=BUCKET_NAME,
                Key=CHAMPION_LATEST_MODEL_PATH,
                Filename=local_path_model,
            )

            logging.info(f"Downloaded champion model to {local_path_model}")

            # Download preprocessor
            self.s3_client.download_file(
                Bucket=BUCKET_NAME,
                Key=CHAMPION_LATEST_PREPROCESSOR_PATH,
                Filename=local_path_preprocessor,
            )

            logging.info(f"Downloaded preprocessor to {local_path_preprocessor}")

            # Download metrics
            self.s3_client.download_file(
                Bucket=BUCKET_NAME,
                Key=CHAMPION_LATEST_METRIC_PATH,
                Filename=local_path_metrics,
            )

            logging.info(f"Downloaded metrics to {local_path_metrics}")

            return (
                local_path_model,
                local_path_preprocessor,
                local_path_metrics,
            )

        except ClientError:
            logging.error("Champion model not found in S3.")
            raise

        except Exception as e:
            raise e

    def upload_model_artifact(self, file_path: str, s3_model_key: str) -> str:
        try:
            logging.info(
                f"Pusher streaming model.tar.gz to S3 path: s3://{BUCKET_NAME}/{s3_model_key}"
            )
            self.s3_client.upload_file(file_path, BUCKET_NAME, s3_model_key)

            model_url = f"s3://{BUCKET_NAME}/{s3_model_key}"
            logging.info(f"Model uploaded to path: {model_url}")

            return model_url

        except Exception as e:
            raise e

    def upload_metrics_artifact(self, metrics: dict, s3_metric_key: str) -> None:
        try:
            logging.info(
                f"Uploading challenger metrics metadata: s3://{BUCKET_NAME}/{s3_metric_key}"
            )
            self.s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_metric_key,
                Body=json.dumps(metrics),
            )
            logging.info(
                f"Model metrics uploaded to path: s3://{BUCKET_NAME}/{s3_metric_key}"
            )

        except Exception as e:
            raise e

    def upload_preprocessor_artifact(
        self, file_path: str, s3_preprocessor_key: str
    ) -> str:
        try:
            logging.info(
                f"Pusher streaming preprocessor to S3 path: s3://{BUCKET_NAME}/{s3_preprocessor_key}"
            )
            self.s3_client.upload_file(file_path, BUCKET_NAME, s3_preprocessor_key)

            preprocessor_url = f"s3://{BUCKET_NAME}/{s3_preprocessor_key}"
            logging.info(f"Preprocessor uploaded to path: {preprocessor_url}")

            return preprocessor_url

        except Exception as e:
            raise e

    def promote_champion_artifacts(
        self, file_path: str, metrics: dict, preprocessor_path: str
    ) -> None:
        try:
            logging.info("Promoting challenger artifacts to champion paths in S3...")
            self.upload_model_artifact(file_path, CHAMPION_LATEST_MODEL_PATH)
            self.upload_metrics_artifact(metrics, CHAMPION_LATEST_METRIC_PATH)
            self.upload_preprocessor_artifact(
                preprocessor_path, CHAMPION_LATEST_PREPROCESSOR_PATH
            )

        except Exception as e:
            raise e

    def get_serving_image_uri(self) -> str:
        try:
            from sagemaker.core import image_uris

            logging.info(
                "Retrieving serving container URI using SageMaker v3 core framework..."
            )
            region = self.boto_session.region_name

            serving_image_uri = image_uris.retrieve(
                framework="sagemaker-tritonserver",
                region=region,
                version=SAGEMAKER_TRITON_VERSION,
                image_scope="inference",
                instance_type=SAGEMAKER_INSTANCE_TYPE,
            )

            logging.info(f"Resolved serving image URI: {serving_image_uri}")

            return serving_image_uri

        except Exception as e:
            raise e

    def create_sagemaker_model(
        self,
        model_name: str,
        model_url: str,
        serving_image_uri: str,
    ):
        try:
            from sagemaker.core.resources import Model

            logging.info(f"Creating SageMaker model resource: {model_name}")

            core_model = Model.create(
                model_name=model_name,
                primary_container={  # pyright: ignore[reportArgumentType]
                    "image": serving_image_uri,
                    "model_data_url": model_url,
                },
                execution_role_arn=SAGEMAKER_ROLE_ARN,
                session=self.boto_session,
            )

            if core_model is None:
                raise RuntimeError(f"Failed to create SageMaker model: {model_name}")

            return core_model

        except Exception as e:
            raise e

    def register_model_package(
        self,
        model_url: str,
        serving_image_uri: str,
        build_id: str,
    ):
        try:
            from sagemaker.core.resources import ModelPackage

            logging.info(
                "Registering model group package to the Model Registry catalog..."
            )

            model_package = ModelPackage.create(
                model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
                model_approval_status="Approved",
                model_package_description=(
                    f"Automated MLOps deployment. Build ID: {build_id}"
                ),
                inference_specification={  # pyright: ignore[reportArgumentType]
                    "containers": [
                        {
                            "image": serving_image_uri,
                            "model_data_url": model_url,
                        }
                    ],
                },
                session=self.boto_session,
            )

            if model_package is None:
                raise RuntimeError("Failed to register SageMaker model package")

            logging.info(
                f"Model package registered. Core Version ARN: {model_package.model_package_arn}"
            )

            return model_package

        except Exception as e:
            raise e
