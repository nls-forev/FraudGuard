import boto3

from src.logger import logging


class HandleBucket:
    def __init__(self):
        self.s3_client = boto3.client("s3")

    def upload_model_to_bucket(self, file_path: str, bucket_name: str, model_key: str):
        logging.info("Connected to S3 client.")

        self.s3_client.upload_file(
            file_path,
            bucket_name,
            model_key,
        )

        logging.info("")
