import os
import pymongo
import certifi

from dotenv import load_dotenv
from from_root import from_root
from src.logger import logging

from src.constants import DATABASE_NAME

load_dotenv(from_root() / ".env")


ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, db_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("CONNECTION_URL")
                if mongo_db_url is None:
                    raise Exception("Environment variable: CONNECTION_URL, is not set.")

                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.db = self.client[db_name]
            self.db_name = db_name
            logging.info("MongoDB connection successful.")

        except Exception as e:
            raise e
