import numpy as np
import pandas as pd
from typing import Optional

from src.configuration.mongo_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.logger import logging


class FraudGuardData:
    """
    Exports data mongoDB as a pandas DataFrame
    """

    def __init__(self) -> None:
        """
        Initializes mongoDB client connection
        """

        try:
            self.mongo_client = MongoDBClient(db_name=DATABASE_NAME)

        except Exception as e:
            raise e

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            if not database_name:
                collection = self.mongo_client.db[collection_name]

            else:
                collection = self.mongo_client.client[database_name][collection_name]  # pyright: ignore[reportOptionalSubscript]

                df = pd.DataFrame(list(collection.find()))
                logging.info("Successfully fetched data from mongoDB")
                df.drop("_id", axis=1, inplace=True)
                df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise e
