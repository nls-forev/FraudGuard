"""Drain the Redis prediction buffer into MongoDB.

Runs as a scheduled Fargate task (EventBridge Scheduler -> ECS RunTask) using
the serving image with this module as the command override. Delivery is
at-least-once: documents are inserted into Mongo before the buffer is trimmed,
so a crash between the two steps re-delivers on the next run rather than
losing records.
"""

import json
import sys

from src.configuration.mongo_connection import MongoDBClient
from src.constants import (
    DATABASE_NAME,
    PREDICTIONS_COLLECTION_NAME,
    REDIS_PREDICTIONS_KEY,
)
from src.data_access.redis_client import get_redis_client
from src.logger import logging


def flush_predictions() -> int:
    redis_client = get_redis_client()

    buffered = redis_client.llen(REDIS_PREDICTIONS_KEY)
    if buffered == 0:
        logging.info("Prediction buffer empty. Nothing to flush.")
        return 0

    raw_records = redis_client.lrange(REDIS_PREDICTIONS_KEY, 0, buffered - 1)

    documents = []
    for raw in raw_records:
        try:
            documents.append(json.loads(raw))
        except json.JSONDecodeError:
            logging.warning(f"Skipping malformed buffer entry: {raw[:200]}")

    if documents:
        collection = MongoDBClient(DATABASE_NAME).db[PREDICTIONS_COLLECTION_NAME]
        collection.insert_many(documents, ordered=False)

    # Trim only what was read; records RPUSHed while flushing survive.
    redis_client.ltrim(REDIS_PREDICTIONS_KEY, buffered, -1)

    logging.info(
        f"Flushed {len(documents)} predictions to "
        f"{DATABASE_NAME}.{PREDICTIONS_COLLECTION_NAME} "
        f"({buffered - len(documents)} malformed entries dropped)."
    )
    return len(documents)


if __name__ == "__main__":
    try:
        flush_predictions()
    except Exception:
        logging.exception("Prediction flush failed.")
        sys.exit(1)
