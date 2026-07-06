import os
import redis

from src.logger import logging


def get_redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(
        redis_url, decode_responses=True, socket_connect_timeout=2
    )
    logging.info("Redis client initialized.")
    return client
