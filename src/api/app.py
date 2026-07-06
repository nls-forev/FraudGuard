import json
import numpy as np
import pandas as pd

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request

from src.utils.model_loader import load_champion_bundle
from src.utils.main_utils import read_yaml_file
from src.utils.feature_engineering import clean_baf_dataframe
from src.data_access.redis_client import get_redis_client
from src.logger import logging

from src.constants import SCHEMA_FILE_PATH, REDIS_PREDICTIONS_KEY

from src.api.schema import Transaction, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_champion_bundle()
    app.state.schema = read_yaml_file(SCHEMA_FILE_PATH.as_posix())
    app.state.redis = get_redis_client()
    yield  # server runs here


# Initialize fastapi app with lifespan
app = FastAPI(title="FraudGuard", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request):
    # Section A: Handle Request
    bundle = request.app.state.bundle

    df = pd.DataFrame([transaction.model_dump()])
    df = clean_baf_dataframe(df, request.app.state.schema)
    features = bundle.preprocessor.transform(df).astype(np.float32)

    outputs = bundle.model.run(None, {"input": features})
    fraud_probability = float(outputs[1][0][1])
    is_fraud = 1 if fraud_probability >= 0.5 else 0

    # Section B: buffer prediction in Redis for later batch flush to MongoDB.
    # Best-effort only — a Redis outage must never break inference.
    try:
        record = {
            **transaction.model_dump(),
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
        }
        request.app.state.redis.rpush(REDIS_PREDICTIONS_KEY, json.dumps(record))
    except Exception:
        logging.exception("Failed to buffer prediction to Redis")

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
    )
