import numpy as np
import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from src.utils.model_loader import load_champion_bundle
from src.utils.main_utils import read_yaml_file
from src.utils.feature_engineering import clean_baf_dataframe

from src.constants import SCHEMA_FILE_PATH

from src.api.schema import Transaction, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_champion_bundle()
    app.state.schema = read_yaml_file(SCHEMA_FILE_PATH.as_posix())
    yield  # server runs here


# Initialize fastapi app with lifespan
app = FastAPI(title="FraudGuard", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request):
    bundle = request.app.state.bundle

    df = pd.DataFrame([transaction.model_dump()])
    df = clean_baf_dataframe(df, request.app.state.schema)
    features = bundle.preprocessor.transform(df).astype(np.float32)

    outputs = bundle.model.run(None, {"input": features})
    fraud_probability = float(outputs[1][0][1])

    return PredictionResponse(
        is_fraud=1 if fraud_probability >= 0.5 else 0,
        fraud_probability=fraud_probability,
    )
