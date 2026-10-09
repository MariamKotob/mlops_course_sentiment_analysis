"""HTTP API serving the sentiment model.

    uv run uvicorn mlops_practitioner_course.api:app --reload
"""

import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated

import bentoml
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field, StringConstraints

from mlops_practitioner_course.config import Settings

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 64

# Whitespace-only text is rejected with a 422 before it reaches the tokenizer.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictRequest(BaseModel):
    text: Text


class BatchPredictRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    text: str
    label: str
    probability: float = Field(description="P(positive)")


app = FastAPI(title="Arabic tweet sentiment API")

def get_model_service(service=Depends(bentoml.get_current_service)):
    return service.model_service

# --- Prometheus metrics ---
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")

async def predict_texts(model_service, texts: Sequence[str]) -> list[Prediction]:
    # Call the BentoML API asynchronously
    probs = await model_service.to_async.predict(texts)
    # Get threshold
    threshold = await model_service.to_async.get_threshold()
    
    return [
        Prediction(
            text=text,
            label="positive" if prob >= threshold else "negative",
            probability=float(prob),
        )
        for text, prob in zip(texts, probs)
    ]


@app.get("/")
async def root(model_service=Depends(get_model_service)) -> dict[str, str | float]:
    settings = Settings.from_yaml()
    threshold = await model_service.to_async.get_threshold()
    return {
        "service": app.title,
        "model_version": settings.model.version,
        "threshold": threshold,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    # BentoML handles readiness/health for the model service natively.
    # This route just confirms the FastAPI app is reachable.
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(body: PredictRequest, model_service=Depends(get_model_service)) -> Prediction:
    predictions = await predict_texts(model_service, [body.text])
    return predictions[0]


@app.post("/predict/batch")
async def predict_batch(body: BatchPredictRequest, model_service=Depends(get_model_service)) -> list[Prediction]:
    return await predict_texts(model_service, body.texts)
