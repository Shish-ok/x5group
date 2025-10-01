import asyncio
from fastapi import APIRouter, HTTPException, Request
from app.schemas.predict import PredictIn, PredictOut
from app.core.config import settings

router = APIRouter()

@router.post("/predict", response_model=PredictOut)
async def predict(body: PredictIn, request: Request) -> PredictOut:
    predictor = request.app.state.predictor
    try:
        return await asyncio.wait_for(
            predictor.predict(body.input),
            timeout=settings.request_timeout_s,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timeout")