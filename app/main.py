from fastapi import FastAPI, Request
from app.api.v1.predict import router as predict_router
from app.services.predictor import NerModel, AsyncPredictor
from app.core.config import settings

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.on_event("startup")
    async def _startup() -> None:
        model = NerModel(settings.model_path)
        app.state.predictor = AsyncPredictor(model, max_workers=settings.max_workers)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        app.state.predictor.shutdown()

    app.include_router(predict_router, prefix="/api", tags=["predict"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/debug/model")
    async def model_info(request: Request) -> dict:
        return request.app.state.predictor.info()

    return app

app = create_app()