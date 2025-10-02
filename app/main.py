from fastapi import FastAPI, Request
from app.api.v1.predict import router as predict_router
from app.services.predictor import NerModel, AsyncPredictor
from app.core.config import settings

import logging
import os
from app.utils.reqlog import JsonlLogger

def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = FastAPI(title=settings.app_name)

    @app.on_event("startup")
    async def _startup() -> None:
        model = NerModel(settings.model_path)
        app.state.predictor = AsyncPredictor(model, max_workers=settings.max_workers)

        if settings.request_log_enabled:
            pid = os.getpid()
            log_path = os.path.join(settings.request_log_dir, f"requests-{pid}.jsonl")
            app.state.req_logger = JsonlLogger(log_path)
        else:
            app.state.req_logger = None

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