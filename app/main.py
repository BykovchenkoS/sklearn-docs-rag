import logging
import os

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

ROOT_PATH = os.getenv("ROOT_PATH", "")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    root_path=ROOT_PATH,
)


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("health check called")
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"version": settings.app_version}
