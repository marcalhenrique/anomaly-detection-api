import uvicorn

from src.core.config import get_settings
from src.core.structlog_config import configure_logging, get_logger

settings = get_settings()
configure_logging(console_level=settings.log_level)
logger = get_logger(__name__)


def main():
    logger.info("Starting Anomaly Detection API", extra={"port": settings.api_port})
    uvicorn.run(
        "src.api.app:create_app",
        host="0.0.0.0",
        port=settings.api_port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
