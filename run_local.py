if __name__ == "__main__":
    import uvicorn

    from src.core.config import get_settings
    from src.core.structlog_config import configure_logging

    settings = get_settings()
    configure_logging(console_level=settings.log_level)

    uvicorn.run(
        "src.api.app:create_app",
        host="0.0.0.0",
        port=settings.api_port,
        factory=True,
        reload=True,
        reload_excludes=[".venv", "log", "*.log", "__pycache__", "*.pyc"],
        log_config=None,
    )
