import redis

from src.core.config import get_settings

_settings = get_settings()

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=_settings.redis_host,
            port=_settings.redis_port,
            db=_settings.redis_db,
            decode_responses=True,
        )
    return _redis_client
