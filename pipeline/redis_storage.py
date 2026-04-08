
from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

import redis

logger = logging.getLogger(__name__)


class RedisConversationStorage:
    """Simple Redis-backed conversation storage with optional TTL."""

    def __init__(self, redis_url: str, ttl: int | float | None = None) -> None:
        self.redis_url = redis_url
        self.ttl = ttl
        try:
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:  # pragma: no cover - connection setup failures are logged
            logger.exception("Failed to connect to Redis at %s", redis_url)
            raise

    def save_conversation(self, conversation_id: str, messages: Sequence[Mapping[str, Any]]) -> None:
        """Persist the conversation under the given identifier."""
        payload = json.dumps(list(messages))
        try:
            if self.ttl is None:
                self.client.set(conversation_id, payload)
            else:
                self.client.set(conversation_id, payload, ex=self.ttl)
        except Exception:
            logger.exception("Failed to save conversation %s to Redis", conversation_id)
            raise

    def load_conversation(self, conversation_id: str) -> list[dict[str, Any]] | None:
        """Fetch a conversation by identifier, returning None when missing."""
        try:
            data = self.client.get(conversation_id)
        except Exception:
            logger.exception("Failed to load conversation %s from Redis", conversation_id)
            raise

        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Stored conversation %s contains invalid JSON; clearing entry", conversation_id)
            self.delete_conversation(conversation_id)
            return None

    def delete_conversation(self, conversation_id: str) -> None:
        """Remove a stored conversation."""
        try:
            self.client.delete(conversation_id)
        except Exception:
            logger.exception("Failed to delete conversation %s from Redis", conversation_id)
            raise
