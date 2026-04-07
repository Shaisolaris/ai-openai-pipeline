"""OpenAI pipeline — streaming, function calling, tool use, structured output."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

import redis
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

client = AsyncOpenAI()


class ConversationMemory(ABC):
    """Abstract base class for managing conversation history."""

    @abstractmethod
    def add_user(self, content: str) -> None:
        pass

    @abstractmethod
    def add_assistant(self, content: str) -> None:
        pass

    @abstractmethod
    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[ChatCompletionMessageParam]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


@dataclass
class InMemoryConversationMemory(ConversationMemory):
    """Manages conversation history in memory with token-aware truncation."""

    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."
    max_messages: int = 50

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._truncate()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._truncate()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    def clear(self) -> None:
        self.messages.clear()

    def _truncate(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


@dataclass
class PipelineConfig:
    """Configuration for the OpenAI pipeline."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_handlers: dict[str, Callable] = field(default_factory=dict)


async def chat(
    messages: list[ChatCompletionMessageParam],
    config: PipelineConfig,
) -> str:
    """Non-streaming chat completion."""
    response = await client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tools=config.tools or None,
    )
    message = response.choices[0].message

    # Handle tool calls
    if message.tool_calls:
        return await _handle_tool_calls(messages, message, config)

    return message.content or ""


async def chat_stream(
    messages: list[ChatCompletionMessageParam],
    config: PipelineConfig,
) -> AsyncGenerator[str, None]:
    """Streaming chat completion."""
    stream = await client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def _handle_tool_calls(
    messages: list[ChatCompletionMessageParam],
    assistant_message: Any,
    config: PipelineConfig,
) -> str:
    """Execute tool calls and get final response."""
    messages = [*messages, assistant_message.model_dump()]

    for tool_call in assistant_message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        handler = config.tool_handlers.get(fn_name)
        if handler:
            result = handler(**fn_args)
            if hasattr(result, "__await__"):
                result = await result
        else:
            result = f"Unknown tool: {fn_name}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result) if not isinstance(result, str) else result,
        })

    # Get final response after tool execution
    response = await client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return response.choices[0].message.content or ""


async def structured_output(
    prompt: str,
    schema: dict[str, Any],
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Get structured JSON output matching a schema."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


async def create_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Create a text embedding."""
    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


async def create_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Create embeddings for multiple texts."""
    response = await client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

@dataclass
class SQLiteConversationMemory(ConversationMemory):
    """Manages conversation history using a SQLite database."""

    db_path: str
    conversation_id: str
    system_prompt: str = "You are a helpful assistant."
    ttl: int = 3600  # Time-to-live in seconds (1 hour)

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        self._enforce_ttl()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_call_id TEXT,
                timestamp REAL NOT NULL
            )
        """)
        self.conn.commit()

    def _enforce_ttl(self):
        cursor = self.conn.cursor()
        cutoff = time.time() - self.ttl
        cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff,))
        self.conn.commit()

    def add_user(self, content: str) -> None:
        self._add_message("user", content)

    def add_assistant(self, content: str) -> None:
        self._add_message("assistant", content)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._add_message("tool", content, tool_call_id)

    def _add_message(self, role: str, content: str, tool_call_id: str | None = None) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (conversation_id, role, content, tool_call_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            (self.conversation_id, role, content, tool_call_id, time.time()),
        )
        self.conn.commit()

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content, tool_call_id FROM conversations WHERE conversation_id = ? ORDER BY timestamp ASC",
            (self.conversation_id,),
        )
        messages = [{"role": "system", "content": self.system_prompt}]
        for role, content, tool_call_id in cursor.fetchall():
            message = {"role": role, "content": content}
            if tool_call_id:
                message["tool_call_id"] = tool_call_id
            messages.append(message)
        return messages

    def clear(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (self.conversation_id,))
        self.conn.commit()


@dataclass
class RedisConversationMemory(ConversationMemory):
    """Manages conversation history using a Redis database."""

    redis_url: str
    conversation_id: str
    system_prompt: str = "You are a helpful assistant."
    ttl: int = 3600  # Time-to-live in seconds (1 hour)

    def __post_init__(self):
        self.redis = redis.from_url(self.redis_url)
        self.key = f"conversation:{self.conversation_id}"

    def add_user(self, content: str) -> None:
        self._add_message({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._add_message({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._add_message({"role": "tool", "content": content, "tool_call_id": tool_call_id})

    def _add_message(self, message: dict[str, Any]) -> None:
        self.redis.rpush(self.key, json.dumps(message))
        self.redis.expire(self.key, self.ttl)

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for message_json in self.redis.lrange(self.key, 0, -1):
            messages.append(json.loads(message_json))
        return messages

    def clear(self) -> None:
        self.redis.delete(self.key)

