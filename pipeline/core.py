"""OpenAI pipeline — streaming, function calling, tool use, structured output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, TYPE_CHECKING

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from pipeline.persistence import SQLiteConversationStorage
else:  # pragma: no cover
    try:
        from pipeline.persistence import SQLiteConversationStorage  # type: ignore
    except ImportError:
        class SQLiteConversationStorage:  # type: ignore[too-many-ancestors]
            """Placeholder when persistence module is unavailable."""

            ...

logger = logging.getLogger(__name__)

client = AsyncOpenAI()


@dataclass
class ConversationMemory:
    """Manages conversation history with token-aware truncation."""

    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."
    max_messages: int = 50
    storage_backend: SQLiteConversationStorage | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        if self.storage_backend and not self.messages:
            stored_messages = self._load_from_backend()
            if stored_messages:
                self.messages = list(stored_messages)
                self._truncate()

    def add_user(self, content: str) -> None:
        self.add_message({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.add_message({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.add_message({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def add_message(self, message: ChatCompletionMessageParam) -> None:
        self.messages.append(message)
        self._truncate()
        self._save_to_backend()

    def get_conversation(self) -> list[ChatCompletionMessageParam]:
        if not self.messages:
            stored_messages = self._load_from_backend()
            if stored_messages:
                self.messages = list(stored_messages)
                self._truncate()
        return self.messages

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        conversation = self.get_conversation()
        return [{"role": "system", "content": self.system_prompt}, *conversation]

    def clear(self) -> None:
        self.messages.clear()
        self._save_to_backend()

    def _truncate(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def _resolve_storage_callable(self, *names: str) -> Callable[..., Any] | None:
        if not self.storage_backend:
            return None
        for name in names:
            handler = getattr(self.storage_backend, name, None)
            if callable(handler):
                return handler
        return None

    def _load_from_backend(self) -> list[ChatCompletionMessageParam] | None:
        loader = self._resolve_storage_callable("load", "load_conversation", "get_conversation")
        if not loader:
            if self.storage_backend:
                logger.warning("Storage backend %s does not expose a load method.", type(self.storage_backend).__name__)
            return None

        attempts: list[tuple[Any, ...]] = []
        if self.conversation_id is not None:
            attempts.append((self.conversation_id,))
        attempts.append(tuple())

        for args in attempts:
            try:
                result = loader(*args)
                if result is None:
                    return None
                if isinstance(result, list):
                    return result
                return list(result)
            except TypeError:
                continue
            except Exception:
                logger.exception("Failed to load conversation from storage backend.")
                return None

        if self.conversation_id is None:
            logger.warning(
                "Storage backend %s requires a conversation identifier; provide `conversation_id` when creating ConversationMemory.",
                type(self.storage_backend).__name__ if self.storage_backend else "unknown",
            )
        else:
            logger.warning(
                "Storage backend %s load callable signature unsupported; skipping load.",
                type(self.storage_backend).__name__,
            )
        return None

    def _save_to_backend(self) -> None:
        saver = self._resolve_storage_callable("save", "save_conversation", "put_conversation")
        if not saver:
            if self.storage_backend:
                logger.warning("Storage backend %s does not expose a save method; skipping persistence.", type(self.storage_backend).__name__)
            return

        attempts: list[tuple[Any, ...]] = []
        if self.conversation_id is not None:
            attempts.append((self.conversation_id, self.messages))
            attempts.append((self.conversation_id,))
        attempts.append((self.messages,))
        attempts.append(tuple())

        for args in attempts:
            try:
                saver(*args)
                return
            except TypeError:
                continue
            except Exception:
                logger.exception("Failed to save conversation to storage backend.")
                return

        if self.conversation_id is None:
            logger.warning(
                "Storage backend %s requires a conversation identifier; provide `conversation_id` when creating ConversationMemory.",
                type(self.storage_backend).__name__ if self.storage_backend else "unknown",
            )
        else:
            logger.warning(
                "Storage backend %s save callable signature unsupported; skipping persistence.",
                type(self.storage_backend).__name__,
            )


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
