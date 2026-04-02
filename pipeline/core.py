"""OpenAI pipeline — streaming, function calling, tool use, structured output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

client = AsyncOpenAI()


@dataclass
class ConversationMemory:
    """Manages conversation history with token-aware truncation."""

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
