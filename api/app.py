"""FastAPI endpoints for the OpenAI pipeline."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pipeline.core import (
    chat, chat_stream, structured_output, create_embedding, create_embeddings,
    cosine_similarity, ConversationMemory, PipelineConfig,
)
from tools.builtin import TOOL_DEFINITIONS, TOOL_HANDLERS

logger = logging.getLogger(__name__)
app = FastAPI(title="OpenAI Pipeline API", version="1.0.0")

# Per-session conversation memory (in production: Redis/DB)
_sessions: dict[str, ConversationMemory] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    system_prompt: str = Field(default="You are a helpful assistant.")
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0, le=2)
    use_tools: bool = Field(default=False)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int


class EmbeddingRequest(BaseModel):
    text: str | list[str]
    model: str = "text-embedding-3-small"


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str


class StructuredRequest(BaseModel):
    prompt: str
    schema_def: dict[str, Any] = Field(alias="schema")
    model: str = "gpt-4o"


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "sessions": len(_sessions)}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Chat with optional tool use. Set stream=true for SSE streaming."""
    memory = _sessions.setdefault(req.session_id, ConversationMemory(system_prompt=req.system_prompt))
    memory.system_prompt = req.system_prompt
    memory.add_user(req.message)

    config = PipelineConfig(
        model=req.model,
        temperature=req.temperature,
        tools=TOOL_DEFINITIONS if req.use_tools else [],
        tool_handlers=TOOL_HANDLERS if req.use_tools else {},
    )

    if req.stream:
        async def generate():
            chunks: list[str] = []
            async for chunk in chat_stream(memory.get_messages(), config):
                chunks.append(chunk)
                yield f"data: {chunk}\n\n"
            full = "".join(chunks)
            memory.add_assistant(full)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    response = await chat(memory.get_messages(), config)
    memory.add_assistant(response)

    return ChatResponse(response=response, session_id=req.session_id, message_count=len(memory.messages))


@app.post("/embeddings")
async def embeddings_endpoint(req: EmbeddingRequest):
    """Create embeddings for text or list of texts."""
    if isinstance(req.text, str):
        embedding = await create_embedding(req.text, req.model)
        return {"embedding": embedding, "dimensions": len(embedding)}
    else:
        embeddings = await create_embeddings(req.text, req.model)
        return {"embeddings": embeddings, "count": len(embeddings), "dimensions": len(embeddings[0]) if embeddings else 0}


@app.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest):
    """Calculate cosine similarity between two texts."""
    emb_a = await create_embedding(req.text_a)
    emb_b = await create_embedding(req.text_b)
    score = cosine_similarity(emb_a, emb_b)
    return {"similarity": round(score, 4), "text_a": req.text_a[:100], "text_b": req.text_b[:100]}


@app.post("/structured")
async def structured_endpoint(req: StructuredRequest):
    """Get structured JSON output matching a schema."""
    result = await structured_output(req.prompt, req.schema_def, req.model)
    return {"result": result}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _sessions:
        _sessions[session_id].clear()
        return {"cleared": True, "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get conversation history for a session."""
    memory = _sessions.get(session_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": memory.messages, "count": len(memory.messages)}
