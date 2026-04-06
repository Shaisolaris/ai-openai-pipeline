import os
# Demo mode: runs with sample data when no API keys configured
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true' or not os.getenv('DATABASE_URL')
"""FastAPI endpoints for the OpenAI pipeline."""
from __future__ import annotations
import logging
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pipeline.core import (ConversationMemory, PipelineConfig, chat, chat_stream, structured_output, create_embedding, create_embeddings, cosine_similarity)
from tools.definitions import TOOLS, TOOL_HANDLERS

logger = logging.getLogger(__name__)
app = FastAPI(title="OpenAI Pipeline API", version="1.0.0")
_sessions: dict[str, ConversationMemory] = {}

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    model: str = "gpt-4o"
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    use_tools: bool = True
    system_prompt: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str

class StructuredRequest(BaseModel):
    prompt: str
    schema_def: dict[str, Any] = Field(..., alias="schema")
    model: str = "gpt-4o"

class EmbeddingRequest(BaseModel):
    text: str | list[str]
    model: str = "text-embedding-3-small"

class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str

@app.get("/health")
def health() -> dict: return {"status": "healthy", "sessions": len(_sessions)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    memory = _sessions.setdefault(req.session_id, ConversationMemory())
    if req.system_prompt: memory.system_prompt = req.system_prompt
    memory.add_user(req.message)
    config = PipelineConfig(model=req.model, temperature=req.temperature, tools=TOOLS if req.use_tools else [], tool_handlers=TOOL_HANDLERS if req.use_tools else {})
    try:
        response = await chat(memory.get_messages(), config)
        memory.add_assistant(response)
        return ChatResponse(response=response, session_id=req.session_id, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    memory = _sessions.setdefault(req.session_id, ConversationMemory())
    if req.system_prompt: memory.system_prompt = req.system_prompt
    memory.add_user(req.message)
    config = PipelineConfig(model=req.model, temperature=req.temperature, stream=True)
    async def generate():
        full = ""
        async for chunk in chat_stream(memory.get_messages(), config):
            full += chunk
            yield f"data: {chunk}\n\n"
        memory.add_assistant(full)
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/structured")
async def structured_endpoint(req: StructuredRequest):
    try:
        result = await structured_output(req.prompt, req.schema_def, req.model)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def embeddings_endpoint(req: EmbeddingRequest):
    try:
        if isinstance(req.text, list):
            embs = await create_embeddings(req.text, req.model)
            return {"embeddings": embs, "count": len(embs), "dimensions": len(embs[0]) if embs else 0}
        else:
            emb = await create_embedding(req.text, req.model)
            return {"embedding": emb, "dimensions": len(emb)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest):
    try:
        a = await create_embedding(req.text_a)
        b = await create_embedding(req.text_b)
        return {"similarity": round(cosine_similarity(a, b), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id in _sessions: _sessions[session_id].clear(); return {"cleared": True}
    raise HTTPException(status_code=404, detail="Session not found")
