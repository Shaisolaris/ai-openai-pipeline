# ai-openai-pipeline

![CI](https://github.com/Shaisolaris/ai-openai-pipeline/actions/workflows/ci.yml/badge.svg)

> Production OpenAI pipeline: streaming chat, function calling with 5 tools, structured JSON output, embeddings with similarity search, conversation memory. FastAPI serving + pip installable.


Production OpenAI pipeline with streaming chat, function calling, text embeddings, structured JSON output, and conversation memory. pip installable with 4 runnable examples.

## Quick Start

```bash
pip install -e .
export OPENAI_API_KEY=sk-...

# Run examples
python examples/01_basic_chat.py
python examples/02_function_calling.py
python examples/03_embeddings_similarity.py
python examples/04_structured_output.py

# Start API server
python main.py
# → http://localhost:8000/docs
```

OpenAI integration pipeline with streaming chat, function calling with 5 built-in tools, structured JSON output, text embeddings with cosine similarity, conversation memory with session management, and a FastAPI serving layer. Supports GPT-4o, tool chaining, and SSE streaming.

## Stack

- **AI:** OpenAI API (GPT-4o, text-embedding-3-small, DALL-E)
- **API:** FastAPI + uvicorn
- **Streaming:** Server-Sent Events (SSE)

## Features

### Chat Pipeline
- Streaming and non-streaming completions
- Configurable model, temperature, max tokens
- Session-based conversation memory with truncation
- System prompt customization per session

### Function Calling / Tool Use
5 built-in tools with automatic execution and response chaining:

| Tool | Description |
|---|---|
| `get_weather` | Weather lookup by location |
| `search_web` | Web search with configurable result count |
| `calculate` | Safe math expression evaluation |
| `get_current_time` | Current UTC datetime |
| `create_image` | DALL-E image generation |

Tools are defined in OpenAI function calling format. The pipeline automatically executes tool calls, feeds results back, and returns the final response.

### Embeddings
- Single text embedding via `text-embedding-3-small`
- Batch embeddings for multiple texts
- Cosine similarity calculation between text pairs

### Structured Output
- JSON mode with schema enforcement
- Zero-temperature for deterministic output
- Schema passed as system prompt constraint

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Chat with optional streaming and tool use |
| POST | `/embeddings` | Create embeddings (single or batch) |
| POST | `/similarity` | Cosine similarity between two texts |
| POST | `/structured` | Structured JSON output with schema |
| GET | `/sessions/{id}` | Get conversation history |
| DELETE | `/sessions/{id}` | Clear session memory |
| GET | `/health` | Health check with session count |

## Architecture

```
ai-openai-pipeline/
├── main.py                    # uvicorn entry point
├── pipeline/
│   └── core.py                # Chat, streaming, tool handling, embeddings, structured output, memory
├── tools/
│   └── builtin.py             # 5 tool definitions + handlers + registry
├── api/
│   └── app.py                 # FastAPI routes with session management
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/Shaisolaris/ai-openai-pipeline.git
cd ai-openai-pipeline
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py
# → http://localhost:8000/docs
```

## Key Design Decisions

**Tool handler registry.** Tools are defined as OpenAI function schemas in `TOOL_DEFINITIONS` and mapped to Python functions in `TOOL_HANDLERS`. Adding a new tool means adding one schema dict and one function. The pipeline automatically detects tool calls, executes handlers, and feeds results back for the final response.

**Session-based memory.** Each `session_id` gets its own `ConversationMemory` with message history and system prompt. Memory auto-truncates to the last 50 messages. In production, swap the in-memory dict for Redis.

**SSE streaming.** When `stream=true`, the endpoint returns `text/event-stream` with chunked responses. Each chunk is a `data: {text}\n\n` line. The final message is `data: [DONE]\n\n`. This works with any SSE client (EventSource, fetch with ReadableStream).

**Structured output via JSON mode.** The schema is injected into the system prompt and `response_format` is set to `json_object`. Temperature 0 ensures deterministic, parseable output.

## License

MIT
