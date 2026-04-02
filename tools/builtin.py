"""Built-in tools for OpenAI function calling."""

from __future__ import annotations

import json
import datetime
from typing import Any


# ─── Tool Definitions (OpenAI format) ────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2 * (3 + 4)'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone, e.g. US/Pacific", "default": "UTC"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_image",
            "description": "Generate an image using DALL-E",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description"},
                    "size": {"type": "string", "enum": ["256x256", "512x512", "1024x1024"], "default": "1024x1024"},
                },
                "required": ["prompt"],
            },
        },
    },
]


# ─── Tool Handlers ───────────────────────────────────────

def get_weather(location: str, unit: str = "fahrenheit") -> dict[str, Any]:
    """Simulated weather lookup."""
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 12,
        "wind_unit": "mph" if unit == "fahrenheit" else "km/h",
    }


def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Simulated web search."""
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i+1}", "snippet": f"Relevant information about {query}..."}
            for i in range(min(max_results, 5))
        ],
        "total_results": max_results,
    }


def calculate(expression: str) -> dict[str, Any]:
    """Safe math expression evaluator."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def get_current_time(timezone: str = "UTC") -> dict[str, str]:
    """Get current time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "timezone": timezone,
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }


def create_image(prompt: str, size: str = "1024x1024") -> dict[str, str]:
    """Simulated DALL-E image generation."""
    return {
        "prompt": prompt,
        "size": size,
        "url": f"https://api.openai.com/v1/images/generations/{hash(prompt) % 10000}",
        "status": "generated",
    }


# ─── Registry ────────────────────────────────────────────

TOOL_HANDLERS: dict[str, Any] = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "create_image": create_image,
}
