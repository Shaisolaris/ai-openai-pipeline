"""Tool definitions and handlers for OpenAI function calling."""

from __future__ import annotations

import json
import datetime
from typing import Any


# ─── Tool Definitions (OpenAI format) ────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'San Francisco, CA'"},
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
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5, "description": "Number of results"},
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
                    "expression": {"type": "string", "description": "Math expression, e.g. '2 * (3 + 4)'"},
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
                    "timezone": {"type": "string", "description": "Timezone, e.g. 'US/Pacific'", "default": "UTC"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a task or reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "due_date": {"type": "string", "description": "ISO date string"},
                },
                "required": ["title"],
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
        "wind_speed": 8,
    }


def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
    """Simulated web search."""
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i+1}", "snippet": f"This is a relevant result about {query}..."}
            for i in range(min(num_results, 5))
        ],
    }


def calculate(expression: str) -> dict[str, Any]:
    """Safe math evaluation."""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def get_current_time(timezone: str = "UTC") -> dict[str, Any]:
    """Get current time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"timezone": timezone, "datetime": now.isoformat(), "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}


def create_task(title: str, description: str = "", priority: str = "medium", due_date: str | None = None) -> dict[str, Any]:
    """Simulated task creation."""
    return {"id": f"task_{hash(title) % 10000}", "title": title, "description": description, "priority": priority, "due_date": due_date, "status": "created"}


TOOL_HANDLERS = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "create_task": create_task,
}
