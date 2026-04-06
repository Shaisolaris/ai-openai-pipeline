"""Structured JSON output with schema enforcement."""
import asyncio
import json
from pipeline.core import structured_output

async def main():
    # Extract structured data from natural language
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience_years": {"type": "number"},
            "summary": {"type": "string"},
        }
    }

    result = await structured_output(
        "John is a senior Python developer with 8 years of experience. "
        "He's skilled in FastAPI, Django, PostgreSQL, Redis, and Docker. "
        "He specializes in building scalable microservices.",
        schema=schema,
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
