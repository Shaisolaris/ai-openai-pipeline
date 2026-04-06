"""Function calling with automatic tool execution."""
import asyncio
from pipeline.core import ConversationMemory, PipelineConfig, chat
from tools.definitions import TOOLS, TOOL_HANDLERS

async def main():
    memory = ConversationMemory(system_prompt="You are a helpful assistant with access to tools.")
    config = PipelineConfig(
        model="gpt-4o",
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
    )

    queries = [
        "What's the weather in San Francisco?",
        "Calculate 15% tip on a $85.50 bill",
        "What time is it right now?",
        "Create a task to review the Q3 report by Friday",
    ]

    for q in queries:
        memory.add_user(q)
        response = await chat(memory.get_messages(), config)
        memory.add_assistant(response)
        print(f"\n👤 {q}\n🤖 {response}")

if __name__ == "__main__":
    asyncio.run(main())
