"""Basic chat completion example."""
import asyncio
from pipeline.core import ConversationMemory, PipelineConfig, chat

async def main():
    memory = ConversationMemory(system_prompt="You are a helpful coding assistant.")
    config = PipelineConfig(model="gpt-4o", temperature=0.7)

    # Multi-turn conversation
    questions = [
        "What's the best way to handle errors in Python?",
        "Can you show me an example with custom exceptions?",
        "How would I add logging to that?",
    ]

    for q in questions:
        memory.add_user(q)
        response = await chat(memory.get_messages(), config)
        memory.add_assistant(response)
        print(f"\n👤 {q}\n🤖 {response[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
