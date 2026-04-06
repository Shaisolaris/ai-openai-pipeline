"""
OpenAI Pipeline Demo — runs without API key using mock responses.
Shows how the pipeline handles chat, function calling, embeddings, and structured output.
Run: python examples/demo.py

For real API calls:
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
  python examples/01_basic_chat.py
"""
import json, math, random, hashlib, datetime

class MockOpenAI:
    """Simulates OpenAI API responses to demonstrate pipeline behavior."""
    
    def chat(self, messages: list, model: str = "gpt-4o", **kwargs) -> dict:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return {
            "id": f"chatcmpl-{hashlib.md5(user_msg.encode()).hexdigest()[:12]}",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": self._generate_response(user_msg)}}],
            "usage": {"prompt_tokens": len(user_msg.split()) * 2, "completion_tokens": 50, "total_tokens": len(user_msg.split()) * 2 + 50},
        }
    
    def function_call(self, messages: list, tools: list) -> dict:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "weather" in user_msg.lower():
            return {"tool": "get_weather", "args": {"city": "San Francisco"}, "result": {"temp": 72, "condition": "sunny", "humidity": 45}}
        elif "calculate" in user_msg.lower() or "tip" in user_msg.lower():
            return {"tool": "calculate", "args": {"expression": "85.50 * 0.15"}, "result": {"value": 12.825}}
        return {"tool": "none", "result": "No tool needed"}
    
    def embeddings(self, texts: list) -> list:
        """Generate mock embeddings (normalized random vectors)."""
        def mock_embed(text):
            random.seed(hashlib.md5(text.encode()).hexdigest())
            vec = [random.gauss(0, 1) for _ in range(8)]
            norm = math.sqrt(sum(x*x for x in vec))
            return [x/norm for x in vec]
        return [mock_embed(t) for t in texts]
    
    def structured_output(self, prompt: str, schema: dict) -> dict:
        return {
            "name": "John Smith",
            "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
            "experience_years": 8,
            "summary": "Senior Python developer specializing in scalable microservices."
        }
    
    def _generate_response(self, prompt: str) -> str:
        if "error" in prompt.lower() or "python" in prompt.lower():
            return "Use try/except blocks for error handling in Python. Custom exceptions help organize error types. Always log errors with context for debugging."
        elif "api" in prompt.lower():
            return "For REST APIs, use FastAPI with Pydantic models for validation. Add middleware for auth, CORS, and rate limiting."
        return f"Here's my analysis of your question about: {prompt[:50]}..."

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

def main():
    print("🤖 OpenAI Pipeline Demo (mock mode, no API key needed)")
    print("=" * 55)
    
    client = MockOpenAI()
    
    # 1. Chat completion
    print("\n📝 1. Chat Completion")
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What's the best way to handle errors in Python?"},
    ]
    result = client.chat(messages)
    print(f"   Model: {result['model']}")
    print(f"   Response: {result['choices'][0]['message']['content']}")
    print(f"   Tokens: {result['usage']}")
    
    # 2. Function calling
    print("\n🔧 2. Function Calling")
    tools = [{"name": "get_weather"}, {"name": "calculate"}]
    result = client.function_call(messages + [{"role": "user", "content": "What's the weather in San Francisco?"}], tools)
    print(f"   Tool: {result['tool']}({json.dumps(result['args'])})")
    print(f"   Result: {json.dumps(result['result'])}")
    
    result = client.function_call(messages + [{"role": "user", "content": "Calculate 15% tip on $85.50"}], tools)
    print(f"   Tool: {result['tool']}({json.dumps(result['args'])})")
    print(f"   Result: ${result['result']['value']:.2f}")
    
    # 3. Embeddings + similarity
    print("\n📊 3. Embeddings & Semantic Similarity")
    texts = ["The cat sat on the mat", "A feline rested on the rug", "The stock market crashed", "Python is a language"]
    embeddings = client.embeddings(texts)
    
    print(f"   Embedding dimension: {len(embeddings[0])}")
    print(f"   Similarity matrix:")
    for i, a in enumerate(texts):
        scores = [cosine_similarity(embeddings[i], embeddings[j]) for j in range(len(texts))]
        print(f"     [{i}] {a[:30]:>30s}  {' '.join(f'{s:.2f}' for s in scores)}")
    
    # 4. Structured output
    print("\n📋 4. Structured Output (JSON)")
    result = client.structured_output(
        "John is a senior Python developer with 8 years of experience in FastAPI, PostgreSQL, Redis, and Docker.",
        {"type": "object", "properties": {"name": {}, "skills": {}, "experience_years": {}, "summary": {}}}
    )
    print(f"   {json.dumps(result, indent=2)}")
    
    print("\n✅ All 4 pipeline features demonstrated")
    print("   For real API calls: pip install -r requirements.txt && export OPENAI_API_KEY=sk-...")

if __name__ == "__main__":
    main()
