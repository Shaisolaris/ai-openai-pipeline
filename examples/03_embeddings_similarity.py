"""Text embeddings and semantic similarity."""
import asyncio
from pipeline.core import create_embedding, create_embeddings, cosine_similarity

async def main():
    # Compare semantic similarity between texts
    texts = [
        "The cat sat on the mat",
        "A feline was resting on the rug",
        "The stock market crashed yesterday",
        "Python is a programming language",
    ]

    embeddings = await create_embeddings(texts)

    print("Semantic Similarity Matrix:")
    print(f"{'':>40}", end="")
    for i in range(len(texts)):
        print(f"  [{i}]", end="")
    print()

    for i, text_a in enumerate(texts):
        print(f"[{i}] {text_a[:37]:>37}  ", end="")
        for j, text_b in enumerate(texts):
            score = cosine_similarity(embeddings[i], embeddings[j])
            print(f" {score:.2f}", end="")
        print()

if __name__ == "__main__":
    asyncio.run(main())
