import json
import os
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types


KNOWLEDGE_BASE_PATH = Path("knowledge_base.json")
MODEL_NAME = "gemini-embedding-001"

TEST_QUERIES = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?",
    "what's the wifi password?",
]


def load_knowledge_base(file_path: Path) -> list[dict]:
    """Load passages from the knowledge base JSON file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def embed_passages(
    client: genai.Client,
    knowledge_base: list[dict],
) -> list[dict]:
    """Create one embedding for each knowledge-base passage."""

    passage_texts = [
        passage["text"]
        for passage in knowledge_base
    ]

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=passage_texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "The embedding API returned no passage embeddings."
        )

    if len(response.embeddings) != len(knowledge_base):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of passages."
        )

    embedded_passages = []

    for passage, embedding in zip(
        knowledge_base,
        response.embeddings,
    ):
        embedded_passages.append(
            {
                "id": passage["id"],
                "source": passage["source"],
                "text": passage["text"],
                "embedding": np.array(
                    embedding.values,
                    dtype=np.float32,
                ),
            }
        )

    return embedded_passages


def embed_query(
    client: genai.Client,
    query: str,
) -> np.ndarray:
    """Turn one search query into an embedding vector."""

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "The embedding API returned no query embedding."
        )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32,
    )


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Compute cosine similarity manually with NumPy."""

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity = dot_product / (
        magnitude_a * magnitude_b
    )

    return float(similarity)


def search_passages(
    query_embedding: np.ndarray,
    embedded_passages: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Rank all passages and return the top-k matches."""

    results = []

    for passage in embedded_passages:
        score = cosine_similarity(
            query_embedding,
            passage["embedding"],
        )

        results.append(
            {
                "id": passage["id"],
                "source": passage["source"],
                "text": passage["text"],
                "score": score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


def print_results(
    query: str,
    results: list[dict],
) -> None:
    """Print the ranked search results clearly."""

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result['id']} "
            f"({result['source']})"
        )
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set in the terminal."
        )

    client = genai.Client(api_key=api_key)

    knowledge_base = load_knowledge_base(
        KNOWLEDGE_BASE_PATH
    )

    print(f"Loaded {len(knowledge_base)} passages.")

    # Passage embeddings are created once and reused for all queries.
    embedded_passages = embed_passages(
        client,
        knowledge_base,
    )

    print(
        f"Created {len(embedded_passages)} "
        "passage embeddings."
    )

    print(
        "Embedding dimension:",
        len(embedded_passages[0]["embedding"]),
    )

    for query in TEST_QUERIES:
        query_embedding = embed_query(
            client,
            query,
        )

        top_results = search_passages(
            query_embedding,
            embedded_passages,
            top_k=3,
        )

        print_results(
            query,
            top_results,
        )


if __name__ == "__main__":
    main()

