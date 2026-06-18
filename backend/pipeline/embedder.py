from typing import List
from google import genai

EMBED_MODEL = "models/gemini-embedding-001"

def get_client():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of document chunks."""
    client = get_client()
    results = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        results.append(response.embeddings[0].values)
    return results


def embed_query(query: str) -> List[float]:
    """Embed a single user query."""
    client = get_client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    )
    return response.embeddings[0].values