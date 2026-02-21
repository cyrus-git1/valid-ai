import os
from pathlib import Path
from typing import List
from openai import OpenAI
import dotenv

dotenv.load_dotenv()


def find_env(start: Path) -> Path:
    """Find .env file by walking up from the given path."""
    for p in [start, *start.parents]:
        candidate = p / ".env"
        if candidate.exists():
            return candidate
    return Path("")


def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use (default: text-embedding-3-small)
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    client = get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

