from google import genai
from google.genai import types
from dotenv import load_dotenv
import ollama

load_dotenv(override=True)

# Lazy import for SentenceTransformer to avoid loading if not needed
_sentence_transformer_model = None


def generate_embeddings_google(text: str) -> list:
    client = genai.Client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    return result.embeddings[0].values


def generate_local_embeddings(text: str, model: str = "embeddinggemma") -> list:
    embeddings = ollama.embed(model=model, input=text)
    return list(embeddings.embeddings[0])


def generate_openai_embeddings(
    text: str,
    model: str = "text-embedding-3-small",
    base_url: str = "",
    api_key: str = "",
) -> list:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def generate_sentence_transformer_embeddings(
    text: str, model_name: str = "nomic-ai/nomic-embed-text-v1.5"
) -> list:
    """
    Generate embeddings using SentenceTransformer models.

    Args:
        text: Text to embed
        model_name: SentenceTransformer model name (default: nomic-ai/nomic-embed-text-v1.5)

    Returns:
        List of embedding values
    """
    global _sentence_transformer_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for SentenceTransformer embeddings. "
            "Install it with: pip install sentence-transformers"
        )

    # Lazy load the model (cache it globally)
    if (
        _sentence_transformer_model is None
        or _sentence_transformer_model.model_name != model_name
    ):
        _sentence_transformer_model = SentenceTransformer(
            model_name, trust_remote_code=True
        )
        _sentence_transformer_model.model_name = (
            model_name  # Store model name for cache checking
        )

    # Generate embeddings
    embeddings = _sentence_transformer_model.encode([text])
    return embeddings[0].tolist()


if __name__ == "__main__":
    text = ["Hello, world!", "Hi, world!"]
    embeddings = generate_embeddings_google(text[-1])
    embeddings = generate_local_embeddings(text[-1])
