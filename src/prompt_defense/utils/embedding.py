from google import genai
from google.genai import types
from dotenv import load_dotenv
import ollama

load_dotenv(override=True)


client = genai.Client()


def generate_embeddings_google(text: str) -> list:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    return result.embeddings[0].values


def generate_local_embeddings(text: str, model: str = "embeddinggemma") -> list:
    embeddings = ollama.embed(model=model, input=text)
    return list(embeddings.embeddings[0])


if __name__ == "__main__":
    text = ["Hello, world!", "Hi, world!"]
    embeddings = generate_embeddings_google(text[-1])
    embeddings = generate_local_embeddings(text[-1])
