from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv(override=True)


client = genai.Client()


def generate_embeddings_google(text: str):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )

    return result.embeddings[0].values


if __name__ == "__main__":
    text = ["Hello, world!", "Hi, world!"]
    embeddings = generate_embeddings_google(text[0])

    print(embeddings)
