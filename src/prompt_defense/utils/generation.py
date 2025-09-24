from tenacity import retry, wait_random_exponential

from prompt_defense.utils.embedding import generate_embeddings_google


# generate
@retry(wait=wait_random_exponential(min=1, max=10))
def generate_response_with_retry(agent, prompt):
    return agent.run_sync([prompt])


@retry(wait=wait_random_exponential(min=1, max=10))
def embeddings_with_retry(text):
    return generate_embeddings_google(text)
