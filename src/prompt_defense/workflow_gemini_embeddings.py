from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.utils.calculate_similarity import calculate_similarity
from prompt_defense.attack_prompts.gemini_generated import (
    role_playing_attack,
    repetetion,
    context_shifting,
    direct_query,
    persona,
)
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from prompt_defense.utils.embedding import generate_embeddings_google
from loguru import logger


def main():
    gemini_agent = init_gemini_agent(system_prompt=SYSTEM_PROMPT)

    # forward pass all of the prompts
    prompts = [role_playing_attack, repetetion, context_shifting, direct_query, persona]

    # first embedd the system prompt
    system_prompt_embeddings = generate_embeddings_google(SYSTEM_PROMPT)

    logger.info("System prompt embeddings generated.")

    # now forward pass all of the prompts and calculate similarity
    results = []
    for prompt in prompts:
        response = gemini_agent.run_sync([prompt])
        response_embeddings = generate_embeddings_google(response.output)
        results.append((prompt, response.output, response_embeddings))

    logger.info("All prompts processed and embeddings generated.")
    # calculate similarity

    all_embeddings = [system_prompt_embeddings] + [r[2] for r in results]
    similarity_matrix = calculate_similarity(all_embeddings)
    print("Similarity Matrix:")
    print(similarity_matrix)
    logger.info("Similarity matrix calculated.")

    # please sort the results by similarity to the system prompt and then print the pairs
    sorted_results = sorted(
        results, key=lambda x: similarity_matrix[0][results.index(x) + 1], reverse=True
    )
    for i, (prompt, response, embeddings) in enumerate(sorted_results):
        print(f"\nAttack Prompt {i + 1}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Similarity to System Prompt: {similarity_matrix[0][i + 1]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
