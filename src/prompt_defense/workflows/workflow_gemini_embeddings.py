from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.utils.calculate_similarity import calculate_similarity
from prompt_defense.attack_prompts.generated import prompts
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from loguru import logger
from prompt_defense.utils.generation import (
    generate_response_with_retry,
    embeddings_with_retry,
)
from prompt_defense.utils.embedding import generate_embeddings_google
from prompt_defense.utils.json_storage import (
    save_workflow_results,
    format_results_for_storage,
    create_results_summary,
)
from tqdm import tqdm


def main():
    gemini_agent = init_gemini_agent(system_prompt=SYSTEM_PROMPT)

    # forward pass all of the prompts (imported from manually.py)

    # first embedd the system prompt
    system_prompt_embeddings = generate_embeddings_google(SYSTEM_PROMPT)

    logger.info("System prompt embeddings generated.")

    # now forward pass all of the prompts and calculate similarity
    results = []
    for prompt in tqdm(prompts):
        response = generate_response_with_retry(gemini_agent, prompt)
        response_embeddings = embeddings_with_retry(response.output)

        results.append((prompt, response.output, response_embeddings))

    logger.info("All prompts processed and embeddings generated.")
    # calculate similarity

    all_embeddings = [system_prompt_embeddings] + [r[2] for r in results]
    similarity_matrix = calculate_similarity(all_embeddings)
    logger.info("Similarity Matrix:")
    logger.info(similarity_matrix)
    logger.info("Similarity matrix calculated.")

    # please sort the results by similarity to the system prompt and then logger.info the pairs
    sorted_results = sorted(
        results, key=lambda x: similarity_matrix[0][results.index(x) + 1], reverse=True
    )
    for i, (prompt, response, embeddings) in enumerate(sorted_results):
        logger.info(f"\nAttack Prompt {i + 1}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")
        logger.info(f"Similarity to System Prompt: {similarity_matrix[0][i + 1]}")
        logger.info("-" * 50)

    # Save results to JSON
    logger.info("Saving results to JSON...")

    # Extract data for storage
    prompts_list = [r[0] for r in results]
    responses_list = [r[1] for r in results]

    # Format results for storage
    formatted_results = format_results_for_storage(
        prompts=prompts_list,
        responses=responses_list,
        similarity_matrix=similarity_matrix.tolist()
        if hasattr(similarity_matrix, "tolist")
        else similarity_matrix,
        system_prompt=SYSTEM_PROMPT,
        additional_data={
            "model_type": "gemini",
            "embedding_model": "google_embeddings",
            "total_prompts_processed": len(prompts),
        },
    )

    # Save to JSON file
    saved_path = save_workflow_results(
        results_data=formatted_results, workflow_name="gemini_embeddings"
    )

    # Create and display summary
    summary = create_results_summary(formatted_results)
    logger.info("\nResults Summary:")
    logger.info(f"- Total prompts: {summary['total_prompts']}")
    logger.info(f"- Has similarity scores: {summary['has_similarity_scores']}")
    if "similarity_stats" in summary:
        stats = summary["similarity_stats"]
        logger.info(f"- Similarity range: {stats['min']:.4f} - {stats['max']:.4f}")
        logger.info(f"- Mean similarity: {stats['mean']:.4f}")

    logger.info(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()
